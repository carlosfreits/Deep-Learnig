# %% [markdown]
# # Questão 1: Rede MLP para regressão
#
# Nesta questão usamos uma rede Perceptron de Múltiplas Camadas (MLP) em dois problemas de regressão, ou seja, problemas em que a rede precisa prever um número.
#
# - Item a: aproximar a função f(x) = 3 sen(2x) + 2 cos(5x) + 0,5x², com x entre 0 e 10.
# - Item b: prever a progressão do diabetes de um paciente a partir de 10 informações clínicas (base Diabetes do scikit-learn).
#
# Nos dois itens separamos os dados em três partes:
# - treino: os dados que a rede usa para aprender;
# - validação: dados que a rede não usa para aprender, mas que olhamos a cada época para ver se ela está melhorando de verdade;
# - teste: dados guardados até o final, usados uma única vez para medir o desempenho.

# %% [markdown]
# ## Bibliotecas

# %%
import copy
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# usa a placa de vídeo se tiver, senão usa o processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rodando em:", device)

# pasta onde as figuras são salvas
PASTA_FIGURAS = Path("../figures/q1")
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Funções
#
# Aqui ficam todas as funções que usamos no notebook. Cada uma tem uma explicação curta logo acima.

# %% [markdown]
# ### fixar_semente
# A rede começa com pesos sorteados e os dados também são embaralhados de forma aleatória. Esta função fixa a "semente" do sorteio para que o resultado seja o mesmo toda vez que o notebook for executado.

# %%
def fixar_semente(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

# %% [markdown]
# ### f
# A função que a rede vai tentar aprender no item a.

# %%
def f(x):
    return 3 * np.sin(2 * x) + 2 * np.cos(5 * x) + 0.5 * x**2

# %% [markdown]
# ### dividir_dados
# Separa os dados em 70% para treino, 15% para validação e 15% para teste. Primeiro tiramos 30% dos dados e depois dividimos esses 30% ao meio.

# %%
def dividir_dados(X, y, semente=42):
    X_treino, X_resto, y_treino, y_resto = train_test_split(X, y, test_size=0.30, random_state=semente)
    X_val, X_teste, y_val, y_teste = train_test_split(X_resto, y_resto, test_size=0.50, random_state=semente)
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste

# %% [markdown]
# ### criar_loader
# O PyTorch treina a rede em pequenos lotes de exemplos. Esta função transforma os nossos arrays em um "loader", que entrega os dados em lotes do tamanho que escolhermos. No treino os lotes são embaralhados a cada época.

# %%
def criar_loader(X, y, tamanho_lote, embaralhar):
    dados = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dados, batch_size=tamanho_lote, shuffle=embaralhar)

# %% [markdown]
# ### MLP
# A nossa rede neural. Ela recebe a lista de camadas escondidas (por exemplo, [128, 128, 128] são três camadas com 128 neurônios cada), a função de ativação e o dropout.
#
# O dropout desliga alguns neurônios ao acaso durante o treino. Isso obriga a rede a não depender demais de nenhum neurônio específico e ajuda a evitar overfitting. A última camada não tem ativação, porque a saída é um número qualquer.

# %%
class MLP(nn.Module):
    def __init__(self, entradas, camadas, ativacao=nn.Tanh, dropout=0.0):
        super().__init__()
        partes = []
        anterior = entradas
        for neuronios in camadas:
            partes.append(nn.Linear(anterior, neuronios))
            partes.append(ativacao())
            if dropout > 0:
                partes.append(nn.Dropout(dropout))
            anterior = neuronios
        partes.append(nn.Linear(anterior, 1))
        self.rede = nn.Sequential(*partes)

    def forward(self, x):
        return self.rede(x)

# %% [markdown]
# ### rodar_epoca
# Passa uma vez por todos os lotes de um loader e devolve o erro médio (MSE).
# - Se receber um otimizador, a rede aprende: calcula o erro, faz o backpropagation e ajusta os pesos.
# - Se não receber, a rede só é avaliada, sem mudar nenhum peso.

# %%
def rodar_epoca(modelo, loader, otimizador=None):
    funcao_erro = nn.MSELoss()
    treinando = otimizador is not None
    modelo.train(treinando)
    erro_total = 0.0
    with torch.set_grad_enabled(treinando):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            erro = funcao_erro(modelo(x), y)
            if treinando:
                otimizador.zero_grad()
                erro.backward()
                otimizador.step()
            erro_total += erro.item() * len(x)
    return erro_total / len(loader.dataset)

# %% [markdown]
# ### treinar
# Treina a rede por várias épocas usando o otimizador Adam. Em cada época:
# 1. a rede aprende com os dados de treino;
# 2. medimos o erro na validação;
# 3. guardamos os pesos da época com o menor erro de validação.
#
# Opções:
# - `paciencia`: se o erro de validação passar esse número de épocas sem melhorar, o treino para (early stopping);
# - `reduzir_lr`: diminui a taxa de aprendizado pela metade quando a validação para de melhorar, o que ajuda a rede a fazer ajustes mais finos;
# - `voltar_melhor`: no final, volta para os pesos da melhor época.
#
# A barra de progresso mostra a época atual, o erro de treino, o erro de validação e a melhor época até agora.

# %%
def treinar(modelo, loader_treino, loader_val, epocas, lr=1e-3, weight_decay=0.0,
            paciencia=None, reduzir_lr=False, voltar_melhor=True, nome="Treinando"):
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    agendador = None
    if reduzir_lr:
        agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, factor=0.5, patience=60, min_lr=1e-5)

    historico = {"treino": [], "val": [], "melhor_epoca": 0}
    menor_erro_val = float("inf")
    melhores_pesos = None
    epocas_sem_melhorar = 0

    barra = tqdm(range(1, epocas + 1), desc=nome, unit="época", ncols=110, mininterval=2)
    for epoca in barra:
        erro_treino = rodar_epoca(modelo, loader_treino, otimizador)
        erro_val = rodar_epoca(modelo, loader_val)
        historico["treino"].append(erro_treino)
        historico["val"].append(erro_val)

        if agendador is not None:
            agendador.step(erro_val)

        if erro_val < menor_erro_val:
            menor_erro_val = erro_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            historico["melhor_epoca"] = epoca
            epocas_sem_melhorar = 0
        else:
            epocas_sem_melhorar += 1

        barra.set_postfix(treino=f"{erro_treino:.2e}", val=f"{erro_val:.2e}", melhor=historico["melhor_epoca"], refresh=False)

        if paciencia is not None and epocas_sem_melhorar >= paciencia:
            barra.close()
            print(f"Parou na época {epoca}: a validação ficou {paciencia} épocas sem melhorar.")
            break

    # se o erro virou NaN (treino "explodiu"), nenhuma época foi guardada
    if melhores_pesos is None:
        raise RuntimeError("O erro de validação nunca foi um número válido (NaN). Tente uma taxa de aprendizado menor.")

    historico["voltou_melhor"] = voltar_melhor
    if voltar_melhor:
        modelo.load_state_dict(melhores_pesos)
    print(f"Melhor época: {historico['melhor_epoca']} (erro de validação = {menor_erro_val:.2e})")
    return historico

# %% [markdown]
# ### prever
# Usa a rede treinada para prever novos valores. Como a rede foi treinada com dados normalizados, a função normaliza a entrada, faz a previsão e depois desfaz a normalização da saída, devolvendo o resultado na escala original.

# %%
def prever(modelo, X, normalizador_x, normalizador_y):
    modelo.eval()
    with torch.no_grad():
        x = torch.tensor(normalizador_x.transform(X), dtype=torch.float32, device=device)
        y_normalizado = modelo(x).cpu().numpy()
    return normalizador_y.inverse_transform(y_normalizado)

# %% [markdown]
# ### calcular_metricas
# Calcula as métricas de erro pedidas na questão:
# - MAE (erro absoluto médio): em média, quanto a previsão erra, na mesma unidade do alvo;
# - MSE (erro quadrático médio): parecido, mas eleva os erros ao quadrado, então erros grandes pesam muito mais;
# - RMSE: a raiz do MSE, que volta para a unidade original do alvo;
# - R²: quanto da variação do alvo a rede consegue explicar (1 é perfeito, 0 é o mesmo que chutar sempre a média).

# %%
def calcular_metricas(y_real, y_previsto):
    mse = mean_squared_error(y_real, y_previsto)
    return {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R²": r2_score(y_real, y_previsto),
    }

# %% [markdown]
# ### tabela_overfitting
# Overfitting é quando a rede "decora" os dados de treino e erra muito em dados que nunca viu. Para verificar isso, calculamos as métricas em treino, validação e teste e colocamos lado a lado. Se o erro no treino for muito menor que nos outros dois, a rede decorou.

# %%
def tabela_overfitting(modelo, conjuntos, normalizador_x, normalizador_y):
    linhas = {}
    for nome, (X, y) in conjuntos.items():
        linhas[nome] = calcular_metricas(y, prever(modelo, X, normalizador_x, normalizador_y))
    return pd.DataFrame(linhas).T

# %% [markdown]
# ### grafico_curvas
# Desenha as curvas de erro de treino e de validação ao longo das épocas. A linha cinza marca a época com o menor erro de validação:
# - tracejada: é a época cujos pesos a rede usa no final;
# - pontilhada: a rede não voltou para essa época e usa os pesos da última época.

# %%
def grafico_curvas(historico, titulo, escala_log=False, arquivo=None):
    epocas = range(1, len(historico["treino"]) + 1)
    melhor = historico["melhor_epoca"]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epocas, historico["treino"], label="treino")
    plt.plot(epocas, historico["val"], label="validação")
    if historico["voltou_melhor"]:
        plt.axvline(melhor, color="gray", linestyle="--", label=f"melhor época ({melhor}), pesos usados")
    else:
        plt.axvline(melhor, color="gray", linestyle=":", label=f"menor erro de validação ({melhor}), pesos não usados")
    if escala_log:
        plt.yscale("log")
    plt.xlabel("época")
    plt.ylabel("erro (MSE nos dados normalizados)")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    if arquivo:
        plt.savefig(PASTA_FIGURAS / arquivo, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### criar_modelo_b e treinar_modelo_b
# A rede e o jeito de treinar do item b ficam em funções porque são usados em dois lugares: no treino principal e na validação cruzada. Assim, se mudarmos algum valor (número de neurônios, dropout, taxa de aprendizado...), a mudança vale para os dois.
#
# - `criar_modelo_b`: rede com 10 entradas, camadas de 32 e 16 neurônios, ReLU e dropout de 20%.
# - `treinar_modelo_b`: até 1500 épocas, taxa de aprendizado 0,001, weight decay 0,001 e parada se a validação ficar 100 épocas sem melhorar.

# %%
def criar_modelo_b():
    fixar_semente()
    return MLP(entradas=10, camadas=[32, 16], ativacao=nn.ReLU, dropout=0.2).to(device)


def treinar_modelo_b(modelo, loader_treino, loader_val, nome="Item b"):
    return treinar(modelo, loader_treino, loader_val, epocas=1500, lr=1e-3,
                   weight_decay=1e-3, paciencia=100, nome=nome)

# %% [markdown]
# ### avaliar_parte
# Usada na validação cruzada. Recebe os dados de treino e de teste de uma das partes e faz o mesmo caminho do item b: separa um pedaço do treino para validação, normaliza, cria os loaders, treina a MLP e também uma regressão linear. Devolve as métricas dos dois modelos no teste dessa parte.

# %%
def avaliar_parte(X_treino, y_treino, X_teste, y_teste, numero_parte):
    X_tr, X_va, y_tr, y_va = train_test_split(X_treino, y_treino, test_size=0.15, random_state=numero_parte)

    norm_X = StandardScaler().fit(X_tr)
    norm_y = StandardScaler().fit(y_tr)
    loader_tr = criar_loader(norm_X.transform(X_tr), norm_y.transform(y_tr), 32, True)
    loader_va = criar_loader(norm_X.transform(X_va), norm_y.transform(y_va), 256, False)

    modelo = criar_modelo_b()
    treinar_modelo_b(modelo, loader_tr, loader_va, nome=f"Parte {numero_parte}")
    linear = LinearRegression().fit(norm_X.transform(X_tr), y_tr)

    metricas_mlp = calcular_metricas(y_teste, prever(modelo, X_teste, norm_X, norm_y))
    metricas_linear = calcular_metricas(y_teste, linear.predict(norm_X.transform(X_teste)))
    return [
        {"parte": numero_parte, "modelo": "MLP", **metricas_mlp},
        {"parte": numero_parte, "modelo": "Regressão linear", **metricas_linear},
    ]

# %% [markdown]
# ---
# # Item a: aproximação da função f(x)

# %% [markdown]
# ## Gerando os dados
# Sorteamos 2000 valores de x entre 0 e 10 e calculamos f(x) para cada um. O x é a entrada da rede e o f(x) é a saída que ela deve aprender a dar.

# %%
fixar_semente()
x = np.random.uniform(0, 10, 2000).reshape(-1, 1)
y = f(x)
print("Quantidade de pontos:", len(x))
print(f"f(x) vai de {y.min():.2f} até {y.max():.2f}")

# %% [markdown]
# ## Dividindo em treino, validação e teste

# %%
x_treino, x_val, x_teste, y_treino, y_val, y_teste = dividir_dados(x, y)
print("Treino:", len(x_treino))
print("Validação:", len(x_val))
print("Teste:", len(x_teste))

# %% [markdown]
# ## Normalizando
# O f(x) vai de cerca de -2 até 55. Valores grandes assim deixam o treino da rede instável. Por isso deixamos x e f(x) com média 0 e desvio padrão 1.
#
# Importante: a média e o desvio são calculados só com os dados de treino. Assim a rede não "espia" nada da validação nem do teste.

# %%
normalizador_x = StandardScaler().fit(x_treino)
normalizador_y = StandardScaler().fit(y_treino)

x_treino_n = normalizador_x.transform(x_treino)
x_val_n = normalizador_x.transform(x_val)
y_treino_n = normalizador_y.transform(y_treino)
y_val_n = normalizador_y.transform(y_val)

# %% [markdown]
# ## Criando os loaders
# No treino usamos lotes de 64 pontos. Na validação não precisamos de lotes pequenos, porque a rede não aprende com eles.

# %%
loader_treino = criar_loader(x_treino_n, y_treino_n, tamanho_lote=64, embaralhar=True)
loader_val = criar_loader(x_val_n, y_val_n, tamanho_lote=512, embaralhar=False)

# %% [markdown]
# ## Gráfico dos dados

# %%
x_linha = np.linspace(0, 10, 1000).reshape(-1, 1)

plt.figure(figsize=(10, 4))
plt.plot(x_linha, f(x_linha), color="black", label="f(x)")
plt.scatter(x_treino, y_treino, s=6, alpha=0.5, label="treino")
plt.scatter(x_val, y_val, s=6, label="validação")
plt.scatter(x_teste, y_teste, s=6, label="teste")
plt.title("Função f(x) e pontos sorteados")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PASTA_FIGURAS / "1a_dados.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - A função tem duas partes: uma parábola que cresce (o 0,5x²) e ondas por cima dela (o seno e o cosseno). A rede precisa aprender as duas coisas ao mesmo tempo.
# - Os pontos de treino, validação e teste estão espalhados por todo o intervalo de 0 a 10. Então a rede vai ser testada em regiões que ela também viu no treino, só que em pontos diferentes.

# %% [markdown]
# ## Criando a rede
# Usamos 3 camadas escondidas com 128 neurônios cada e ativação Tanh. A Tanh é uma curva suave, o que combina com uma função suave como a f(x). Com ReLU a saída da rede é formada por pedaços de retas. Mais abaixo, na seção "Tanh ou ReLU?", treinamos a mesma rede com ReLU para conferir essa escolha.

# %%
fixar_semente()
modelo_a = MLP(entradas=1, camadas=[128, 128, 128], ativacao=nn.Tanh).to(device)
print(modelo_a)
print("Total de pesos treináveis:", sum(p.numel() for p in modelo_a.parameters()))

# %% [markdown]
# ## Treinando
# - Até 2000 épocas, taxa de aprendizado inicial de 0,003.
# - A taxa de aprendizado é reduzida quando a validação para de melhorar.
# - O treino para sozinho se a validação ficar 300 épocas sem melhorar.

# %%
historico_a = treinar(modelo_a, loader_treino, loader_val, epocas=2000, lr=3e-3,
                      paciencia=300, reduzir_lr=True, nome="Item a")

# %% [markdown]
# ## Curvas de erro de treino e de validação

# %%
grafico_curvas(historico_a, "Item a: erro durante o treino", escala_log=True, arquivo="1a_curvas.png")

# %% [markdown]
# O que observar:
# - O erro cai de cerca de 0,1 para menos de 0,000001 (o gráfico está em escala logarítmica, então cada linha de grade é 10 vezes menor que a anterior).
# - As curvas de treino e de validação ficam praticamente uma em cima da outra durante todo o treino. Isso mostra que a rede não está decorando o treino: ela erra igual em pontos que nunca viu.
# - Os picos que aparecem (por exemplo, perto da época 250) são momentos em que o passo de aprendizado foi grande demais. Com a redução automática da taxa de aprendizado eles vão ficando menores.
# - A melhor época foi a 1991, quase no fim. O erro ainda estava caindo devagar, então com mais épocas daria para melhorar um pouco mais.

# %% [markdown]
# ## Métricas no conjunto de teste
# Valores na escala original de f(x).

# %%
y_previsto_teste = prever(modelo_a, x_teste, normalizador_x, normalizador_y)
metricas_a = calcular_metricas(y_teste, y_previsto_teste)
pd.DataFrame([metricas_a], index=["teste"]).round(5)

# %% [markdown]
# ## Função real e função aproximada
# Aqui pedimos para a rede prever 1000 pontos igualmente espaçados entre 0 e 10 e desenhamos junto da função real. Embaixo mostramos o erro (real menos previsto) em cada ponto.

# %%
y_linha_previsto = prever(modelo_a, x_linha, normalizador_x, normalizador_y)
erro_linha = f(x_linha) - y_linha_previsto

fig, (cima, baixo) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
cima.plot(x_linha, f(x_linha), color="black", linewidth=2, label="f(x) real")
cima.plot(x_linha, y_linha_previsto, "r--", label="aproximada pela rede")
cima.set_ylabel("f(x)")
cima.set_title("Função real e função aproximada")
cima.legend()
cima.grid(alpha=0.3)
baixo.plot(x_linha, erro_linha, color="purple")
baixo.axhline(0, color="black", linewidth=0.8)
baixo.set_xlabel("x")
baixo.set_ylabel("erro")
baixo.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1a_real_vs_aprox.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - A curva vermelha (rede) fica em cima da preta (real) em todo o intervalo de 0 a 10. Ela acompanha tanto a parábola quanto as ondas.
# - No gráfico de baixo, o erro fica quase sempre entre -0,03 e +0,03. Para uma função que vai de -2 a 55, isso é muito pouco.
# - O maior erro, de cerca de -0,2, aparece logo em x = 0. Na borda do intervalo só existem pontos de um lado, então a rede tem menos informação ali.

# %% [markdown]
# ## Curvas real e predita no conjunto de teste
# Agora só com os pontos de teste, que a rede nunca usou. À esquerda as duas curvas; à direita cada ponto é um exemplo de teste, com o valor real no eixo x e o previsto no eixo y. Se a rede acertasse tudo, todos os pontos ficariam em cima da linha tracejada.

# %%
ordem = np.argsort(x_teste[:, 0])
limites = [y_teste.min() - 2, y_teste.max() + 2]

fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.5))
esquerda.plot(x_teste[ordem], y_teste[ordem], color="black", label="real")
esquerda.plot(x_teste[ordem], y_previsto_teste[ordem], "r--", label="predita")
esquerda.set_xlabel("x")
esquerda.set_ylabel("f(x)")
esquerda.set_title("Teste: curva real e curva predita")
esquerda.legend()
esquerda.grid(alpha=0.3)
direita.scatter(y_teste, y_previsto_teste, s=10)
direita.plot(limites, limites, "k--", linewidth=1)
direita.set_xlim(limites)
direita.set_ylim(limites)
direita.set_xlabel("valor real")
direita.set_ylabel("valor previsto")
direita.set_title("Teste: real x previsto")
direita.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1a_teste_real_vs_pred.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - À esquerda, a curva predita nos pontos de teste é igual à curva real.
# - À direita, todos os pontos estão em cima da linha tracejada. Não existe nenhuma faixa de valores (baixos ou altos) em que a rede erre mais.

# %% [markdown]
# ## Verificando overfitting

# %%
conjuntos_a = {"treino": (x_treino, y_treino), "validação": (x_val, y_val), "teste": (x_teste, y_teste)}
tabela_overfitting(modelo_a, conjuntos_a, normalizador_x, normalizador_y).round(5)

# %% [markdown]
# Os erros em treino, validação e teste são praticamente iguais (RMSE de 0,0136, 0,0132 e 0,0138). Então não há overfitting no item a.
#
# Isso faz sentido: os dados não têm ruído e os três conjuntos cobrem o mesmo intervalo de 0 a 10, então aprender bem o treino é o mesmo que aprender bem a função.

# %% [markdown]
# ## Fora do intervalo de treino
# A rede só viu valores de x entre 0 e 10. Para ver o que acontece fora desse intervalo, pedimos a previsão até x = 12.

# %%
x_fora = np.linspace(0, 12, 1200).reshape(-1, 1)
y_fora_previsto = prever(modelo_a, x_fora, normalizador_x, normalizador_y)
depois_de_10 = x_fora[:, 0] > 10

plt.figure(figsize=(10, 4))
plt.plot(x_fora, f(x_fora), color="black", linewidth=2, label="f(x) real")
plt.plot(x_fora, y_fora_previsto, "r--", label="aproximada pela rede")
plt.axvspan(10, 12, color="gray", alpha=0.15, label="fora do intervalo de treino")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("A rede fora do intervalo em que foi treinada")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PASTA_FIGURAS / "1a_extrapolacao.png", bbox_inches="tight")
plt.show()

pd.DataFrame({
    "x entre 0 e 10": calcular_metricas(f(x_fora[~depois_de_10]), y_fora_previsto[~depois_de_10]),
    "x entre 10 e 12": calcular_metricas(f(x_fora[depois_de_10]), y_fora_previsto[depois_de_10]),
}).T[["MAE", "RMSE"]].round(3)

# %% [markdown]
# O que observar:
# - Até x = 10 a curva da rede fica em cima da real, como antes.
# - Logo depois de x = 10 a rede para de acompanhar a função: a previsão fica praticamente reta, perto de 58, enquanto a função real continua subindo e ondulando até 67. O erro médio (MAE) passa de 0,012 dentro do intervalo para 3,6 fora dele, cerca de 300 vezes maior.
# - Isso acontece porque a rede não aprendeu a fórmula de f(x). Ela aprendeu a ligar os valores de x que viu aos valores de f(x) correspondentes. Fora dessa região os neurônios com Tanh "saturam" (a Tanh fica presa perto de 1 ou de -1) e a saída deixa de mudar.
# - A conclusão é que uma MLP interpola bem (pontos novos dentro da região de treino), mas não extrapola (pontos fora dela).

# %% [markdown]
# ## Comparação entre Tanh e ReLU
# Na hora de criar a rede escolhemos a Tanh. Para conferir essa escolha, treinamos a mesma rede (3 camadas de 128 neurônios, mesma semente e mesmo jeito de treinar) trocando só a ativação por ReLU.

# %%
fixar_semente()
modelo_relu = MLP(entradas=1, camadas=[128, 128, 128], ativacao=nn.ReLU).to(device)
historico_relu = treinar(modelo_relu, loader_treino, loader_val, epocas=2000, lr=3e-3,
                         paciencia=300, reduzir_lr=True, nome="Item a (ReLU)")

# %%
y_linha_relu = prever(modelo_relu, x_linha, normalizador_x, normalizador_y)

fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.5))
esquerda.plot(x_linha, f(x_linha) - y_linha_previsto, label="Tanh")
esquerda.plot(x_linha, f(x_linha) - y_linha_relu, alpha=0.8, label="ReLU")
esquerda.axhline(0, color="black", linewidth=0.8)
esquerda.set_xlabel("x")
esquerda.set_ylabel("erro (real menos previsto)")
esquerda.set_title("Erro em cada ponto")
esquerda.legend()
esquerda.grid(alpha=0.3)
trecho = (x_linha[:, 0] >= 4.0) & (x_linha[:, 0] <= 4.6)
direita.plot(x_linha[trecho], f(x_linha[trecho]), color="black", linewidth=2, label="f(x) real")
direita.plot(x_linha[trecho], y_linha_previsto[trecho], "--", label="Tanh")
direita.plot(x_linha[trecho], y_linha_relu[trecho], "--", label="ReLU")
direita.set_xlabel("x")
direita.set_ylabel("f(x)")
direita.set_title("Um trecho ampliado da curva")
direita.legend()
direita.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1a_tanh_vs_relu.png", bbox_inches="tight")
plt.show()

pd.DataFrame({
    "Tanh": calcular_metricas(y_teste, y_previsto_teste),
    "ReLU": calcular_metricas(y_teste, prever(modelo_relu, x_teste, normalizador_x, normalizador_y)),
}).T.round(5)

# %% [markdown]
# O que observar:
# - As duas redes aprendem a função muito bem. No teste, o RMSE é 0,0138 com Tanh e 0,0194 com ReLU. A Tanh foi melhor, com um erro cerca de 30% menor, mas a ReLU também resolveria o problema.
# - A diferença aparece no formato do erro (gráfico da esquerda). Com Tanh o erro é uma ondulação suave. Com ReLU ele é cheio de picos: a saída de uma rede ReLU é feita de pedaços de reta, e os picos aparecem nos pontos de emenda.
# - No trecho ampliado (à direita) as duas curvas ficam em cima da real. Não dá para ver "quinas" na curva da ReLU: com 3 camadas de 128 neurônios os pedaços de reta são tantos e tão pequenos que a curva parece lisa.
# - A ideia de que a ReLU deixaria as ondas "quebradas" não se confirma para uma rede deste tamanho. O que se confirma é uma vantagem pequena da Tanh nesta função suave.
# - Nos dois casos o maior erro fica em x = 0, a borda do intervalo.
# - Esta comparação usa uma única semente, então a diferença de 30% é uma indicação e não uma prova.

# %% [markdown]
# ---
# # Item b: base Diabetes

# %% [markdown]
# ## Carregando a base
# Carregamos a versão sem normalização (`scaled=False`), para ver os valores nas unidades reais. A versão padrão do scikit-learn já vem normalizada.

# %%
diabetes = load_diabetes(as_frame=True, scaled=False)
tabela = diabetes.frame
tabela.head()

# %% [markdown]
# São 442 pacientes e 10 atributos:
# - age: idade
# - sex: sexo (1 ou 2)
# - bmi: índice de massa corporal (IMC)
# - bp: pressão arterial média
# - s1: colesterol total
# - s2: LDL (colesterol "ruim")
# - s3: HDL (colesterol "bom")
# - s4: razão entre colesterol total e HDL
# - s5: logaritmo dos triglicerídeos
# - s6: glicose
#
# A coluna target é o que queremos prever: uma medida de quanto a doença avançou um ano depois.

# %% [markdown]
# ## Análise exploratória
# ### Valores ausentes e linhas repetidas

# %%
print("Tamanho da tabela:", tabela.shape)
print("Valores ausentes:", tabela.isna().sum().sum())
print("Linhas repetidas:", tabela.duplicated().sum())

# %% [markdown]
# A base está completa: não tem nenhum valor faltando nem linha repetida. Então não precisamos preencher nem remover nada.

# %% [markdown]
# ### Estatísticas de cada coluna

# %%
tabela.describe().T.round(2)

# %% [markdown]
# As colunas estão em escalas bem diferentes. A idade vai de 19 a 79, o colesterol total (s1) vai de 97 a 301, e o s5 fica entre 3 e 6. Se a rede receber os dados assim, as colunas com números maiores dominam o aprendizado. Por isso vamos normalizar.

# %% [markdown]
# ### Distribuição de cada coluna

# %%
fig, eixos = plt.subplots(3, 4, figsize=(15, 9))
for eixo, coluna in zip(eixos.flat, tabela.columns):
    cor = "tab:red" if coluna == "target" else "tab:blue"
    sns.histplot(tabela[coluna], kde=True, ax=eixo, color=cor)
    eixo.set_title(coluna)
for eixo in eixos.flat[len(tabela.columns):]:
    eixo.axis("off")
plt.suptitle("Distribuição dos atributos (azul) e do alvo (vermelho)")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1b_histogramas.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - age, bmi, bp, s1, s2, s3, s5 e s6 têm formato parecido com um sino. bmi e s3 têm uma "cauda" para a direita, com alguns pacientes de valores bem altos.
# - sex só tem dois valores (1 e 2), com quantidades parecidas de pacientes em cada um.
# - s4 só tem alguns valores quebrados (2, 3, 4, 5...), porque é uma razão arredondada.
# - O alvo (target) tem mais pacientes com valores baixos, entre 50 e 150, e vai diminuindo até 346.

# %% [markdown]
# ### Correlação entre as colunas
# A correlação vai de -1 a 1. Perto de 1: as duas colunas sobem juntas. Perto de -1: quando uma sobe, a outra desce. Perto de 0: não tem relação linear.

# %%
correlacao = tabela.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False)
plt.title("Correlação entre as colunas")
plt.savefig(PASTA_FIGURAS / "1b_correlacao.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Os atributos que mais se relacionam com o alvo são bmi (0,59), s5 (0,57), bp (0,44) e s4 (0,43). s3 tem relação negativa (-0,39): quanto mais HDL, o colesterol bom, menor a progressão.
# - sex (0,04) e s2 (0,17) quase não têm relação direta com o alvo.
# - s1 e s2 têm correlação de 0,90, ou seja, carregam quase a mesma informação. s3 e s4 também andam juntos (-0,74). Isso é redundância. Como são só 10 atributos e a rede tem proteções contra overfitting, decidimos manter todos.

# %% [markdown]
# ### Relação dos atributos mais importantes com o alvo

# %%
mais_importantes = correlacao["target"].drop("target").abs().sort_values(ascending=False).index[:4]

fig, eixos = plt.subplots(1, 4, figsize=(16, 3.8))
for eixo, coluna in zip(eixos, mais_importantes):
    sns.regplot(data=tabela, x=coluna, y="target", ax=eixo,
                scatter_kws={"s": 8, "alpha": 0.5}, line_kws={"color": "red"})
    eixo.set_title(f"{coluna} (correlação = {correlacao.loc[coluna, 'target']:.2f})")
    eixo.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1b_dispersao.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Nos quatro gráficos a linha vermelha sobe: quanto maior o atributo, maior tende a ser a progressão.
# - Mas os pontos estão muito espalhados em volta da linha. Pacientes com o mesmo IMC, por exemplo, têm progressões que vão de 50 a 300. Então nenhum atributo sozinho consegue prever bem o alvo, e já dá para esperar que o erro final não vai ser pequeno.

# %% [markdown]
# ## Separando entradas e saída

# %%
X = tabela.drop(columns="target").values
y = tabela[["target"]].values
print("Entradas:", X.shape)
print("Saída:", y.shape)

# %% [markdown]
# ## Dividindo em treino, validação e teste

# %%
X_treino, X_val, X_teste, y_treino, y_val, y_teste = dividir_dados(X, y)
print("Treino:", len(X_treino))
print("Validação:", len(X_val))
print("Teste:", len(X_teste))

# %% [markdown]
# ## Normalizando
# Mesma ideia do item a: média 0 e desvio 1, calculados só com o treino. Normalizamos as entradas e também o alvo.

# %%
normalizador_X = StandardScaler().fit(X_treino)
normalizador_y = StandardScaler().fit(y_treino)

X_treino_n = normalizador_X.transform(X_treino)
X_val_n = normalizador_X.transform(X_val)
y_treino_n = normalizador_y.transform(y_treino)
y_val_n = normalizador_y.transform(y_val)

# %% [markdown]
# ## Criando os loaders

# %%
loader_treino = criar_loader(X_treino_n, y_treino_n, tamanho_lote=32, embaralhar=True)
loader_val = criar_loader(X_val_n, y_val_n, tamanho_lote=256, embaralhar=False)

# %% [markdown]
# ## Criando a rede
# A base é pequena: só 309 pacientes no treino. Uma rede grande decoraria esses pacientes. Por isso usamos uma rede pequena:
# - entrada com 10 atributos;
# - camada escondida com 32 neurônios, ReLU e dropout de 20%;
# - camada escondida com 16 neurônios, ReLU e dropout de 20%;
# - saída com 1 neurônio (a progressão prevista).

# %%
modelo_b = criar_modelo_b()
print(modelo_b)
print("Total de pesos treináveis:", sum(p.numel() for p in modelo_b.parameters()))

# %% [markdown]
# ## Treinando
# Além do dropout, usamos weight decay, que penaliza pesos muito grandes e também ajuda contra overfitting. O treino para se a validação ficar 100 épocas sem melhorar.

# %%
historico_b = treinar_modelo_b(modelo_b, loader_treino, loader_val)

# %% [markdown]
# ## Curvas de erro de treino e de validação

# %%
grafico_curvas(historico_b, "Item b: erro durante o treino", arquivo="1b_curvas.png")

# %% [markdown]
# O que observar:
# - A validação atinge o menor erro bem cedo, na época 21. Depois disso ela sobe devagar, sinal de que a rede começaria a decorar o treino. O early stopping parou o treino na época 121 e voltou para os pesos da época 21.
# - A curva de validação fica abaixo da de treino no começo. Parece estranho, e tem dois motivos. O primeiro é o dropout: no treino, 20% dos neurônios são desligados, o que aumenta o erro de treino, e na validação a rede usa todos os neurônios. O segundo é que os 66 pacientes que caíram na validação são um pouco mais "fáceis" que os de treino: na tabela da seção de overfitting, em que os dois conjuntos são medidos do mesmo jeito (sem dropout), o erro da validação continua menor que o do treino.
# - A curva de treino é mais "tremida" por causa do dropout: a cada lote, neurônios diferentes são desligados.
# - Com só 66 pacientes, a curva de validação também depende da sorte da divisão. Por isso, mais abaixo, repetimos tudo com validação cruzada.

# %% [markdown]
# ## Métricas no conjunto de teste
# Para saber se o resultado é bom, comparamos com uma regressão linear simples, que é o modelo mais básico para esse tipo de problema.

# %%
y_previsto_mlp = prever(modelo_b, X_teste, normalizador_X, normalizador_y)

regressao_linear = LinearRegression().fit(X_treino_n, y_treino)
y_previsto_linear = regressao_linear.predict(normalizador_X.transform(X_teste))

pd.DataFrame({
    "MLP": calcular_metricas(y_teste, y_previsto_mlp),
    "Regressão linear": calcular_metricas(y_teste, y_previsto_linear),
}).T.round(2)

# %% [markdown]
# ## Valores reais e preditos

# %%
ordem = np.argsort(y_teste[:, 0])

fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 5))
esquerda.scatter(y_teste, y_previsto_mlp, label="MLP")
esquerda.scatter(y_teste, y_previsto_linear, marker="x", alpha=0.6, label="Regressão linear")
esquerda.plot([0, 360], [0, 360], "k--", linewidth=1)
esquerda.set_xlim(0, 360)
esquerda.set_ylim(0, 360)
esquerda.set_xlabel("progressão real")
esquerda.set_ylabel("progressão prevista")
esquerda.set_title("Teste: real x previsto")
esquerda.legend()
esquerda.grid(alpha=0.3)
direita.plot(y_teste[ordem], "ko-", markersize=3, label="real")
direita.plot(y_previsto_mlp[ordem], "r.--", label="previsto pela MLP")
direita.set_xlabel("pacientes do teste (do menor para o maior valor real)")
direita.set_ylabel("progressão")
direita.set_title("Teste: valores reais e previstos")
direita.legend()
direita.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "1b_real_vs_pred.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Os pontos da MLP (azul) e da regressão linear (laranja) ficam quase nos mesmos lugares. As duas fazem previsões muito parecidas.
# - Os pontos formam uma nuvem "deitada" em relação à linha tracejada: para pacientes com progressão baixa a rede prevê um pouco mais alto, e para progressão alta prevê mais baixo. A rede puxa as previsões para perto da média (cerca de 150), porque os atributos não têm informação suficiente para separar bem os casos extremos.
# - À direita dá para ver isso de novo: a linha vermelha sobe junto com a preta, mas com bastante oscilação e sem chegar aos valores mais altos.

# %% [markdown]
# ## Verificando overfitting
# ### Métricas em treino, validação e teste

# %%
conjuntos_b = {"treino": (X_treino, y_treino), "validação": (X_val, y_val), "teste": (X_teste, y_teste)}
tabela_overfitting(modelo_b, conjuntos_b, normalizador_X, normalizador_y).round(2)

# %% [markdown]
# O RMSE fica em 52,7 no treino, 49,5 na validação e 56,0 no teste. A diferença entre treino e teste é pequena (cerca de 3 pontos), então a rede não decorou o treino.
#
# A validação tem o menor erro dos três. Como aqui os três conjuntos são medidos do mesmo jeito, isso não é efeito do dropout: é a sorte da divisão (os 66 pacientes da validação são mais fáceis) somada ao fato de que escolhemos a época justamente pelo menor erro nela.
#
# A rede tem 897 pesos e o treino tem 309 pacientes. Mesmo assim, com dropout, weight decay e early stopping, ela ficou com um erro de treino parecido com o de dados novos.

# %% [markdown]
# ### Como seria uma rede com overfitting
# Para mostrar o que acontece sem os cuidados que tomamos, treinamos uma rede maior (128, 64 e 32 neurônios) sem dropout, sem weight decay e sem early stopping, por 1000 épocas, e ficamos com os pesos da última época.

# %%
fixar_semente()
modelo_sem_protecao = MLP(entradas=10, camadas=[128, 64, 32], ativacao=nn.ReLU).to(device)
historico_sem_protecao = treinar(modelo_sem_protecao, loader_treino, loader_val, epocas=1000, lr=1e-3,
                                 voltar_melhor=False, nome="Sem proteção")

# %%
grafico_curvas(historico_sem_protecao, "Rede sem dropout, weight decay e early stopping", arquivo="1b_curvas_overfitting.png")

# %% [markdown]
# Este é o overfitting clássico:
# - O erro de treino cai até quase zero: a rede decorou a progressão de cada um dos 309 pacientes.
# - O erro de validação, ao contrário, atinge o menor valor (0,38) na época 4, depois sobe até cerca de 0,72 e termina perto de 0,65.
# - A linha pontilhada mostra que a melhor época foi a 4, mas essa rede usa os pesos da época 1000.

# %%
tabela_overfitting(modelo_sem_protecao, conjuntos_b, normalizador_X, normalizador_y).round(2)

# %% [markdown]
# Nos números, o RMSE de treino é só 1,44, mas no teste ele vai para 70,9 (contra 56,0 da nossa rede) e o R² cai para 0,12. Ou seja, a rede ficou pior que a nossa justamente nos pacientes que nunca viu.
#
# Um cuidado na leitura: esta rede difere da nossa em quatro coisas ao mesmo tempo (é maior, não tem dropout, não tem weight decay e não usa early stopping). O experimento mostra que o conjunto desses cuidados faz diferença, mas não diz quanto vem de cada um. Pela curva, o early stopping sozinho já resolveria boa parte: parando na época 4, o erro de validação desta mesma rede seria 0,38, parecido com o da nossa (0,40).

# %% [markdown]
# ### Validação cruzada
# O teste tem só 67 pacientes, então o resultado depende muito de quais pacientes caíram nele. Para ter um número mais confiável, repetimos tudo 5 vezes, cada vez usando um pedaço diferente da base como teste (validação cruzada com 5 partes).

# %%
lista_resultados = []
partes = KFold(n_splits=5, shuffle=True, random_state=0)

for numero_parte, (indices_treino, indices_teste) in enumerate(partes.split(X), start=1):
    lista_resultados += avaliar_parte(X[indices_treino], y[indices_treino],
                                      X[indices_teste], y[indices_teste], numero_parte)

# %%
tabela_resultados = pd.DataFrame(lista_resultados)
tabela_resultados.groupby("modelo")[["MAE", "MSE", "RMSE", "R²"]].agg(["mean", "std"]).round(2)

# %% [markdown]
# Nas 5 repetições a MLP teve RMSE médio de 55,1 (variando 2,9 para mais ou para menos) e a regressão linear teve 54,9 (variando 2,2). As duas têm R² de 0,48.
#
# A diferença entre elas é menor que a variação entre as repetições. Então, na prática, a MLP e a regressão linear empatam nesta base.

# %% [markdown]
# ---
# # Conclusão
#
# Item a
# - A MLP aprendeu a função f(x) com RMSE de 0,014 no teste. As curvas real e aproximada ficam uma em cima da outra.
# - Não houve overfitting: treino, validação e teste têm o mesmo erro.
# - A rede só vale dentro do intervalo em que foi treinada. Entre x = 10 e x = 12 a previsão fica reta e o erro médio sobe de 0,012 para 3,6.
# - Trocando a Tanh por ReLU a rede também aprende a função (RMSE de 0,019). A Tanh foi um pouco melhor e deixa um erro mais suave, mas a escolha da ativação não foi decisiva.
#
# Item b
# - A MLP teve RMSE de 56,0 e MAE de 46,0 no teste (R² de 0,45). Em média, a previsão erra cerca de 46 pontos numa escala que vai de 25 a 346.
# - Esse resultado é igual ao de uma regressão linear. A relação entre os atributos e a progressão é quase linear e tem muito ruído, porque a doença depende de muitos fatores que não estão nesses 10 atributos. Com só 442 pacientes não há o que a rede aprender além do que a regressão linear já pega.
# - O experimento sem proteção mostrou que uma rede maior, sem regularização e sem early stopping, decora os pacientes de treino e piora no teste (RMSE de 70,9). Por isso usamos uma rede pequena com dropout, weight decay e early stopping.
# - Para melhorar de verdade seria preciso ter mais pacientes ou mais informações sobre eles, e não uma rede maior.

