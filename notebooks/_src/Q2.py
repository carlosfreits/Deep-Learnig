# %% [markdown]
# # Questão 2: Rede MLP para classificar a qualidade de vinhos
#
# Nesta questão usamos uma rede MLP para prever a qualidade de um vinho a partir das suas características químicas (acidez, álcool, pH, densidade etc.). A base é a Wine Quality, da UCI, com vinhos tintos e brancos portugueses.
#
# Cada vinho recebeu uma nota de 0 a 10. Como pede o enunciado, transformamos essa nota em três categorias:
# - baixa: notas de 0 a 4;
# - média: notas 5 e 6;
# - alta: notas de 7 a 10.
#
# Assim o problema vira uma classificação com 3 classes.

# %% [markdown]
# ## Bibliotecas

# %%
import copy
import io
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# usa a placa de vídeo se tiver, senão usa o processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rodando em:", device)

PASTA_DADOS = Path("../data/wine")
PASTA_FIGURAS = Path("../figures/q2")
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

CLASSES = ["baixa", "média", "alta"]

# %% [markdown]
# ## Funções
#
# Aqui ficam todas as funções que usamos no notebook. Cada uma tem uma explicação curta logo acima.

# %% [markdown]
# ### fixar_semente
# Fixa a "semente" dos sorteios (pesos iniciais da rede, embaralhamento dos dados) para que o resultado seja o mesmo toda vez que o notebook for executado.

# %%
def fixar_semente(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

# %% [markdown]
# ### carregar_vinhos
# Baixa a base do site da UCI (só na primeira vez; depois usa o arquivo salvo na pasta `data`). A base vem em dois arquivos, um de vinhos tintos e outro de brancos. Juntamos os dois em uma tabela só e criamos a coluna `tinto` (1 para tinto, 0 para branco), para a rede saber de qual tipo é cada vinho.

# %%
def carregar_vinhos():
    arquivo_zip = PASTA_DADOS / "wine.zip"
    if not arquivo_zip.exists():
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        url = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
        urllib.request.urlretrieve(url, arquivo_zip)

    with zipfile.ZipFile(arquivo_zip) as z:
        tintos = pd.read_csv(io.BytesIO(z.read("winequality-red.csv")), sep=";")
        brancos = pd.read_csv(io.BytesIO(z.read("winequality-white.csv")), sep=";")
    tintos["tinto"] = 1
    brancos["tinto"] = 0
    return pd.concat([tintos, brancos], ignore_index=True)

# %% [markdown]
# ### nota_para_classe
# Transforma a nota (0 a 10) no número da classe: 0 para baixa (até 4), 1 para média (5 e 6) e 2 para alta (7 ou mais).

# %%
def nota_para_classe(nota):
    if nota <= 4:
        return 0
    if nota <= 6:
        return 1
    return 2

# %% [markdown]
# ### calcular_vif
# Mede se uma coluna é redundante, ou seja, se ela pode ser "adivinhada" a partir das outras colunas. Para cada coluna, treinamos uma regressão linear que tenta prever essa coluna usando todas as outras e calculamos o VIF = 1 / (1 - R²).
# - VIF perto de 1: a coluna traz informação nova.
# - VIF acima de 10: a coluna é quase uma combinação das outras, então é redundante.

# %%
def calcular_vif(tabela):
    vif = {}
    for coluna in tabela.columns:
        outras = tabela.drop(columns=coluna)
        r2 = LinearRegression().fit(outras, tabela[coluna]).score(outras, tabela[coluna])
        vif[coluna] = 1 / (1 - r2)
    return pd.Series(vif).sort_values(ascending=False)

# %% [markdown]
# ### criar_loader
# Transforma os arrays em um "loader", que entrega os dados em lotes para o PyTorch. As classes são números inteiros (0, 1 ou 2), por isso o `y` vira um tensor do tipo `long`.

# %%
def criar_loader(X, y, tamanho_lote, embaralhar):
    dados = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dados, batch_size=tamanho_lote, shuffle=embaralhar)

# %% [markdown]
# ### MLP
# A nossa rede. Recebe o número de entradas, a lista de camadas escondidas e o dropout. A saída tem 3 neurônios, um para cada classe. O neurônio com o maior valor é a classe prevista.
#
# A última camada não tem softmax de propósito: a função de erro que usamos (CrossEntropyLoss) já aplica o softmax por dentro.

# %%
class MLP(nn.Module):
    def __init__(self, entradas, camadas, saidas=3, dropout=0.0):
        super().__init__()
        partes = []
        anterior = entradas
        for neuronios in camadas:
            partes.append(nn.Linear(anterior, neuronios))
            partes.append(nn.ReLU())
            if dropout > 0:
                partes.append(nn.Dropout(dropout))
            anterior = neuronios
        partes.append(nn.Linear(anterior, saidas))
        self.rede = nn.Sequential(*partes)

    def forward(self, x):
        return self.rede(x)

# %% [markdown]
# ### rodar_epoca
# Passa uma vez por todos os lotes de um loader. Devolve o erro médio (entropia cruzada) e as classes previstas para cada exemplo.
# - Com otimizador, a rede aprende (ajusta os pesos).
# - Sem otimizador, a rede só é avaliada.

# %%
def rodar_epoca(modelo, loader, funcao_erro, otimizador=None):
    treinando = otimizador is not None
    modelo.train(treinando)
    erro_total = 0.0
    previstos, reais = [], []
    with torch.set_grad_enabled(treinando):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            saida = modelo(x)
            erro = funcao_erro(saida, y)
            if treinando:
                otimizador.zero_grad()
                erro.backward()
                otimizador.step()
            erro_total += erro.item() * len(x)
            previstos.append(saida.argmax(dim=1).cpu())
            reais.append(y.cpu())
    return erro_total / len(loader.dataset), torch.cat(reais).numpy(), torch.cat(previstos).numpy()

# %% [markdown]
# ### treinar
# Treina a rede com o otimizador Adam. A cada época:
# 1. a rede aprende com os dados de treino;
# 2. medimos o erro e o F1 macro na validação;
# 3. guardamos os pesos da época com o maior F1 macro na validação.
#
# Por que F1 macro e não acurácia? Como 77% dos vinhos são de qualidade média, uma rede que sempre responde "média" já acerta 77%. O F1 macro dá o mesmo peso para as três classes, então ele só fica alto se a rede acertar também os vinhos de qualidade baixa e alta.
#
# Se o F1 da validação ficar `paciencia` épocas sem melhorar, o treino para (early stopping).
#
# `pesos_classes` é usado em uma das estratégias de balanceamento: o erro de cada classe é multiplicado por um peso, para que errar uma classe rara "custe" mais.
#
# `mostrar=False` desliga a barra de progresso e as mensagens (usado na validação cruzada, que treina muitas redes).

# %%
def treinar(modelo, loader_treino, loader_val, epocas, lr=1e-3, weight_decay=0.0,
            paciencia=None, pesos_classes=None, nome="Treinando", mostrar=True):
    if pesos_classes is not None:
        pesos_classes = torch.tensor(pesos_classes, dtype=torch.float32, device=device)
    funcao_erro = nn.CrossEntropyLoss(weight=pesos_classes)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    historico = {"treino": [], "val": [], "f1_treino": [], "f1_val": [], "melhor_epoca": 0}
    melhor_f1 = -1.0
    melhores_pesos = None
    epocas_sem_melhorar = 0

    barra = tqdm(range(1, epocas + 1), desc=nome, unit="época", ncols=110, mininterval=2, disable=not mostrar)
    for epoca in barra:
        erro_treino, reais_tr, previstos_tr = rodar_epoca(modelo, loader_treino, funcao_erro, otimizador)
        erro_val, reais_val, previstos_val = rodar_epoca(modelo, loader_val, funcao_erro)
        f1_treino = f1_score(reais_tr, previstos_tr, average="macro")
        f1_val = f1_score(reais_val, previstos_val, average="macro")

        historico["treino"].append(erro_treino)
        historico["val"].append(erro_val)
        historico["f1_treino"].append(f1_treino)
        historico["f1_val"].append(f1_val)

        if f1_val > melhor_f1:
            melhor_f1 = f1_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            historico["melhor_epoca"] = epoca
            epocas_sem_melhorar = 0
        else:
            epocas_sem_melhorar += 1

        barra.set_postfix(erro_treino=f"{erro_treino:.3f}", erro_val=f"{erro_val:.3f}",
                          f1_val=f"{f1_val:.3f}", melhor=historico["melhor_epoca"], refresh=False)

        if paciencia is not None and epocas_sem_melhorar >= paciencia:
            barra.close()
            if mostrar:
                print(f"Parou na época {epoca}: o F1 da validação ficou {paciencia} épocas sem melhorar.")
            break

    # se o erro virou NaN (treino "explodiu"), nenhuma época foi guardada
    if melhores_pesos is None:
        raise RuntimeError("O F1 de validação nunca foi calculado direito. Tente uma taxa de aprendizado menor.")

    modelo.load_state_dict(melhores_pesos)
    if mostrar:
        print(f"Melhor época: {historico['melhor_epoca']} (F1 macro na validação = {melhor_f1:.3f})")
    return historico

# %% [markdown]
# ### prever
# Devolve a classe prevista pela rede para cada vinho e também a probabilidade de cada classe (aplicando o softmax na saída).

# %%
def prever(modelo, X):
    modelo.eval()
    with torch.no_grad():
        saida = modelo(torch.tensor(X, dtype=torch.float32, device=device))
        probabilidades = torch.softmax(saida, dim=1).cpu().numpy()
    return probabilidades.argmax(axis=1), probabilidades

# %% [markdown]
# ### calcular_metricas
# Calcula as métricas pedidas na questão:
# - acurácia: porcentagem de vinhos classificados corretamente;
# - precisão: dos vinhos que a rede disse ser de uma classe, quantos realmente são;
# - recall: dos vinhos que são de uma classe, quantos a rede encontrou;
# - F1: média harmônica entre precisão e recall. Diferente da média comum, ela fica perto do menor dos dois valores, então só é alta se as duas forem altas.
#
# Precisão, recall e F1 são calculados para cada classe e depois tiramos a média simples das três (média "macro"), para que a classe rara pese o mesmo que a classe comum.

# %%
def calcular_metricas(y_real, y_previsto):
    return {
        "acurácia": accuracy_score(y_real, y_previsto),
        "precisão (macro)": precision_score(y_real, y_previsto, average="macro", zero_division=0),
        "recall (macro)": recall_score(y_real, y_previsto, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_real, y_previsto, average="macro", zero_division=0),
    }

# %% [markdown]
# ### grafico_curvas
# Desenha, lado a lado, o erro (entropia cruzada) e o F1 macro de treino e de validação ao longo das épocas. A linha tracejada marca a melhor época, cujos pesos a rede usa no final.

# %%
def grafico_curvas(historico, titulo, arquivo=None):
    epocas = range(1, len(historico["treino"]) + 1)
    melhor = historico["melhor_epoca"]
    fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.5))
    esquerda.plot(epocas, historico["treino"], label="treino")
    esquerda.plot(epocas, historico["val"], label="validação")
    esquerda.set_ylabel("erro (entropia cruzada)")
    direita.plot(epocas, historico["f1_treino"], label="treino")
    direita.plot(epocas, historico["f1_val"], label="validação")
    direita.set_ylabel("F1 macro")
    for eixo in (esquerda, direita):
        eixo.axvline(melhor, color="gray", linestyle="--", label=f"melhor época ({melhor})")
        eixo.set_xlabel("época")
        eixo.legend()
        eixo.grid(alpha=0.3)
    fig.suptitle(titulo)
    plt.tight_layout()
    if arquivo:
        plt.savefig(PASTA_FIGURAS / arquivo, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### grafico_matriz_confusao
# Desenha a matriz de confusão. Cada linha é a classe real e cada coluna é a classe que a rede previu. A diagonal são os acertos. À esquerda mostramos as quantidades e à direita a porcentagem de cada linha (ou seja, o recall de cada classe na diagonal).

# %%
def grafico_matriz_confusao(y_real, y_previsto, titulo, arquivo=None):
    matriz = confusion_matrix(y_real, y_previsto)
    porcentagem = matriz / matriz.sum(axis=1, keepdims=True)
    fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=esquerda)
    sns.heatmap(porcentagem, annot=True, fmt=".0%", cmap="Blues", cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES, ax=direita)
    esquerda.set_title("Quantidade de vinhos")
    direita.set_title("Porcentagem de cada classe real")
    for eixo in (esquerda, direita):
        eixo.set_xlabel("classe prevista")
        eixo.set_ylabel("classe real")
    fig.suptitle(titulo)
    plt.tight_layout()
    if arquivo:
        plt.savefig(PASTA_FIGURAS / arquivo, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### criar_modelo e treinar_modelo
# A mesma rede e o mesmo jeito de treinar são usados em todas as estratégias de balanceamento. Assim a única diferença entre elas é o balanceamento.
# - `criar_modelo`: rede com 128 e 64 neurônios nas camadas escondidas, ReLU e dropout de 30%.
# - `treinar_modelo`: até 300 épocas, taxa de aprendizado 0,001, weight decay 0,0001, lotes de 64 vinhos e parada se o F1 da validação ficar 40 épocas sem melhorar.

# %%
def criar_modelo(entradas):
    fixar_semente()
    return MLP(entradas=entradas, camadas=[128, 64], dropout=0.3).to(device)


def treinar_modelo(modelo, X_treino, y_treino, X_val, y_val, pesos_classes=None, nome="Treinando", mostrar=True):
    fixar_semente()
    loader_treino = criar_loader(X_treino, y_treino, tamanho_lote=64, embaralhar=True)
    loader_val = criar_loader(X_val, y_val, tamanho_lote=512, embaralhar=False)
    return treinar(modelo, loader_treino, loader_val, epocas=300, lr=1e-3, weight_decay=1e-4,
                   paciencia=40, pesos_classes=pesos_classes, nome=nome, mostrar=mostrar)

# %% [markdown]
# ### validacao_cruzada
# Com uma única divisão dos dados, a comparação entre estratégias depende de quais vinhos caíram na validação (são só 35 de qualidade baixa). Esta função repete a comparação várias vezes: divide os dados em 5 partes e, a cada vez, treina com 4 delas e mede o F1 macro na parte que ficou de fora. Devolve os 5 valores de F1.
#
# Em cada repetição o normalizador, os pesos das classes e o SMOTE são calculados só com as 4 partes de treino. A parte que ficou de fora também é usada para o early stopping, como a validação no treino principal. Por isso os valores saem um pouco otimistas, mas isso vale igualmente para todas as estratégias, então a comparação entre elas é justa.

# %%
def validacao_cruzada(X, y, estrategia, nome):
    partes = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lista_f1 = []
    for indices_tr, indices_va in tqdm(list(partes.split(X, y)), desc=nome, unit="parte", ncols=110):
        normalizador_parte = StandardScaler().fit(X[indices_tr])
        X_tr, y_tr = normalizador_parte.transform(X[indices_tr]), y[indices_tr]
        X_va, y_va = normalizador_parte.transform(X[indices_va]), y[indices_va]

        pesos_parte = None
        if estrategia in ("pesos nas classes", "pesos suavizados"):
            pesos_parte = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_tr)
            if estrategia == "pesos suavizados":
                pesos_parte = np.sqrt(pesos_parte)
        if estrategia == "SMOTE":
            X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

        modelo = criar_modelo(X_tr.shape[1])
        treinar_modelo(modelo, X_tr, y_tr, X_va, y_va, pesos_classes=pesos_parte, mostrar=False)
        previstos, _ = prever(modelo, X_va)
        lista_f1.append(f1_score(y_va, previstos, average="macro"))
    return lista_f1

# %% [markdown]
# ---
# # Análise exploratória

# %% [markdown]
# ## Carregando a base

# %%
vinhos = carregar_vinhos()
print("Tamanho da tabela:", vinhos.shape)
vinhos.head()

# %% [markdown]
# Cada linha é um vinho. As colunas são:
# - fixed acidity: acidez fixa
# - volatile acidity: acidez volátil (em excesso dá gosto de vinagre)
# - citric acid: ácido cítrico
# - residual sugar: açúcar que sobra depois da fermentação
# - chlorides: cloretos (sal)
# - free sulfur dioxide e total sulfur dioxide: dióxido de enxofre livre e total (conservante)
# - density: densidade
# - pH
# - sulphates: sulfatos
# - alcohol: teor alcoólico
# - quality: a nota do vinho (o que queremos prever)
# - tinto: coluna que criamos (1 = tinto, 0 = branco)

# %% [markdown]
# ## Valores ausentes

# %%
vinhos.isna().sum()

# %% [markdown]
# Nenhuma coluna tem valor ausente. Não é preciso preencher nem remover nada por esse motivo.

# %% [markdown]
# ## Linhas repetidas

# %%
print("Linhas repetidas:", vinhos.duplicated().sum())
vinhos = vinhos.drop_duplicates().reset_index(drop=True)
print("Tamanho depois de remover as repetidas:", vinhos.shape)

# %% [markdown]
# A base tem 1177 linhas repetidas (vinhos com exatamente os mesmos valores em todas as colunas). Removemos essas linhas por dois motivos:
# - elas não trazem informação nova;
# - se uma cópia cair no treino e a outra no teste, a rede é testada num vinho que ela já viu, e o resultado fica melhor do que deveria.

# %% [markdown]
# ## Estatísticas de cada coluna

# %%
vinhos.describe().T.round(2)

# %% [markdown]
# As colunas estão em escalas muito diferentes. O total sulfur dioxide vai de 6 a 440, enquanto a densidade fica entre 0,99 e 1,04 e os cloretos entre 0,01 e 0,61. Por isso vamos normalizar antes de treinar a rede.

# %% [markdown]
# ## Criando as três classes de qualidade

# %%
vinhos["classe"] = vinhos["quality"].apply(nota_para_classe)

contagem_notas = vinhos["quality"].value_counts().sort_index()
contagem_classes = vinhos["classe"].value_counts().sort_index()
contagem_classes.index = CLASSES
pd.DataFrame({"vinhos": contagem_classes, "porcentagem": (contagem_classes / len(vinhos) * 100).round(1)})

# %% [markdown]
# ## Distribuição das notas e das classes

# %%
fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.5))
cores_notas = ["tab:red" if n <= 4 else "tab:gray" if n <= 6 else "tab:green" for n in contagem_notas.index]
esquerda.bar(contagem_notas.index, contagem_notas.values, color=cores_notas)
esquerda.set_xlabel("nota")
esquerda.set_ylabel("quantidade de vinhos")
esquerda.set_title("Notas originais (vermelho = baixa, cinza = média, verde = alta)")
direita.bar(CLASSES, contagem_classes.values, color=["tab:red", "tab:gray", "tab:green"])
for i, valor in enumerate(contagem_classes.values):
    direita.text(i, valor + 50, f"{valor} ({valor / len(vinhos):.0%})", ha="center")
direita.set_ylabel("quantidade de vinhos")
direita.set_title("Classes que vamos prever")
for eixo in (esquerda, direita):
    eixo.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "2_classes.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Quase todos os vinhos têm nota 5 ou 6. Notas extremas são raras: só uns 30 vinhos com nota 3 e 5 com nota 9. Nenhum vinho tem nota 0, 1, 2 ou 10.
# - Depois de agrupar, as classes ficam muito desbalanceadas: 77% média, 19% alta e só 4% baixa (236 vinhos).
# - Isso significa que uma rede "preguiçosa", que sempre responde "média", acertaria 77% dos vinhos sem aprender nada. Por isso a acurácia sozinha não serve para avaliar o modelo, e vamos olhar principalmente o F1 macro e o recall de cada classe.

# %% [markdown]
# ## Distribuição de cada atributo

# %%
atributos = vinhos.columns.drop(["quality", "classe", "tinto"])
nome_classe = vinhos["classe"].map(dict(enumerate(CLASSES)))
cores = {"baixa": "tab:red", "média": "tab:gray", "alta": "tab:green"}

fig, eixos = plt.subplots(3, 4, figsize=(16, 10))
for eixo, coluna in zip(eixos.flat, atributos):
    sns.boxplot(x=nome_classe, y=vinhos[coluna], order=CLASSES, hue=nome_classe,
                palette=cores, legend=False, ax=eixo, fliersize=2)
    eixo.set_xlabel("")
    eixo.set_title(coluna)
eixos.flat[-1].axis("off")
plt.suptitle("Atributos separados por classe de qualidade")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "2_boxplots.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Como ler: a caixa mostra onde está a metade do meio dos vinhos, a linha dentro dela é a mediana e as bolinhas são valores muito fora do comum (outliers).
#
# O que observar:
# - O álcool é o atributo que mais separa as classes: os vinhos de qualidade alta têm mediana de cerca de 11,6% de álcool, contra cerca de 10,1% dos médios e baixos.
# - A acidez volátil é mais alta nos vinhos de qualidade baixa (mediana perto de 0,40, contra 0,28 nos altos). Faz sentido: acidez volátil em excesso dá gosto de vinagre.
# - A densidade é um pouco menor nos vinhos de qualidade alta, e os cloretos também.
# - Nos outros atributos as caixas das três classes ficam quase na mesma altura, ou seja, sozinhos eles ajudam pouco a separar as classes.
# - Quase todos os atributos têm muitos outliers (por exemplo, açúcar residual de 65 e dióxido de enxofre livre de 289). Não removemos esses vinhos, porque são valores possíveis e não erros de medição. A normalização ajuda a rede a lidar com essas escalas diferentes.
# - As caixas das três classes se sobrepõem bastante em todos os atributos. Isso já indica que separar as classes vai ser difícil.

# %% [markdown]
# ## Correlação entre os atributos

# %%
correlacao = vinhos.drop(columns="classe").corr()

plt.figure(figsize=(11, 9))
sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False)
plt.title("Correlação entre as colunas")
plt.savefig(PASTA_FIGURAS / "2_correlacao.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Na linha quality, o atributo mais ligado à nota é o álcool (0,47), seguido da densidade (-0,33) e da acidez volátil (-0,27). Todos os outros têm correlação fraca com a nota.
# - Alguns pares de atributos andam juntos: dióxido de enxofre livre e total (0,72) e densidade com álcool (-0,67) e com açúcar residual (0,52).
# - A coluna tinto tem correlação forte com vários atributos (acidez volátil 0,65 e dióxido de enxofre total -0,69). Vinhos tintos e brancos têm composições bem diferentes, e é por isso que é importante a rede saber o tipo de cada vinho.

# %% [markdown]
# ## Atributos redundantes
# A correlação compara as colunas só de duas em duas. Uma coluna também pode ser redundante por ser uma combinação de várias outras. Para verificar isso usamos o VIF (explicado na seção de funções).

# %%
vif = calcular_vif(vinhos.drop(columns=["quality", "classe"]))
vif.round(1)

# %% [markdown]
# A densidade tem VIF de 20,7, bem acima de 10. Isso quer dizer que ela pode ser calculada quase perfeitamente a partir dos outros atributos. Faz sentido físico: a densidade do vinho depende principalmente de quanto álcool (mais leve que a água) e quanto açúcar (mais pesado) ele tem.
#
# Por isso removemos a densidade. Ela não traz informação nova e só repete o que o álcool e o açúcar já dizem.
#
# Um cuidado: o VIF vem da regressão linear, onde colunas redundantes atrapalham de verdade. Uma MLP costuma conviver bem com isso. Então, mais adiante (na seção de balanceamento), conferimos se remover a densidade muda o resultado da rede.
#
# Obs.: o VIF foi calculado com a base inteira, antes de separar o teste. Como ele só olha a relação entre os atributos (e não a nota), e a densidade está muito acima do limite, a decisão seria a mesma usando só o treino.

# %%
densidade = vinhos["density"].values  # guardada para um teste mais adiante
vinhos = vinhos.drop(columns="density")
print("VIF depois de remover a densidade:")
calcular_vif(vinhos.drop(columns=["quality", "classe"])).round(1)

# %% [markdown]
# Sem a densidade, todos os atributos ficam com VIF abaixo de 5. O açúcar residual, por exemplo, caiu de 8,4 para 1,4, porque a parte "repetida" dele estava ligada à densidade.
#
# O dióxido de enxofre livre e o total têm correlação de 0,72, mas o VIF deles é baixo (2,2 e 4,0). Então mantemos os dois: um mede só a parte livre e o outro mede tudo.

# %% [markdown]
# ---
# # Pré-processamento

# %% [markdown]
# ## Separando entradas e saída
# A coluna quality não pode entrar como entrada, porque a classe é calculada a partir dela.

# %%
X = vinhos.drop(columns=["quality", "classe"]).values
y = vinhos["classe"].values
nomes_atributos = vinhos.drop(columns=["quality", "classe"]).columns.tolist()
print("Entradas:", X.shape)
print("Atributos usados:", nomes_atributos)

# %% [markdown]
# ## Dividindo em treino, validação e teste
# Separamos 70% para treino, 15% para validação e 15% para teste. A divisão é estratificada: cada parte fica com a mesma proporção de vinhos baixos, médios e altos. Sem isso, a classe baixa, que é rara, poderia ficar com pouquíssimos exemplos na validação ou no teste.

# %%
indices = np.arange(len(X))
indices_treino, indices_resto = train_test_split(indices, test_size=0.30, stratify=y, random_state=42)
indices_val, indices_teste = train_test_split(indices_resto, test_size=0.50, stratify=y[indices_resto], random_state=42)

X_treino, X_val, X_teste = X[indices_treino], X[indices_val], X[indices_teste]
y_treino, y_val, y_teste = y[indices_treino], y[indices_val], y[indices_teste]

pd.DataFrame({
    "treino": np.bincount(y_treino),
    "validação": np.bincount(y_val),
    "teste": np.bincount(y_teste),
}, index=CLASSES)

# %% [markdown]
# ## Normalizando
# Deixamos cada atributo com média 0 e desvio padrão 1. A média e o desvio são calculados só com o treino.

# %%
normalizador = StandardScaler().fit(X_treino)
X_treino_n = normalizador.transform(X_treino)
X_val_n = normalizador.transform(X_val)
X_teste_n = normalizador.transform(X_teste)

# %% [markdown]
# ---
# # Balanceamento das classes
#
# Só 4% dos vinhos são de qualidade baixa. Se treinarmos a rede sem fazer nada, ela tende a quase nunca responder "baixa", porque errar esses poucos vinhos quase não aumenta o erro total.
#
# Para ver se precisamos de balanceamento, treinamos a mesma rede de quatro jeitos e comparamos na validação:
# 1. sem balanceamento;
# 2. com pesos nas classes: o erro de cada vinho é multiplicado por um peso maior quanto mais rara for a classe dele;
# 3. com pesos suavizados: a mesma ideia, mas usando a raiz quadrada dos pesos, para a diferença entre as classes não ser tão grande;
# 4. com SMOTE: criamos vinhos "artificiais" das classes raras no treino, misturando vinhos parecidos que já existem, até as três classes terem a mesma quantidade.
#
# O balanceamento é feito só no treino. Validação e teste continuam com as proporções reais, porque é assim que os vinhos aparecem no mundo real.

# %% [markdown]
# ## Estratégia 1: sem balanceamento

# %%
modelo_sem = criar_modelo(X_treino_n.shape[1])
historico_sem = treinar_modelo(modelo_sem, X_treino_n, y_treino, X_val_n, y_val, nome="Sem balanceamento")

# %% [markdown]
# ## Estratégia 2: pesos nas classes

# %%
pesos = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_treino)
print("Peso de cada classe:", dict(zip(CLASSES, pesos.round(2))))

# %%
modelo_pesos = criar_modelo(X_treino_n.shape[1])
historico_pesos = treinar_modelo(modelo_pesos, X_treino_n, y_treino, X_val_n, y_val,
                                 pesos_classes=pesos, nome="Pesos nas classes")

# %% [markdown]
# ## Estratégia 3: pesos suavizados
# Com os pesos da estratégia 2, errar um vinho de qualidade baixa custa 17 vezes mais do que errar um vinho médio (7,52 contra 0,44). Isso pode fazer a rede exagerar e começar a chamar muitos vinhos médios de baixos. Tirando a raiz quadrada, a diferença cai para cerca de 4 vezes.

# %%
pesos_suaves = np.sqrt(pesos)
print("Peso de cada classe:", dict(zip(CLASSES, pesos_suaves.round(2))))

# %%
modelo_suave = criar_modelo(X_treino_n.shape[1])
historico_suave = treinar_modelo(modelo_suave, X_treino_n, y_treino, X_val_n, y_val,
                                 pesos_classes=pesos_suaves, nome="Pesos suavizados")

# %% [markdown]
# ## Estratégia 4: SMOTE

# %%
X_treino_smote, y_treino_smote = SMOTE(random_state=42).fit_resample(X_treino_n, y_treino)
print("Vinhos no treino antes do SMOTE:", dict(zip(CLASSES, np.bincount(y_treino))))
print("Vinhos no treino depois do SMOTE:", dict(zip(CLASSES, np.bincount(y_treino_smote))))

# %%
modelo_smote = criar_modelo(X_treino_n.shape[1])
historico_smote = treinar_modelo(modelo_smote, X_treino_smote, y_treino_smote, X_val_n, y_val, nome="SMOTE")

# %% [markdown]
# ## Comparando as quatro estratégias na validação
# Primeiro olhamos as quatro redes na validação. O teste fica guardado para o final.

# %%
estrategias = {"sem balanceamento": modelo_sem, "pesos nas classes": modelo_pesos,
               "pesos suavizados": modelo_suave, "SMOTE": modelo_smote}

comparacao = {}
recall_por_classe = {}
for nome, modelo in estrategias.items():
    previstos_val, _ = prever(modelo, X_val_n)
    comparacao[nome] = calcular_metricas(y_val, previstos_val)
    recall_por_classe[nome] = recall_score(y_val, previstos_val, average=None)

tabela_comparacao = pd.DataFrame(comparacao).T
tabela_recall = pd.DataFrame(recall_por_classe, index=[f"recall {c}" for c in CLASSES]).T
pd.concat([tabela_comparacao, tabela_recall], axis=1).round(3)

# %% [markdown]
# ## Gráfico da comparação

# %%
tabela_grafico = pd.concat([tabela_comparacao[["acurácia", "F1 (macro)"]], tabela_recall], axis=1)

tabela_grafico.T.plot.bar(figsize=(11, 4.5), rot=0, color=["tab:gray", "tab:blue", "tab:cyan", "tab:orange"])
plt.ylabel("valor na validação")
plt.title("Comparação das estratégias de balanceamento (validação)")
plt.ylim(0, 1)
plt.grid(alpha=0.3, axis="y")
plt.legend(title="estratégia")
plt.savefig(PASTA_FIGURAS / "2_comparacao_balanceamento.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - Sem balanceamento a rede tem a maior acurácia (80,5%), mas só encontra 14% dos vinhos de qualidade baixa. Ela quase sempre responde "média", exatamente o problema que esperávamos. Então o balanceamento é necessário.
# - Com os pesos completos e com o SMOTE acontece o contrário: a rede passa a encontrar mais de 40% dos vinhos baixos e mais de 80% dos altos, mas começa a errar muitos vinhos médios (recall de 64% e 66%), e a acurácia cai para perto de 68%.
# - Os pesos suavizados ficam no meio-termo: encontram 37% dos baixos, 85% dos médios e 59% dos altos, e têm o maior F1 macro (0,603).
# - Não existe estratégia que ganhe em tudo: sempre que a rede passa a encontrar mais vinhos raros, ela erra mais vinhos médios. A escolha depende do que importa mais. Aqui usamos o F1 macro, que trata as três classes como igualmente importantes.
# - As diferenças de F1 são pequenas (de 0,540 a 0,603) para uma validação com só 35 vinhos baixos. Antes de escolher, conferimos se a ordem se mantém com outras divisões dos dados.

# %% [markdown]
# ## Validação cruzada das estratégias de balanceamento
# Na tabela acima os valores de F1 macro das quatro estratégias são próximos, e a validação tem só 35 vinhos de qualidade baixa: acertar ou errar 2 ou 3 deles já muda a ordem. Para não escolher a estratégia "por sorte", repetimos a comparação com validação cruzada de 5 partes (função `validacao_cruzada`), usando os vinhos de treino e de validação juntos. O teste continua de fora.

# %%
X_desenvolvimento = np.concatenate([X_treino, X_val])
y_desenvolvimento = np.concatenate([y_treino, y_val])

f1_cruzada = {nome: validacao_cruzada(X_desenvolvimento, y_desenvolvimento, nome, nome) for nome in estrategias}

# %%
tabela_cruzada = pd.DataFrame(f1_cruzada, index=[f"parte {i}" for i in range(1, 6)]).T
tabela_cruzada["média"] = tabela_cruzada.mean(axis=1)
tabela_cruzada["desvio"] = tabela_cruzada.drop(columns="média").std(axis=1)
tabela_cruzada.round(3)

# %% [markdown]
# O que observar:
# - Os pesos suavizados continuam na frente: F1 macro médio de 0,595, contra 0,559 do SMOTE, 0,546 dos pesos completos e 0,540 sem balanceamento.
# - Além disso, os pesos suavizados têm o maior F1 em cada uma das 5 partes. A vantagem não depende de uma divisão específica dos dados.
# - A variação de uma parte para outra é de cerca de 0,03 em uma mesma estratégia. Então a vantagem dos pesos suavizados sobre a segunda colocada (0,036) é real, mas não é grande. Já as outras três estratégias ficam a menos de 0,02 uma da outra, dentro dessa variação: entre elas não dá para dizer qual é melhor.
# - O valor de 0,603 que vimos na validação única fica dentro do que a validação cruzada mostra (de 0,561 a 0,628).
#
# Escolhemos então a estratégia com o maior F1 macro médio na validação cruzada: os pesos suavizados.

# %% [markdown]
# ## Teste com e sem a densidade
# Removemos a densidade por causa do VIF. Para conferir se isso prejudica a rede, repetimos a validação cruzada da estratégia escolhida colocando a densidade de volta como 12º atributo.

# %%
melhor_estrategia = tabela_cruzada["média"].idxmax()
print("Estratégia com o maior F1 macro médio:", melhor_estrategia)

X_com_densidade = np.column_stack([X, densidade])
X_desenvolvimento_cd = np.concatenate([X_com_densidade[indices_treino], X_com_densidade[indices_val]])
f1_com_densidade = validacao_cruzada(X_desenvolvimento_cd, y_desenvolvimento, melhor_estrategia, "com densidade")

tabela_densidade = pd.DataFrame({
    "sem densidade (11 atributos)": f1_cruzada[melhor_estrategia],
    "com densidade (12 atributos)": f1_com_densidade,
}, index=[f"parte {i}" for i in range(1, 6)]).T
tabela_densidade["média"] = tabela_densidade.mean(axis=1)
tabela_densidade["desvio"] = tabela_densidade.drop(columns="média").std(axis=1)
tabela_densidade.round(3)

# %% [markdown]
# Com a densidade o F1 macro médio é 0,604, e sem ela é 0,595. A diferença (0,009) é bem menor que a variação entre as partes (cerca de 0,03), e em uma das cinco partes o resultado sem a densidade é melhor.
#
# Então remover a densidade não prejudicou a rede de forma que dê para medir. Também não ajudou: para uma MLP, a redundância apontada pelo VIF não era um problema. Mantemos a remoção porque ela deixa o modelo mais simples (um atributo a menos) sem perda clara, mas a decisão contrária também seria defensável.

# %% [markdown]
# ---
# # Modelo escolhido

# %% [markdown]
# ## Arquitetura da rede

# %%
modelo_final = estrategias[melhor_estrategia]
historicos = {"sem balanceamento": historico_sem, "pesos nas classes": historico_pesos,
              "pesos suavizados": historico_suave, "SMOTE": historico_smote}
historico_final = historicos[melhor_estrategia]

print("Estratégia escolhida:", melhor_estrategia)
print(modelo_final)
print("Total de pesos treináveis:", sum(p.numel() for p in modelo_final.parameters()))

# %% [markdown]
# A rede tem:
# - camada de entrada com 11 atributos (os 10 atributos químicos que sobraram mais a coluna tinto);
# - primeira camada escondida com 128 neurônios, ReLU e dropout de 30%;
# - segunda camada escondida com 64 neurônios, ReLU e dropout de 30%;
# - camada de saída com 3 neurônios, um por classe.
#
# Treinamento: otimizador Adam, taxa de aprendizado 0,001, weight decay 0,0001, lotes de 64 vinhos e função de erro entropia cruzada.

# %% [markdown]
# ## Curvas de erro de treino e de validação

# %%
grafico_curvas(historico_final, f"Treino com a estratégia: {melhor_estrategia}", arquivo="2_curvas.png")

# %% [markdown]
# O que observar:
# - No gráfico da esquerda, os erros de treino e de validação caem juntos até perto da época 20. Depois o erro de treino continua caindo, mas o de validação começa a subir. Esse é o começo do overfitting: a rede passa a decorar detalhes do treino que não valem para vinhos novos.
# - No gráfico da direita, o F1 da validação para de melhorar por volta da época 30 (0,603), enquanto o F1 de treino continua subindo até cerca de 0,65.
# - O early stopping guardou os pesos da época 30 e parou o treino na época 70. Assim a rede final é a de antes do overfitting.
# - As curvas da validação são mais "tremidas" porque a classe baixa tem só 35 vinhos na validação: acertar ou errar 2 ou 3 deles já muda bastante o F1.
# - Como a curva oscila, a época 30 é um pico de uma curva tremida. Escolher o ponto mais alto de uma curva assim tende a dar um valor um pouco acima do que a rede realmente vale. É um dos motivos de o F1 no teste sair menor, como veremos adiante.

# %% [markdown]
# ## Matriz de confusão no teste

# %%
previstos_teste, probabilidades_teste = prever(modelo_final, X_teste_n)
grafico_matriz_confusao(y_teste, previstos_teste, f"Matriz de confusão no teste ({melhor_estrategia})",
                        arquivo="2_matriz_confusao.png")

# %% [markdown]
# O que observar:
# - Vinhos médios: a rede acerta 81% (492 de 611). Os erros vão mais para "alta" (83) do que para "baixa" (36).
# - Vinhos altos: a rede acerta metade (76 de 151). A outra metade é confundida com média. Nenhum vinho alto foi chamado de baixo.
# - Vinhos baixos: a rede acerta 33% (12 de 36). A maioria (23) é chamada de média, e só 1 é chamado de alto.
# - Quase não há confusão entre baixa e alta (1 caso em 798). Os erros ficam sempre entre classes vizinhas. Isso faz sentido: um vinho de nota 4 é parecido com um de nota 5, e um de nota 7 é parecido com um de nota 6, então a fronteira entre as classes é difícil de achar.

# %% [markdown]
# ## Métricas no teste

# %%
pd.DataFrame([calcular_metricas(y_teste, previstos_teste)], index=["teste"]).round(3)

# %% [markdown]
# ## Métricas por classe

# %%
print(classification_report(y_teste, previstos_teste, target_names=CLASSES, digits=3))

# %% [markdown]
# O que observar:
# - Acurácia de 72,7%: a rede acerta quase 3 de cada 4 vinhos.
# - F1 macro de 0,531. A classe média tem F1 de 0,82, a alta 0,49 e a baixa só 0,29.
# - A precisão da classe baixa é de 0,25: de cada 4 vinhos que a rede chama de baixo, só 1 realmente é. Isso acontece porque, para encontrar vinhos baixos, a rede acaba marcando também vinhos médios parecidos com eles, e como existem muito mais médios do que baixos, esses alarmes falsos são maioria.
# - O F1 macro do teste (0,531) ficou abaixo do da validação (0,603). Isso tem dois motivos:
#   - a validação foi usada para escolher a melhor época, e os vinhos dela também entraram na validação cruzada que escolheu a estratégia de balanceamento. Escolhendo pelo melhor resultado na validação, esse resultado tende a ficar um pouco acima do real. O teste, que não participou de nenhuma escolha, mostra o valor mais confiável;
#   - com só 36 vinhos baixos no teste, poucos acertos a mais ou a menos mudam muito o F1.

# %% [markdown]
# ---
# # Verificando overfitting

# %% [markdown]
# ## Métricas em treino, validação e teste
# Para o treino usamos os vinhos originais (sem os vinhos artificiais do SMOTE).

# %%
tabela_overfitting = {}
for nome, (X_parte, y_parte) in {"treino": (X_treino_n, y_treino), "validação": (X_val_n, y_val), "teste": (X_teste_n, y_teste)}.items():
    previstos, _ = prever(modelo_final, X_parte)
    tabela_overfitting[nome] = calcular_metricas(y_parte, previstos)
pd.DataFrame(tabela_overfitting).T.round(3)

# %% [markdown]
# O F1 macro é 0,634 no treino, 0,603 na validação e 0,531 no teste. A acurácia é 78,3%, 77,8% e 72,7%.
#
# Treino e validação estão bem próximos, então a rede não decorou o treino: o early stopping parou antes disso, como vimos nas curvas. A queda maior no teste não é overfitting no treino. Ela vem de a validação ter sido usada para escolher a época e de ter participado da escolha da estratégia (o que deixa o resultado da validação um pouco otimista) e da classe baixa, que tem poucos exemplos (36) e por isso varia muito de um sorteio para outro.

# %% [markdown]
# ## Comparação com outro modelo
# Para ter uma referência, treinamos também uma Random Forest (um conjunto de árvores de decisão), que costuma ir bem em tabelas como esta. Ela recebe os mesmos dados e usa pesos nas classes para lidar com o desbalanceamento.

# %%
floresta = RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=3,
                                  random_state=42, n_jobs=-1)
floresta.fit(X_treino_n, y_treino)

pd.DataFrame({
    f"MLP ({melhor_estrategia})": calcular_metricas(y_teste, previstos_teste),
    "Random Forest": calcular_metricas(y_teste, floresta.predict(X_teste_n)),
}).T.round(3)

# %% [markdown]
# A Random Forest teve F1 macro de 0,537 e acurácia de 70,9%, muito perto da nossa MLP (0,531 e 72,7%). Ou seja, a dificuldade não está na rede: é o problema que é difícil. Com estes atributos químicos, os vinhos de qualidade baixa e alta são muito parecidos com os de qualidade média, como vimos nos boxplots.

# %% [markdown]
# ---
# # Conclusão
#
# Análise e pré-processamento
# - A base não tem valores ausentes, mas tinha 1177 vinhos repetidos, que removemos.
# - A densidade era redundante (VIF de 20,7, quase uma combinação de álcool e açúcar) e foi removida. Conferimos com validação cruzada que isso não prejudica a rede: o F1 macro fica em 0,595 sem ela e 0,604 com ela, diferença menor que a variação entre as repetições. Os outros atributos foram mantidos e normalizados.
# - As classes são muito desbalanceadas (4% baixa, 77% média, 19% alta), então usamos divisão estratificada e o F1 macro como métrica principal.
#
# Balanceamento
# - Sem balanceamento, a rede quase nunca reconhece vinhos ruins (recall de 14%). Então o balanceamento é necessário.
# - Testamos pesos nas classes, pesos suavizados e SMOTE. Os pesos suavizados foram os melhores na validação e também na validação cruzada, em todas as 5 partes (F1 macro médio de 0,595 contra 0,559 da segunda colocada).
#
# Resultado
# - Rede com 11 entradas, camadas de 128 e 64 neurônios (ReLU e dropout de 30%) e 3 saídas.
# - No teste: acurácia de 72,7%, precisão macro de 0,52, recall macro de 0,55 e F1 macro de 0,53.
# - A rede reconhece bem os vinhos médios (81%), metade dos altos e um terço dos baixos. Os erros acontecem quase sempre entre classes vizinhas.
# - Não houve overfitting relevante. O early stopping parou o treino quando a validação começou a piorar.
# - Uma Random Forest chegou praticamente ao mesmo resultado, o que mostra que o limite está nos dados: só com a química do vinho não dá para separar bem as notas, que também dependem do gosto de quem avaliou.
#
# O que ficou de fora
# - As três classes têm uma ordem (baixa, média, alta), e quase todos os erros são entre vizinhas. Tratamos o problema como uma classificação comum, que ignora essa ordem. Um modelo que a levasse em conta (por exemplo, prevendo a nota e depois cortando em faixas) é um caminho natural para testar.
# - O número do teste vem de uma única divisão, com só 36 vinhos de qualidade baixa. Ele deve ser lido com essa margem de incerteza.

