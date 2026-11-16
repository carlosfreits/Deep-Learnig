# %% [markdown]
# # Questão 3: MLP e CNN na base CIFAR-10
#
# A CIFAR-10 tem 60 mil imagens coloridas pequenas (32 x 32 pixels), divididas em 10 classes: avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão.
#
# Nesta questão treinamos dois classificadores e comparamos os resultados:
# - item a: uma rede MLP, que recebe a imagem "achatada" em uma lista de números;
# - item b: uma rede convolucional (CNN), que olha a imagem como imagem, analisando pedaços vizinhos de pixels.
#
# Para cada um mostramos a arquitetura, as curvas de erro, a matriz de confusão e a acurácia no teste.

# %% [markdown]
# ## Bibliotecas

# %%
import copy
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import datasets
from tqdm import tqdm

# usa a placa de vídeo se tiver, senão usa o processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Rodando em:", device)

PASTA_DADOS = Path("../data/cifar10")
PASTA_MODELOS = Path("../checkpoints")
PASTA_FIGURAS = Path("../figures/q3")
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

CLASSES = ["avião", "automóvel", "pássaro", "gato", "cervo", "cachorro", "sapo", "cavalo", "navio", "caminhão"]

# %% [markdown]
# ## Funções
#
# Aqui ficam todas as funções que usamos no notebook. Cada uma tem uma explicação curta logo acima.

# %% [markdown]
# ### fixar_semente
# Fixa a "semente" dos sorteios (pesos iniciais, embaralhamento, data augmentation) e pede para a placa de vídeo usar sempre os mesmos algoritmos. Assim o resultado é o mesmo toda vez que o notebook for executado.

# %%
def fixar_semente(semente=42):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    # faz a placa de vídeo usar sempre os mesmos algoritmos, para o resultado não variar entre execuções
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %% [markdown]
# ### carregar_cifar
# Baixa a CIFAR-10 (só na primeira vez) e devolve as imagens e as classes como arrays. As imagens vêm no formato (quantidade, altura, largura, cor), com valores de 0 a 255.

# %%
def carregar_cifar():
    treino = datasets.CIFAR10(PASTA_DADOS, train=True, download=True)
    teste = datasets.CIFAR10(PASTA_DADOS, train=False, download=True)
    return treino.data, np.array(treino.targets), teste.data, np.array(teste.targets)

# %% [markdown]
# ### para_tensor
# Prepara as imagens para a rede:
# 1. muda a ordem para (quantidade, cor, altura, largura), que é a ordem usada pelo PyTorch;
# 2. divide por 255 para os pixels ficarem entre 0 e 1;
# 3. normaliza cada canal de cor (vermelho, verde, azul) com a média e o desvio calculados no treino.
#
# Tudo vai direto para a placa de vídeo, o que deixa o treino bem mais rápido.

# %%
def para_tensor(imagens, media, desvio):
    x = torch.tensor(imagens, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    return (x - media.view(1, 3, 1, 1)) / desvio.view(1, 3, 1, 1)

# %% [markdown]
# ### aumentar_dados
# Data augmentation: a cada lote, criamos versões levemente diferentes das imagens de treino. Fazemos duas coisas:
# - espelhamos metade das imagens na horizontal (um carro virado para a esquerda continua sendo um carro);
# - deslocamos cada imagem até 4 pixels para os lados e para cima ou para baixo (preenchendo a borda com o reflexo da imagem).
#
# Assim a rede quase nunca vê exatamente a mesma imagem duas vezes, o que dificulta que ela decore o treino.

# %%
def aumentar_dados(x):
    n = x.shape[0]
    # espelhamento horizontal em metade das imagens
    espelhar = torch.rand(n, device=x.device) < 0.5
    x = torch.where(espelhar.view(-1, 1, 1, 1), x.flip(3), x)
    # deslocamento aleatório de até 4 pixels
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    desloc_linha = torch.randint(0, 9, (n,), device=x.device)
    desloc_coluna = torch.randint(0, 9, (n,), device=x.device)
    posicoes = torch.arange(32, device=x.device)
    linhas = (desloc_linha.view(-1, 1) + posicoes).view(n, 1, 32, 1).expand(n, 3, 32, 40)
    colunas = (desloc_coluna.view(-1, 1) + posicoes).view(n, 1, 1, 32).expand(n, 3, 32, 32)
    x = x.gather(2, linhas)  # escolhe 32 linhas a partir do deslocamento
    return x.gather(3, colunas)  # escolhe 32 colunas a partir do deslocamento

# %% [markdown]
# ### lotes
# Divide as imagens em lotes (pedaços) de um tamanho fixo. No treino os lotes são sorteados de novo a cada época.

# %%
def lotes(X, y, tamanho_lote, embaralhar):
    ordem = torch.randperm(len(X), device=X.device) if embaralhar else torch.arange(len(X), device=X.device)
    for inicio in range(0, len(X), tamanho_lote):
        indices = ordem[inicio:inicio + tamanho_lote]
        yield X[indices], y[indices]

# %% [markdown]
# ### MLP
# A rede do item a. A imagem de 32 x 32 pixels com 3 cores vira uma lista de 3072 números (32 x 32 x 3). Depois vêm três camadas escondidas (1024, 512 e 256 neurônios) e a saída com 10 neurônios, um por classe.
#
# Em cada camada escondida usamos:
# - BatchNorm: normaliza os valores dentro da rede, o que deixa o treino mais estável e mais rápido;
# - ReLU: a função de ativação;
# - Dropout de 30%: desliga neurônios ao acaso no treino, para evitar overfitting.

# %%
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        camadas = []
        anterior = 3 * 32 * 32
        for neuronios in [1024, 512, 256]:
            camadas += [nn.Linear(anterior, neuronios), nn.BatchNorm1d(neuronios), nn.ReLU(), nn.Dropout(0.3)]
            anterior = neuronios
        camadas.append(nn.Linear(anterior, 10))
        self.rede = nn.Sequential(nn.Flatten(), *camadas)

    def forward(self, x):
        return self.rede(x)

# %% [markdown]
# ### bloco_convolucional
# Um "bloco" da CNN: duas convoluções seguidas, cada uma com BatchNorm e ReLU, e depois um MaxPool, que reduz a imagem pela metade (por exemplo, de 32 x 32 para 16 x 16), e um Dropout.
#
# A convolução passa pequenos filtros de 3 x 3 pixels por toda a imagem. Cada filtro aprende a detectar um padrão (uma borda, uma cor, uma textura) em qualquer lugar da imagem.

# %%
def bloco_convolucional(entrada, saida, dropout):
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
        nn.BatchNorm2d(saida),
        nn.ReLU(),
        nn.Conv2d(saida, saida, kernel_size=3, padding=1),
        nn.BatchNorm2d(saida),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )

# %% [markdown]
# ### CNN
# A rede do item b, com três blocos convolucionais:
# - bloco 1: 64 filtros, imagem de 32 x 32 vira 16 x 16;
# - bloco 2: 128 filtros, 16 x 16 vira 8 x 8;
# - bloco 3: 256 filtros, 8 x 8 vira 4 x 4.
#
# No final, o "global average pooling" tira a média de cada um dos 256 mapas, gerando 256 números, e uma camada linear transforma isso nas 10 classes.
#
# Quanto mais fundo o bloco, mais complexos são os padrões: o primeiro bloco detecta bordas e cores, e os últimos detectam partes de objetos (rodas, olhos, asas).

# %%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloco1 = bloco_convolucional(3, 64, dropout=0.1)
        self.bloco2 = bloco_convolucional(64, 128, dropout=0.2)
        self.bloco3 = bloco_convolucional(128, 256, dropout=0.3)
        self.saida = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10))

    def forward(self, x):
        return self.saida(self.bloco3(self.bloco2(self.bloco1(x))))

# %% [markdown]
# ### rodar_epoca
# Passa uma vez por todos os lotes e devolve o erro médio (entropia cruzada) e a acurácia.
# - Com otimizador, a rede aprende. Se `aumentar` for verdadeiro, as imagens passam pela data augmentation antes.
# - Sem otimizador, a rede só é avaliada.
#
# Usamos "precisão mista" (autocast): a placa de vídeo faz parte das contas com números menores (16 bits), o que deixa o treino mais rápido e gasta menos memória, sem mudar o resultado.

# %%
def rodar_epoca(modelo, X, y, tamanho_lote, otimizador=None, agendador=None, escalador=None, aumentar=False):
    treinando = otimizador is not None
    modelo.train(treinando)
    funcao_erro = nn.CrossEntropyLoss()
    erro_total, acertos = 0.0, 0
    with torch.set_grad_enabled(treinando):
        for x, alvo in lotes(X, y, tamanho_lote, embaralhar=treinando):
            if aumentar:
                x = aumentar_dados(x)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                saida = modelo(x)
                erro = funcao_erro(saida, alvo)
            if treinando:
                otimizador.zero_grad()
                escalador.scale(erro).backward()
                escalador.step(otimizador)
                escalador.update()
                if agendador is not None:
                    agendador.step()
            erro_total += erro.item() * len(x)
            acertos += (saida.argmax(dim=1) == alvo).sum().item()
    return erro_total / len(X), acertos / len(X)

# %% [markdown]
# ### treinar
# Treina a rede por várias épocas. A cada época:
# 1. a rede aprende com o treino;
# 2. medimos o erro e a acurácia na validação;
# 3. guardamos os pesos da época com a maior acurácia na validação.
#
# Opções:
# - `paciencia`: se a acurácia da validação passar esse número de épocas sem melhorar, o treino para (early stopping);
# - `one_cycle`: usa o agendador OneCycle, que começa com uma taxa de aprendizado baixa, aumenta até o máximo e depois vai diminuindo até quase zero. Isso costuma deixar a CNN mais precisa no final do treino;
# - `aumentar`: liga a data augmentation;
# - `salvar_epocas`: lista de épocas em que salvamos uma cópia dos pesos em arquivo (usamos isso na Questão 4 para ver como a rede muda ao longo do treino).
#
# A barra de progresso mostra o erro e a acurácia de treino e de validação.

# %%
def treinar(modelo, dados, epocas, lr, weight_decay=0.0, paciencia=None, one_cycle=False,
            aumentar=False, tamanho_lote=128, salvar_epocas=(), nome_arquivo=None, extras=None, nome="Treinando"):
    X_treino, y_treino, X_val, y_val = dados
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    agendador = None
    if one_cycle:
        passos = (len(X_treino) + tamanho_lote - 1) // tamanho_lote
        agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=lr, epochs=epocas, steps_per_epoch=passos)
    escalador = torch.amp.GradScaler(enabled=device.type == "cuda")

    historico = {"erro_treino": [], "erro_val": [], "acc_treino": [], "acc_val": [], "melhor_epoca": 0}
    melhor_acc = -1.0
    melhores_pesos = None
    epocas_sem_melhorar = 0
    inicio = time.time()

    barra = tqdm(range(1, epocas + 1), desc=nome, unit="época", ncols=110, mininterval=2)
    for epoca in barra:
        erro_tr, acc_tr = rodar_epoca(modelo, X_treino, y_treino, tamanho_lote, otimizador, agendador, escalador, aumentar)
        erro_val, acc_val = rodar_epoca(modelo, X_val, y_val, 1000)
        historico["erro_treino"].append(erro_tr)
        historico["erro_val"].append(erro_val)
        historico["acc_treino"].append(acc_tr)
        historico["acc_val"].append(acc_val)

        if acc_val > melhor_acc:
            melhor_acc = acc_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            historico["melhor_epoca"] = epoca
            epocas_sem_melhorar = 0
        else:
            epocas_sem_melhorar += 1

        if epoca in salvar_epocas:
            salvar_modelo(modelo, PASTA_MODELOS / f"{nome_arquivo}_epoca_{epoca:02d}.pt", epoca, extras)

        barra.set_postfix(acc_treino=f"{acc_tr:.3f}", acc_val=f"{acc_val:.3f}",
                          erro_val=f"{erro_val:.3f}", melhor=historico["melhor_epoca"], refresh=False)

        if paciencia is not None and epocas_sem_melhorar >= paciencia:
            barra.close()
            print(f"Parou na época {epoca}: a acurácia da validação ficou {paciencia} épocas sem melhorar.")
            break

    # se o erro virou NaN (treino "explodiu"), nenhuma época foi guardada
    if melhores_pesos is None:
        raise RuntimeError("A acurácia de validação nunca foi calculada direito. Tente uma taxa de aprendizado menor.")

    modelo.load_state_dict(melhores_pesos)
    minutos = (time.time() - inicio) / 60
    print(f"Melhor época: {historico['melhor_epoca']} (acurácia na validação = {melhor_acc:.4f}). Tempo: {minutos:.1f} min")
    return historico

# %% [markdown]
# ### salvar_modelo
# Salva os pesos da rede em um arquivo, junto com a época e a média e o desvio usados na normalização. A Questão 4 precisa dessas informações para usar a mesma rede.

# %%
def salvar_modelo(modelo, caminho, epoca, extras=None):
    torch.save({"pesos": modelo.state_dict(), "epoca": epoca, **(extras or {})}, caminho)

# %% [markdown]
# ### prever
# Devolve a classe prevista e as probabilidades de cada classe (softmax) para um conjunto de imagens. Processa em lotes de 1000 imagens para não estourar a memória da placa de vídeo.

# %%
def prever(modelo, X):
    modelo.eval()
    probabilidades = []
    with torch.no_grad():
        for inicio in range(0, len(X), 1000):
            saida = modelo(X[inicio:inicio + 1000])
            probabilidades.append(torch.softmax(saida.float(), dim=1).cpu())
    probabilidades = torch.cat(probabilidades).numpy()
    return probabilidades.argmax(axis=1), probabilidades

# %% [markdown]
# ### acuracia_por_classe
# Para cada classe, calcula a porcentagem de imagens daquela classe que a rede acertou.

# %%
def acuracia_por_classe(y_real, y_previsto):
    return pd.Series({classe: (y_previsto[y_real == i] == i).mean() for i, classe in enumerate(CLASSES)})

# %% [markdown]
# ### grafico_curvas
# Desenha o erro e a acurácia de treino e de validação ao longo das épocas. A linha tracejada marca a melhor época, cujos pesos a rede usa no final.

# %%
def grafico_curvas(historico, titulo, arquivo=None):
    epocas = range(1, len(historico["erro_treino"]) + 1)
    melhor = historico["melhor_epoca"]
    fig, (esquerda, direita) = plt.subplots(1, 2, figsize=(13, 4.5))
    esquerda.plot(epocas, historico["erro_treino"], label="treino")
    esquerda.plot(epocas, historico["erro_val"], label="validação")
    esquerda.set_ylabel("erro (entropia cruzada)")
    direita.plot(epocas, historico["acc_treino"], label="treino")
    direita.plot(epocas, historico["acc_val"], label="validação")
    direita.set_ylabel("acurácia")
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
# Desenha a matriz de confusão. Cada linha é a classe real e cada coluna é a classe prevista. Cada classe tem 1000 imagens no teste, então os números da diagonal são quantas das 1000 a rede acertou.

# %%
def grafico_matriz_confusao(y_real, y_previsto, titulo, arquivo=None):
    matriz = confusion_matrix(y_real, y_previsto)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("classe prevista")
    plt.ylabel("classe real")
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if arquivo:
        plt.savefig(PASTA_FIGURAS / arquivo, bbox_inches="tight")
    plt.show()
    return matriz

# %% [markdown]
# ### maiores_confusoes
# Lista os pares de classes que a rede mais confunde (fora da diagonal da matriz de confusão).

# %%
def maiores_confusoes(matriz, quantidade=5):
    pares = []
    for real in range(10):
        for previsto in range(10):
            if real != previsto:
                pares.append({"classe real": CLASSES[real], "prevista como": CLASSES[previsto], "imagens": matriz[real, previsto]})
    return pd.DataFrame(pares).sort_values("imagens", ascending=False).head(quantidade).reset_index(drop=True)

# %% [markdown]
# ### tabela_overfitting
# Calcula a acurácia em treino, validação e teste lado a lado. Se o treino for muito melhor que os outros dois, a rede decorou o treino.

# %%
def tabela_overfitting(modelo, conjuntos):
    linhas = {}
    for nome, (X, y) in conjuntos.items():
        previstos, _ = prever(modelo, X)
        linhas[nome] = {"acurácia": (previstos == y.cpu().numpy()).mean()}
    return pd.DataFrame(linhas).T

# %% [markdown]
# ---
# # Preparando os dados

# %% [markdown]
# ## Carregando a base

# %%
imagens_treino, classes_treino, imagens_teste, classes_teste = carregar_cifar()
print("Imagens de treino:", imagens_treino.shape)
print("Imagens de teste:", imagens_teste.shape)

# %% [markdown]
# ## Quantidade de imagens por classe

# %%
pd.DataFrame({
    "treino": np.bincount(classes_treino),
    "teste": np.bincount(classes_teste),
}, index=CLASSES)

# %% [markdown]
# As 10 classes têm exatamente a mesma quantidade de imagens: 5000 no treino e 1000 no teste. A base é balanceada, então a acurácia é uma boa métrica aqui e não precisamos de técnicas de balanceamento.

# %% [markdown]
# ## Exemplos de cada classe

# %%
fig, eixos = plt.subplots(10, 10, figsize=(11, 12))
for i, classe in enumerate(CLASSES):
    exemplos = np.where(classes_treino == i)[0][:10]
    for j, indice in enumerate(exemplos):
        eixos[i, j].imshow(imagens_treino[indice])
        eixos[i, j].axis("off")
    eixos[i, 0].text(-4, 16, classe, ha="right", va="center", fontsize=10)
plt.suptitle("10 imagens de cada classe")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "3_exemplos.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - As imagens são muito pequenas (32x32 pixels). Mesmo para uma pessoa, algumas são difíceis de reconhecer.
# - Dentro de uma mesma classe as fotos variam muito: pássaros podem ser um avestruz ou um passarinho, aparecer de frente ou de lado, com fundo de céu, grama ou água.
# - O fundo ajuda e atrapalha: aviões e navios costumam ter céu ou água azul atrás, e cervos e cavalos costumam ter grama verde. A rede pode aprender a usar o fundo como pista.
# - Gato e cachorro são parecidos em tamanho, cor e pose (deitados, de frente). Já dá para esperar que esse par seja o mais confundido.

# %% [markdown]
# ## Separando a validação
# Tiramos 5000 imagens do treino (500 de cada classe) para a validação. Ficam 45 mil para treinar. As 10 mil imagens de teste só são usadas no final.

# %%
indices = np.arange(len(imagens_treino))
indices_treino, indices_val = train_test_split(indices, test_size=5000, stratify=classes_treino, random_state=42)
print("Treino:", len(indices_treino))
print("Validação:", len(indices_val))
print("Teste:", len(imagens_teste))

# %% [markdown]
# ## Normalizando
# Calculamos a média e o desvio de cada canal de cor (vermelho, verde e azul) usando só as imagens de treino.

# %%
pixels_treino = imagens_treino[indices_treino] / 255.0
media = torch.tensor(pixels_treino.mean(axis=(0, 1, 2)), dtype=torch.float32, device=device)
desvio = torch.tensor(pixels_treino.std(axis=(0, 1, 2)), dtype=torch.float32, device=device)
del pixels_treino
print("Média (R, G, B):", media.cpu().numpy().round(3))
print("Desvio (R, G, B):", desvio.cpu().numpy().round(3))

# %% [markdown]
# ## Levando os dados para a placa de vídeo

# %%
X_treino = para_tensor(imagens_treino[indices_treino], media, desvio)
X_val = para_tensor(imagens_treino[indices_val], media, desvio)
X_teste = para_tensor(imagens_teste, media, desvio)
y_treino = torch.tensor(classes_treino[indices_treino], device=device)
y_val = torch.tensor(classes_treino[indices_val], device=device)
y_teste = torch.tensor(classes_teste, device=device)

dados = (X_treino, y_treino, X_val, y_val)
conjuntos = {"treino": (X_treino, y_treino), "validação": (X_val, y_val), "teste": (X_teste, y_teste)}
print("Formato do treino:", tuple(X_treino.shape))

# %% [markdown]
# ## Exemplo da data augmentation
# Na primeira coluna está a imagem original e nas outras, versões criadas pela data augmentation, como a CNN vai ver durante o treino.

# %%
fixar_semente()
exemplos = X_treino[:6]
versoes = [exemplos] + [aumentar_dados(exemplos) for _ in range(5)]

def para_exibir(x):
    # desfaz a normalização para mostrar a imagem com as cores certas
    return (x * desvio.view(3, 1, 1) + media.view(3, 1, 1)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

fig, eixos = plt.subplots(6, 6, figsize=(8, 8))
for linha in range(6):
    for coluna in range(6):
        eixos[linha, coluna].imshow(para_exibir(versoes[coluna][linha]))
        eixos[linha, coluna].axis("off")
eixos[0, 0].set_title("original")
for coluna in range(1, 6):
    eixos[0, coluna].set_title(f"versão {coluna}")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "3_augmentation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Cada versão é a mesma imagem deslocada alguns pixels e, às vezes, espelhada. O objeto continua o mesmo, mas os pixels mudam de lugar. Isso ensina a CNN que a posição exata do objeto não importa.

# %% [markdown]
# ---
# # Item a: MLP

# %% [markdown]
# ## Arquitetura

# %%
fixar_semente()
mlp = MLP().to(device)
print(mlp)
print("Total de pesos treináveis:", sum(p.numel() for p in mlp.parameters()))

# %% [markdown]
# A MLP tem:
# - entrada: a imagem achatada em 3072 números;
# - três camadas escondidas com 1024, 512 e 256 neurônios, cada uma com BatchNorm, ReLU e dropout de 30%;
# - saída com 10 neurônios, um por classe.
#
# São cerca de 3,8 milhões de pesos. Quase todos estão na primeira camada (3072 x 1024 = 3,1 milhões), porque cada pixel de cada cor se liga a todos os neurônios.

# %% [markdown]
# ## Treinando
# Otimizador AdamW com taxa de aprendizado 0,001 e weight decay 0,0001, lotes de 128 imagens, até 60 épocas. O treino para se a acurácia da validação ficar 8 épocas sem melhorar.

# %%
fixar_semente()
historico_mlp = treinar(mlp, dados, epocas=60, lr=1e-3, weight_decay=1e-4, paciencia=8, nome="MLP")

# %% [markdown]
# ## Curvas de erro de treino e de validação

# %%
grafico_curvas(historico_mlp, "MLP: erro e acurácia durante o treino", arquivo="3a_curvas_mlp.png")

# %% [markdown]
# O que observar:
# - O erro de treino cai o tempo todo, mas o erro de validação para de cair por volta da época 13 e depois começa a subir. É um sinal claro de overfitting: a rede passa a decorar as imagens de treino.
# - Na acurácia, o treino chega a mais de 80%, enquanto a validação fica parada perto de 58% a 59%.
# - O early stopping parou o treino na época 37, porque a validação ficou 8 épocas sem melhorar. Voltamos para os pesos da melhor época, a 29 (59,2% na validação).
# - Mesmo com dropout e BatchNorm, a MLP não consegue passar de cerca de 59%. O limite não é a quantidade de treino, é o tipo de rede.

# %% [markdown]
# ## Matriz de confusão no teste

# %%
previstos_mlp, probabilidades_mlp = prever(mlp, X_teste)
matriz_mlp = grafico_matriz_confusao(classes_teste, previstos_mlp, "MLP: matriz de confusão no teste", arquivo="3a_matriz_mlp.png")

# %% [markdown]
# O que observar:
# - A acurácia da MLP no teste foi 58,4%. Acerta bem mais que um chute (10%), mas erra muito.
# - A maior confusão é entre gato e cachorro: 224 gatos foram classificados como cachorro e 183 cachorros como gato.
# - Automóvel e caminhão também se confundem bastante (182 automóveis viraram caminhão e 108 caminhões viraram automóvel), e avião é confundido com navio (114), provavelmente por causa do fundo azul.
# - Os animais (pássaro, gato, cervo, cachorro) são as classes mais difíceis. Os erros se espalham por várias colunas, o que mostra que a MLP não está entendendo bem a forma do animal.

# %% [markdown]
# ## Acurácia no teste

# %%
acuracia_mlp = (previstos_mlp == classes_teste).mean()
print(f"Acurácia da MLP no teste: {acuracia_mlp:.2%}")

# %%
maiores_confusoes(matriz_mlp)

# %% [markdown]
# ---
# # Item b: CNN

# %% [markdown]
# ## Arquitetura

# %%
fixar_semente()
cnn = CNN().to(device)
print(cnn)
print("Total de pesos treináveis:", sum(p.numel() for p in cnn.parameters()))

# %% [markdown]
# A CNN tem:
# - bloco 1: duas convoluções com 64 filtros de 3 x 3, BatchNorm e ReLU, MaxPool (32 x 32 vira 16 x 16) e dropout de 10%;
# - bloco 2: duas convoluções com 128 filtros, MaxPool (16 x 16 vira 8 x 8) e dropout de 20%;
# - bloco 3: duas convoluções com 256 filtros, MaxPool (8 x 8 vira 4 x 4) e dropout de 30%;
# - global average pooling: a média de cada um dos 256 mapas;
# - camada linear de 256 para 10 classes.
#
# A CNN tem cerca de 1,15 milhão de pesos, menos de um terço da MLP. Isso acontece porque cada filtro de 3 x 3 é usado em todas as posições da imagem: a rede reaproveita os mesmos pesos, em vez de ter um peso para cada pixel.

# %% [markdown]
# ## Treinando
# - Otimizador AdamW com weight decay 0,0005 e agendador OneCycle, com taxa de aprendizado máxima de 0,003.
# - 40 épocas com lotes de 128 imagens e data augmentation.
# - Guardamos os pesos das épocas 1, 5 e 40 em arquivos para a Questão 4.
#
# Aqui não usamos early stopping: com o OneCycle a taxa de aprendizado só fica bem pequena nas últimas épocas, e é nelas que a rede costuma dar o último salto de acurácia. Mesmo assim, guardamos os pesos da melhor época na validação.

# %%
fixar_semente()
extras = {"media": media.cpu(), "desvio": desvio.cpu()}
historico_cnn = treinar(cnn, dados, epocas=40, lr=3e-3, weight_decay=5e-4, one_cycle=True, aumentar=True,
                        salvar_epocas=(1, 5, 40), nome_arquivo="cnn", extras=extras, nome="CNN")
salvar_modelo(cnn, PASTA_MODELOS / "cnn_final.pt", historico_cnn["melhor_epoca"], extras)

# %% [markdown]
# ## Curvas de erro de treino e de validação

# %%
grafico_curvas(historico_cnn, "CNN: erro e acurácia durante o treino", arquivo="3b_curvas_cnn.png")

# %% [markdown]
# O que observar:
# - Treino e validação caem juntos durante quase todo o treino. A curva de validação oscila bastante nas primeiras 15 épocas, por causa da taxa de aprendizado alta no início do OneCycle, e depois fica estável.
# - No final a taxa de aprendizado fica bem pequena, e as duas curvas se estabilizam. A melhor época foi a 37, com 92,8% de acurácia na validação.
# - Existe uma diferença entre treino (cerca de 97%) e validação (cerca de 93%), mas o erro de validação não sobe em nenhum momento. Ou seja, a rede aprende mais do que decora.
# - O treino é feito com data augmentation (imagens espelhadas e deslocadas), então a curva de treino é calculada em imagens "mais difíceis" que as de validação.

# %% [markdown]
# ## Matriz de confusão no teste

# %%
previstos_cnn, probabilidades_cnn = prever(cnn, X_teste)
matriz_cnn = grafico_matriz_confusao(classes_teste, previstos_cnn, "CNN: matriz de confusão no teste", arquivo="3b_matriz_cnn.png")

# %% [markdown]
# O que observar:
# - A acurácia da CNN no teste foi 92,0%. A diagonal ficou bem mais escura que a da MLP.
# - O par mais confundido continua sendo gato e cachorro (82 gatos viraram cachorro e 62 cachorros viraram gato), mas são cerca de 3 vezes menos erros que na MLP.
# - Os outros erros aparecem entre classes realmente parecidas: caminhão e automóvel (31 e 26), navio e avião (30 e 20), pássaro e cervo ou sapo (cerca de 29 cada).
# - Quase não há confusão entre veículos e animais. A CNN aprendeu a separar bem os dois grandes grupos.

# %% [markdown]
# ## Acurácia no teste

# %%
acuracia_cnn = (previstos_cnn == classes_teste).mean()
print(f"Acurácia da CNN no teste: {acuracia_cnn:.2%}")

# %%
maiores_confusoes(matriz_cnn)

# %% [markdown]
# ---
# # Verificando overfitting
# Para o treino medimos a acurácia nas imagens originais, sem data augmentation.

# %%
pd.concat({
    "MLP": tabela_overfitting(mlp, conjuntos),
    "CNN": tabela_overfitting(cnn, conjuntos),
}, axis=1).round(4)

# %% [markdown]
# Na tabela, a acurácia no treino é medida sem dropout e sem data augmentation, por isso é maior que a da curva.
#
# - MLP: 90,0% no treino, mas só 59,2% na validação e 58,4% no teste. São mais de 30 pontos de diferença. A MLP tem overfitting forte: decora as imagens de treino e não generaliza. O early stopping e o dropout limitam o estrago, mas não resolvem.
# - CNN: 99,3% no treino, 92,8% na validação e 92,0% no teste. A diferença é de cerca de 7 pontos, bem menor. Existe um pouco de overfitting, mas a rede generaliza bem.
# - Nos dois casos, validação e teste ficam muito próximos. Isso mostra que escolher o modelo pela validação foi uma boa estimativa do resultado em imagens novas.

# %% [markdown]
# ---
# # Comparando a MLP e a CNN

# %% [markdown]
# ## Acurácia por classe

# %%
por_classe = pd.DataFrame({
    "MLP": acuracia_por_classe(classes_teste, previstos_mlp),
    "CNN": acuracia_por_classe(classes_teste, previstos_cnn),
})

por_classe.plot.bar(figsize=(11, 4.5), rot=45, color=["tab:gray", "tab:blue"])
plt.ylabel("acurácia no teste")
plt.ylim(0, 1)
plt.title("Acurácia por classe no teste")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PASTA_FIGURAS / "3_comparacao_classes.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# O que observar:
# - A CNN ganha da MLP em todas as 10 classes.
# - As classes mais difíceis são as mesmas nos dois modelos: gato (38,8% na MLP e 82,4% na CNN), pássaro (43,8% e 87,1%) e cachorro (52,5% e 88,6%).
# - A maior melhora aparece justamente nos animais: gato sobe 44 pontos, pássaro 43 e cervo 43. Para reconhecer um animal é preciso entender formas como orelhas, patas e focinho, e é isso que as convoluções fazem bem.
# - Os veículos já eram as classes mais fáceis para a MLP, por causa da cor e do fundo. Mesmo assim a CNN sobe todos para mais de 93%.

# %% [markdown]
# ## Resumo

# %%
pd.DataFrame({
    "pesos treináveis": [sum(p.numel() for p in mlp.parameters()), sum(p.numel() for p in cnn.parameters())],
    "épocas treinadas": [len(historico_mlp["erro_treino"]), len(historico_cnn["erro_treino"])],
    "acurácia no teste": [acuracia_mlp, acuracia_cnn],
}, index=["MLP", "CNN"])

# %% [markdown]
# # Conclusão
#
# - A MLP chegou a 58,4% de acurácia no teste e a CNN chegou a 92,0%, mesmo tendo cerca de 3 vezes menos pesos (1,1 milhão contra 3,8 milhões).
# - A MLP recebe a imagem "achatada" em uma lista de 3072 números. Ela perde a noção de quais pixels são vizinhos e precisa aprender cada padrão separadamente para cada posição da imagem. Por isso decora o treino (90%) e generaliza mal.
# - A CNN olha pequenos pedaços da imagem e usa o mesmo filtro em todas as posições. Assim ela reconhece uma borda, uma textura ou uma orelha em qualquer lugar da foto. Juntando isso com BatchNorm, dropout e data augmentation, o overfitting ficou pequeno (99% no treino e 92% no teste).
# - Nos dois modelos o erro mais comum é gato com cachorro, que também é difícil para pessoas em imagens de 32x32 pixels.
# - Para a Q4 salvamos os pesos da CNN nas épocas 1, 5 e no final, para ver o que a rede aprendeu.
