# Deep Learning: Lista de Exercícios 2026.2

Resolução da lista de exercícios de Aprendizado Profundo (UFPB, Prof. Tiago Maritan). O enunciado está em [`doc/`](doc/).

**Integrantes:** Humberto Nunes de Lira, Wesley Henrique da Silva Vieira e Carlos Vitor Freitas Santos.

Cada questão tem um notebook próprio, que roda sozinho do começo ao fim. Todos seguem a mesma estrutura: funções no início (cada uma com uma explicação), um passo por bloco, cada gráfico seguido de uma análise e uma seção de overfitting.

## Notebooks

| Questão | Notebook | Assunto | Principais resultados (teste) |
|---|---|---|---|
| 1 | [Q1_MLP_regressao.ipynb](notebooks/Q1_MLP_regressao.ipynb) | MLP para regressão: (a) função 3sin(2x) + 2cos(5x) + 0,5x²; (b) base Diabetes | (a) RMSE 0,014; (b) RMSE 56,0 e R² 0,45 (empata com a regressão linear) |
| 2 | [Q2_wine_quality.ipynb](notebooks/Q2_wine_quality.ipynb) | MLP para classificar vinhos em 3 classes de qualidade, com EDA e balanceamento | acurácia 72,7%, F1 macro 0,53 |
| 3 | [Q3_cifar10.ipynb](notebooks/Q3_cifar10.ipynb) | CIFAR-10: (a) MLP x (b) CNN | MLP 57,9% x CNN 91,9% de acurácia (MLP com a receita de treino da CNN: 61,0%) |
| 4 | [Q4_cnn_interpretacao.ipynb](notebooks/Q4_cnn_interpretacao.ipynb) | Interpretação da CNN da Q3: filtros, mapas de ativação, oclusão, acertos e erros | confiança média de 97% nos acertos e 76% nos erros |
| 5 | [Q5_autoencoders.ipynb](notebooks/Q5_autoencoders.ipynb) | Fashion-MNIST: (a) autoencoders com latente 100/50/30; (b) autoencoder que remove ruído | MSE 0,0040 / 0,0057 / 0,0075; ruído: PSNR de 12,9 para 22,7 dB |

As figuras geradas ficam em [`figures/`](figures/), uma pasta por questão.

Além do que o enunciado pede, os notebooks trazem alguns experimentos de conferência:

- **Q1:** a rede fora do intervalo de treino (extrapolação) e a mesma rede com ReLU no lugar da Tanh.
- **Q2:** validação cruzada das quatro estratégias de balanceamento e teste com e sem a densidade.
- **Q3:** a MLP treinada com a mesma receita da CNN, para separar o efeito da arquitetura do efeito do jeito de treinar.
- **Q4:** filtros e ativações também antes do treino (época 0), teste de oclusão e tabela de calibração da confiança.
- **Q5:** o autoencoder do item a aplicado às imagens ruidosas, como comparação para o item b.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (no Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
jupyter notebook notebooks/
```

- As bases são baixadas automaticamente para a pasta `data/` na primeira execução.
- **Rode a Q3 antes da Q4.** A Q3 salva os pesos da CNN em `checkpoints/`, e a Q4 carrega esses pesos.
- Os notebooks gravados no repositório foram executados com Python 3.13, PyTorch 2.11 com CUDA 12.8 e uma placa RTX 4050 Laptop de 6 GB.
- Tempo medido nessa placa, sem contar o download das bases: Q1 ~6 min, Q2 ~16 min (a maior parte é a validação cruzada), Q3 ~7 min, Q4 ~1 min, Q5 ~3 min.
- As sementes são fixas em todos os notebooks e, nos que usam convolução (Q3 e Q5), a GPU também é posta em modo determinístico. Na mesma máquina os resultados se repetem a cada execução. Entre máquinas diferentes os números das redes convolucionais mudam um pouco (veja o benchmark abaixo).

## Benchmark de comparação

A pasta [`benchmark/`](benchmark/) compara duas execuções completas dos notebooks: a original (RTX 3050, Python 3.14) e a atual (RTX 4050, Python 3.13). A tabela completa está em [`benchmark/comparacao.md`](benchmark/comparacao.md). Em resumo:

- **Q1 e Q2** (redes densas em dados tabulares): os resultados são idênticos nas duas máquinas em todas as métricas de teste.
- **Q3 e Q4** (CNN): a acurácia da CNN muda de 91,96% para 91,93% e a da MLP de 58,44% para 57,87%. A melhor época também muda. As conclusões são as mesmas.
- **Q5** (autoencoders): diferenças de menos de 1% no MSE.
- **Tempo de treino da CNN:** 7,3 min na RTX 3050 e 4,2 min na RTX 4050.

Para refazer a comparação depois de executar os notebooks em outra máquina:

```bash
python notebooks/_src/benchmark.py extrair notebooks benchmark/minha_maquina.json "descrição do ambiente"
python notebooks/_src/benchmark.py comparar benchmark/original_rtx3050.json benchmark/minha_maquina.json benchmark/comparacao.md
```

## Estrutura

```
notebooks/        notebooks das questões (já executados, com as saídas)
notebooks/_src/   fonte de cada notebook em formato .py (py2nb.py converte para .ipynb)
                  e benchmark.py, que extrai e compara as métricas
figures/          figuras geradas pelos notebooks
benchmark/        métricas de cada execução e a tabela de comparação
doc/              enunciado
```
