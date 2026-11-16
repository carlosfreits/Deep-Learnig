# Deep Learning: Lista de Exercícios 2026.2

Resolução da lista de exercícios de Aprendizado Profundo (UFPB, Prof. Tiago Maritan). O enunciado está em [`doc/`](doc/).

Cada questão tem um notebook próprio, que roda sozinho do começo ao fim. Todos seguem a mesma estrutura: funções no início (cada uma com uma explicação), um passo por bloco, cada gráfico seguido de uma análise e uma seção de overfitting.

## Notebooks

| Questão | Notebook | Assunto | Principais resultados (teste) |
|---|---|---|---|
| 1 | [Q1_MLP_regressao.ipynb](notebooks/Q1_MLP_regressao.ipynb) | MLP para regressão: (a) função 3sin(2x) + 2cos(5x) + 0,5x²; (b) base Diabetes | (a) RMSE 0,014; (b) RMSE 56,0 e R² 0,45 (empata com a regressão linear) |
| 2 | [Q2_wine_quality.ipynb](notebooks/Q2_wine_quality.ipynb) | MLP para classificar vinhos em 3 classes de qualidade, com EDA e balanceamento | acurácia 72,7%, F1 macro 0,53 |
| 3 | [Q3_cifar10.ipynb](notebooks/Q3_cifar10.ipynb) | CIFAR-10: (a) MLP x (b) CNN | MLP 58,4% x CNN 92,0% de acurácia |
| 4 | [Q4_cnn_interpretacao.ipynb](notebooks/Q4_cnn_interpretacao.ipynb) | Interpretação da CNN da Q3: filtros, mapas de ativação, acertos e erros | confiança média de 97% nos acertos e 77% nos erros |
| 5 | [Q5_autoencoders.ipynb](notebooks/Q5_autoencoders.ipynb) | Fashion-MNIST: (a) autoencoders com latente 100/50/30; (b) autoencoder que remove ruído | MSE 0,0040 / 0,0058 / 0,0074; ruído: PSNR de 12,9 para 22,7 dB |

As figuras geradas ficam em [`figures/`](figures/), uma pasta por questão.

## Como rodar

O ambiente usado foi Python 3.14, PyTorch 2.11 com CUDA 12.8 e uma placa RTX 3050 de 4 GB.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (no Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
jupyter notebook notebooks/
```

- As bases são baixadas automaticamente para a pasta `data/` na primeira execução.
- **Rode a Q3 antes da Q4.** A Q3 salva os pesos da CNN em `checkpoints/`, e a Q4 carrega esses pesos.
- Tempo medido com a RTX 3050: Q1 ~3,5 min, Q2 ~1,5 min, Q3 ~9 min, Q4 ~1 min, Q5 ~4 min.
- As sementes são fixas e a GPU usa algoritmos determinísticos, então os resultados se repetem a cada execução (conferido rodando de novo a Q3 e a Q5: mesmos pesos e mesmos números).

## Estrutura

```
notebooks/        notebooks das questões (já executados, com as saídas)
notebooks/_src/   fonte de cada notebook em formato .py (py2nb.py converte para .ipynb)
figures/          figuras geradas pelos notebooks
doc/              enunciado e anotações
```
