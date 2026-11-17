# Benchmark de comparação

Compara as métricas principais de duas execuções dos notebooks. Os valores são lidos das saídas gravadas nos próprios notebooks pelo script [`notebooks/_src/benchmark.py`](../notebooks/_src/benchmark.py).

- **Original:** RTX 3050 (4 GB), Python 3.14, PyTorch 2.11 + CUDA 12.8, notebooks do commit 013a455
- **Atual:** RTX 4050 Laptop (6 GB), Python 3.13, PyTorch 2.11 + CUDA 12.8, notebooks com as correções

A coluna "variação" é (atual − original) / original. Métricas que só existem em uma das execuções aparecem com um traço na outra.


## Questão 1

| Item | Métrica | Original | Atual | Diferença | Variação |
|---|---|---:|---:|---:|---:|
| Q1a | MAE (teste) | 0.01141 | 0.01141 | +0 | +0.0% |
| Q1a | RMSE (teste) | 0.01384 | 0.01384 | +0 | +0.0% |
| Q1b | MLP: MAE (teste) | 45.98 | 45.98 | +0 | +0.0% |
| Q1b | MLP: RMSE (teste) | 55.95 | 55.95 | +0 | +0.0% |
| Q1b | MLP: R² (teste) | 0.45 | 0.45 | +0 | +0.0% |
| Q1b | Regressão linear: RMSE (teste) | 56.03 | 56.03 | +0 | +0.0% |
| Q1b | MLP: RMSE médio (validação cruzada) | 55.12 | 55.12 | +0 | +0.0% |
| Q1b | Regressão linear: RMSE médio (validação cruzada) | 54.89 | 54.89 | +0 | +0.0% |

## Questão 2

| Item | Métrica | Original | Atual | Diferença | Variação |
|---|---|---:|---:|---:|---:|
| Q2 | acurácia (teste) | 0.727 | 0.727 | +0 | +0.0% |
| Q2 | precisão macro (teste) | 0.52 | 0.52 | +0 | +0.0% |
| Q2 | recall macro (teste) | 0.547 | 0.547 | +0 | +0.0% |
| Q2 | F1 macro (teste) | 0.531 | 0.531 | +0 | +0.0% |
| Q2 | Random Forest: F1 macro (teste) | 0.537 | 0.537 | +0 | +0.0% |
| Q2 | F1 macro na validação: sem balanceamento | 0.55 | 0.548 | -0.002 | -0.4% |
| Q2 | F1 macro na validação: pesos nas classes | 0.555 | 0.555 | +0 | +0.0% |
| Q2 | F1 macro na validação: pesos suavizados | 0.603 | 0.603 | +0 | +0.0% |
| Q2 | F1 macro na validação: SMOTE | 0.54 | 0.54 | +0 | +0.0% |
| Q2 | estratégia escolhida | pesos suavizados | pesos suavizados | igual | igual |

## Questão 3

| Item | Métrica | Original | Atual | Diferença | Variação |
|---|---|---:|---:|---:|---:|
| Q3 | MLP: acurácia (teste, %) | 58.44 | 57.87 | -0.57 | -1.0% |
| Q3 | CNN: acurácia (teste, %) | 91.96 | 91.93 | -0.03 | -0.0% |
| Q3 | MLP: tempo de treino (min) | 1.60 | 1.10 | -0.5 | -31.2% |
| Q3 | CNN: tempo de treino (min) | 7.30 | 4.20 | -3.1 | -42.5% |
| Q3 | MLP: melhor época | 29.00 | 21.00 | -8 | -27.6% |
| Q3 | CNN: melhor época | 37.00 | 38.00 | +1 | +2.7% |

## Questão 4

| Item | Métrica | Original | Atual | Diferença | Variação |
|---|---|---:|---:|---:|---:|
| Q4 | acurácia depois da época 1 (%) | 56.93 | 54.98 | -1.95 | -3.4% |
| Q4 | acurácia depois da época 5 (%) | 70.66 | 69.06 | -1.6 | -2.3% |
| Q4 | confiança média nos acertos (%) | 97.30 | 97.30 | +0 | +0.0% |
| Q4 | confiança média nos erros (%) | 77.40 | 76.20 | -1.2 | -1.6% |
| Q4 | erros com confiança acima de 90% (%) | 34.20 | 33.00 | -1.2 | -3.5% |

## Questão 5

| Item | Métrica | Original | Atual | Diferença | Variação |
|---|---|---:|---:|---:|---:|
| Q5a | latente 100: MSE (teste) | 0.004 | 0.00397 | -3e-05 | -0.8% |
| Q5a | latente 100: PSNR (teste, dB) | 25.45 | 25.51 | +0.05508 | +0.2% |
| Q5a | latente 50: MSE (teste) | 0.00578 | 0.00575 | -3e-05 | -0.5% |
| Q5a | latente 50: PSNR (teste, dB) | 23.80 | 23.82 | +0.01917 | +0.1% |
| Q5a | latente 30: MSE (teste) | 0.00743 | 0.00748 | +5e-05 | +0.7% |
| Q5a | latente 30: PSNR (teste, dB) | 22.58 | 22.55 | -0.03765 | -0.2% |
| Q5b | 10 imagens: MSE com ruído | 0.0519 | 0.0519 | +0 | +0.0% |
| Q5b | 10 imagens: MSE depois da rede | 0.0063 | 0.0063 | +0 | +0.0% |
| Q5b | 10 imagens: PSNR com ruído (dB) | 12.86 | 12.86 | +0 | +0.0% |
| Q5b | 10 imagens: PSNR depois da rede (dB) | 22.66 | 22.66 | +0.001 | +0.0% |
| Q5b | teste inteiro: MSE depois da rede | 0.00801 | 0.00801 | +0 | +0.0% |
