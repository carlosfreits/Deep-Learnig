# Análise do repositório de referência

Repositório: https://github.com/talesnobre/Deep-Learning (Listas 1 e 2 de 2024, **não são as mesmas questões** desta lista).

## O que tem lá

| Notebook | Conteúdo | Framework | Resultado |
|---|---|---|---|
| Lista 1/Q1 | Perceptron de Rosenblatt do zero, 8 classes (vértices do cubo) | NumPy puro | 100% |
| Lista 1/Q2 | MLP para XOR e aproximação de funções (SGD × Adam) | PyTorch | — |
| Lista 1/Q3 | Perceptron com e sem momento, 8 regiões no círculo | PyTorch | 93,6% / 97% |
| Lista 1/Q4 | MLP para série temporal (10 entradas → 3 passos) | PyTorch | RMSE 0,0014 |
| Lista 2/Q1 | MLP Titanic | Keras | ~85% |
| Lista 2/Q2 | MNIST: MLP × CNN | PyTorch | 98,1% / 99,0% |
| Lista 2/Q3 | CNN CIFAR-10 (VGG-like, 7 épocas) | PyTorch | 76,4% |

## O que dá para aproveitar

| Questão nossa | Referência | Uso |
|---|---|---|
| Q1a / Q1b (regressão MLP) | Lista 1/Q2, Q4 | Loop de treino com histórico de loss de treino e de teste por época, plot das curvas |
| Q2 (Wine) | Lista 2/Q1 | Estrutura de pré-processamento e avaliação com matriz de confusão |
| Q3a (MLP CIFAR) | Lista 2/Q2 (MLP MNIST) | Arquitetura MLP com camadas decrescentes, heatmap da matriz de confusão |
| Q3b (CNN CIFAR) | Lista 2/Q3 | Arquitetura VGG-like, split de validação de 10%, plots de loss/acc |
| Q4c (acertos/erros) | Lista 2/Q2 | Grid de imagens com rótulos "Predito / Real" |
| Q4a, Q4b, Q5 | — | Nada pronto, implementação nova |

## Problemas encontrados (não repetir)

1. **Softmax + CrossEntropyLoss** (Lista 2/Q2): o `CrossEntropyLoss` do PyTorch já aplica log-softmax. Aplicar Softmax antes achata os gradientes (por isso a loss dele ficou travada em ~1,47).
2. **Seleção do modelo e early stopping pelo conjunto de teste** (Lista 2/Q2, Lista 1/Q4): é vazamento de dados, porque o teste deixa de ser independente. Usar um conjunto de **validação** separado.
3. **Normalização com estatísticas do dataset inteiro** (Lista 2/Q1): o scaler deve ser ajustado só no treino.
4. **`torch.cuda.is_available` sem `()`** (Lista 2/Q3): é sempre truthy, então quebra em máquina sem GPU.
5. **CNN CIFAR sem BatchNorm, Dropout ou data augmentation, treinada por 7 épocas**: 76% é pouco. Com BN, augmentation e scheduler dá para chegar a ≥85%.
6. Curva de loss de treino registrada só pelo último batch (Lista 1/Q3): é ruidosa. Usar a média da época.
