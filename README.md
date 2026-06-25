# 🎬 Sistemas de Recomendação: Filtragem Colaborativa (From Scratch)

Este repositório contém a implementação do zero (sem uso de frameworks de Machine Learning como Scikit-Learn ou Surprise) de algoritmos clássicos de Sistemas de Recomendação. O objetivo do projeto é analisar, hiperparametrizar e comparar abordagens de Filtragem Colaborativa baseadas em memória e baseadas em modelo, utilizando o clássico dataset **MovieLens 100k**.

## 🚀 Algoritmos Implementados

A biblioteca de modelos foi desenvolvida utilizando operações vetorizadas em NumPy para máxima performance matemática. Foram implementados:

* **Baseline (Média Global):** Preditor simples utilizado como limite inferior de comparação.
* **KNN User-Based:** Algoritmo *lazy* baseado em memória, utilizando a Similaridade do Cosseno sobre matrizes centralizadas na média (*mean-centered*).
* **KNN Item-Based:** Variante focada na similaridade entre filmes, explorando a maior estabilidade do perfil de itens em relação aos usuários.
* **SVD (Singular Value Decomposition):** Modelo paramétrico de fatoração de matrizes regularizado, otimizado via Descida de Gradiente Estocástico (SGD) para aprender vetores latentes.
* **SVD++:** Extensão do SVD que incorpora *feedback* implícito, utilizando o histórico de interações do usuário (quais filmes ele avaliou) para enriquecer a representação latente.

## 📊 Metodologia e Avaliação

Para garantir o rigor estatístico e evitar vazamento de dados (*data leakage*):
* Foi implementada uma rotina própria de **Validação Cruzada K-Fold ($K=5$)**.
* **Inner Loop:** Utilizado os 3 primeiros *folds* exclusivamente para a hiperparametrização (busca do melhor $K$ para o KNN e do melhor número de fatores latentes $f$ para o SVD/SVD++).
* **Outer Loop:** Utilizado os 5 *folds* completos para a avaliação final de desempenho.
* **Métricas:** O Erro Absoluto Médio (MAE) e a Raiz do Erro Quadrático Médio (RMSE) foram utilizados para mensurar a acurácia da predição de notas (escala de 1 a 5).

## 🏆 Principais Resultados

O SVD++ apresentou o melhor desempenho geral, evidenciando o poder de aliar fatores latentes com o comportamento implícito dos usuários. Um resumo da performance dos modelos parametrizados (média de 5 folds):

| Algoritmo | RMSE | MAE | Hiperparâmetro Ótimo |
| :--- | :---: | :---: | :--- |
| **Baseline (Média)** | 1.1257 | 0.9447 | - |
| **KNN User-Based** | 0.9293 | 0.7301 | $K = 30$ |
| **KNN Item-Based** | 0.9233 | 0.7250 | $K = 30$ |
| **SVD Regularizado** | 0.9234 | 0.7315 | $f = 100$ |
| **SVD++** | **0.9178** | **0.7242** | $f = 50$ |

O código também gera automaticamente gráficos de análise cobrindo:
1. Curvas de hiperparametrização (impacto do ruído vs. estabilidade).
2. Curvas de convergência do treinamento (SGD ao longo de épocas).
3. Estabilidade do erro por *fold*.

## 🛠️ Como Executar

### Pré-requisitos
Certifique-se de ter o Python 3.8+ instalado. As bibliotecas necessárias são apenas para manipulação matricial e visualização de dados:

```bash
pip install numpy pandas matplotlib seaborn
