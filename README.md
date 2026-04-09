# Classificação de Categorias de Preços de Imóveis (California Housing)

Este projeto realiza a classificação de imóveis da Califórnia em faixas de preço (**Baixo**, **Médio** e **Alto**) utilizando o algoritmo **XGBoost Classifier**. O foco está na engenharia de atributos (binning) e na avaliação de performance para problemas multiclasse.

---

## 🎯 Objetivo do Projeto
Transformar um problema de regressão original em um problema de **classificação**, agrupando os valores dos imóveis em 3 categorias balanceadas através da técnica de quantis (`qcut`), e treinar um modelo para prever a qual categoria uma casa pertence com base em suas características geográficas e socioeconômicas.

---

## 🛠️ Tecnologias Utilizadas
* **Pandas**: Manipulação de dados e criação dos bins de preço.
* **Matplotlib & Seaborn**: Visualização da distribuição das classes e matriz de confusão.
* **Scikit-learn**: Codificação de variáveis categóricas e métricas de avaliação (Acurácia, Report).
* **XGBoost**: Algoritmo de classificação de alto desempenho.

---

## 📈 Etapas do Pipeline

1.  **Engenharia de Labels (Binning)**: Utilização do `pd.qcut` para dividir o valor dos imóveis em três categorias com quantidades iguais de dados.
2.  **Análise Visual**: Plotagem de histograma com as linhas de corte (bins) para validar a separação das categorias.
3.  **Pré-processamento**:
    * Transformação das categorias de texto para numéricas.
    * Codificação da variável `proximidade_oceano`.
4.  **Treinamento**:
    * Separação em conjuntos de Treino e Teste (80/20).
    * Aplicação do `XGBClassifier` com 100 estimadores.
5.  **Avaliação de Performance**:
    * **Acurácia Global**.
    * **Classification Report**: Precision, Recall e F1-Score para cada categoria.
    * **Matriz de Confusão**: Visualização de erros e acertos por classe via Heatmap.

---

## 📋 Pré-requisitos
Instale as bibliotecas necessárias antes de executar:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost
