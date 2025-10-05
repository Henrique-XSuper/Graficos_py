# 🧠 Exemplos de Redes Neurais e Perceptron em Python

Este repositório contém diversos exemplos implementados em **Python** utilizando `NumPy`, `Matplotlib` e `scikit-learn`.
Os códigos cobrem desde a implementação manual do **Perceptron** até modelos mais avançados como **MLP (Multi-Layer Perceptron)** aplicados em classificação, regressão e problemas clássicos de lógica.

---

## 📂 Estrutura dos Exemplos

### 1. Perceptron Simples — Classificação de Aprovação/Reprovação

* Implementação do algoritmo **Perceptron** do zero.
* Dataset fictício com notas em duas matérias.
* Treinamento e visualização da **fronteira de decisão**.
* Saída: gráfico mostrando **aprovados (verde)** e **reprovados (vermelho)**.

### 2. Portas Lógicas (AND / OR)

* Treinamento de perceptrons para simular portas **AND** e **OR**.
* Visualização da fronteira de decisão para cada porta lógica.

### 3. Classificação de Caranguejos (4 features)

* Geração de dados sintéticos com 4 atributos.
* Uso de `MLPClassifier` (scikit-learn).
* Visualização em 2D usando PCA.
* Avaliação com métricas de **acurácia** e **relatório de classificação**.

### 4. Jacobiana e Hessiana com MLP

* Uso de `MLPRegressor` para aproximar funções não lineares.
* Cálculo numérico de **jacobiana** e **hessiana** do modelo.

### 5. Problema do XOR

* Implementação de um MLP para resolver o problema clássico do **XOR**.
* Visualização da **fronteira de decisão** aprendida pela rede.

### 6. Classificação de Vinhos

* Dataset `wine` do scikit-learn.
* Rede neural MLP para classificação.
* Visualização com **PCA**.
* Plotagem de **matriz de confusão** e **perda por época**.

### 7. Simulação de LSTM com Regressão Linear

* Previsão de sequência numérica simples.
* Demonstração de como uma regressão linear pode simular aprendizado sequencial.

### 8. Predição de Séries Temporais

* Geração de dados senoidais com ruído.
* Uso de `MLPRegressor` para prever valores futuros.
* Avaliação com **RMSE** e visualização de predições.

### 9. Classificação de Dígitos (MNIST-like)

* Uso do dataset `digits` do scikit-learn.
* Rede neural para identificar se o dígito é **5** ou não.
* Exibição de **matriz de confusão** e amostras classificadas.

### 10. Implementação de GRU (Demo)

* Implementação manual de uma **GRU (Gated Recurrent Unit)** simplificada.
* Apenas forward pass (sem treinamento).
* Demonstração em uma série temporal senoidal.

---

## 🚀 Como Executar

1. Clone este repositório:

   ```bash
   git clone https://github.com/Henrique-XSuper/Redes_Neurais.git
   cd seu-repositorio
   ```

2. Instale as dependências:

   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. Execute qualquer exemplo:

   ```bash
   python exemplo1_perceptron.py
   ```

---

## 📊 Dependências

* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-learn](https://scikit-learn.org/)

---

## 🎯 Objetivo

Este repositório serve como **material de estudo prático** para quem deseja aprender fundamentos de **redes neurais, perceptrons, classificação e regressão** de forma simples e visual.


