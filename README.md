# New York City Yellow Taxi - Trip Fare Prediction Project
[ 🇧🇷 Português | 🇺🇸 English ]

---

## 🇧🇷 Apresentação do Projeto (Português)

Este repositório contém a versão modularizada do projeto de Machine Learning para prever o valor de corridas de táxi amarelo em **Nova York**. O projeto utiliza dados históricos de viagens para analisar padrões relacionados à distância, duração, horário, rota, quantidade de passageiros e características operacionais da corrida, com o objetivo de estimar o `fare_amount` de forma precisa e interpretável.

### Objetivos
*   **Prever o valor da tarifa da corrida** com base em variáveis históricas e características da viagem.
*   **Analisar fatores que influenciam o preço final**, como distância, duração, horário de embarque, dia da semana, mês e rota.
*   **Aplicar engenharia de atributos** para criar variáveis temporais, indicadores de horário de pico, períodos do dia e médias por rota.
*   **Treinar e comparar múltiplos modelos de regressão** supervisionada (`Linear Regression`, `HistGradientBoostingRegressor`, `Decision Tree`, `Random Forest` e `XGBoost`).

### Principais Descobertas
*   A distância e a duração da corrida são os principais fatores associados ao valor da tarifa.
*   O tratamento de outliers é uma etapa crítica, pois valores extremos de `fare_amount`, `trip_distance` e `trip_duration` impactam fortemente o treinamento dos modelos.
*   Variáveis temporais, como hora de embarque, mês, dia da semana e horário de pico, enriquecem o conjunto de dados e melhoram a capacidade explicativa dos modelos.
*   Modelos baseados em árvores e boosting tendem a capturar melhor relações não lineares entre as características da viagem e o valor da tarifa.

---

## 🇺🇸 Project Overview (English)

This repository contains the modularized version of the Machine Learning project for predicting **New York City Yellow Taxi** trip fares. The project uses historical taxi trip data to analyze patterns related to distance, duration, pickup time, route, passenger count, and operational trip attributes, aiming to estimate `fare_amount` accurately and interpretably.

### Objectives
*   **Predict taxi trip fare amount** based on historical trip features.
*   **Analyze the main drivers of fare price**, including trip distance, trip duration, pickup hour, weekday, month, and route.
*   **Apply feature engineering** to create temporal variables, rush-hour indicators, day-period groups, and route-level averages.
*   **Train and compare multiple supervised regression models** (`Linear Regression`, `HistGradientBoostingRegressor`, `Decision Tree`, `Random Forest`, and `XGBoost`).

### Key Insights
*   Trip distance and trip duration are the strongest factors associated with fare amount.
*   Outlier treatment is critical because extreme `fare_amount`, `trip_distance`, and `trip_duration` values strongly affect model training.
*   Temporal features such as pickup hour, month, weekday, and rush-hour indicators enrich the dataset and improve model explanatory power.
*   Tree-based and boosting models tend to capture nonlinear relationships between trip characteristics and fare amount more effectively.

---

## 📁 Estrutura de Pastas / Project Structure

O projeto foi estruturado seguindo a mesma organização modular do projeto de classificação HR Salifort Motors, separando o notebook exploratório do pipeline executável e dos módulos reutilizáveis:

```text
regression/
│
├── README.md                              # Documentação principal do projeto
├── requirements.txt                       # Dependências e bibliotecas do projeto
├── main.py                                # Script principal/orquestrador do pipeline
├── yellow_taxi_regression.ipynb      # Notebook original mantido na raiz
│
├── data/
│   └── processed/
│       └── 2017_Yellow_Taxi_Trip_Data.csv
│
├── src/                                   # Módulos de código-fonte
│   ├── __init__.py                        # Inicialização do pacote Python
│   ├── config.py                          # Constantes, caminhos e configurações globais
│   ├── data_loader.py                     # Carregamento dos dados e visão geral do dataset
│   ├── feature_engineering.py             # Criação de features base e avançadas
│   ├── eda.py                             # Análise exploratória, outliers, VIF e visualizações
│   ├── model_training.py                  # Split, escala, tuning e treinamento dos modelos
│   ├── model_evaluation.py                # Métricas, diagnósticos, curva de aprendizado e avaliação segmentada
│   └── insights.py                        # Análise estatística por tipo de pagamento
│
├── models/                                # Pasta para modelos e scaler serializados (.pkl)
└── outputs/
    └── figures/                           # Gráficos gerados automaticamente pelo pipeline
```

---

## 🚀 Como Executar o Projeto / How to Run the Project

### Pré-requisitos
Certifique-se de possuir Python 3.8+ instalado em sua máquina.

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd google_coursera/regression
```

### 2. Instalar as dependências
Recomenda-se o uso de um ambiente virtual (`venv`):

```bash
# Criar e ativar o ambiente virtual (Windows)
python -m venv venv
venv\Scripts\activate

# Criar e ativar o ambiente virtual (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Instalar pacotes
pip install -r requirements.txt
```

### 3. Disponibilizar o dataset
Para execução local, coloque o arquivo CSV neste caminho:

```text
data/processed/2017_Yellow_Taxi_Trip_Data.csv
```

O pipeline também preserva o caminho original do Kaggle usado no notebook:

```text
/kaggle/input/new-york-city-taxi-trips-2017/2017_Yellow_Taxi_Trip_Data.csv
```

### 4. Rodar o pipeline completo
Execute o orquestrador `main.py`. O pipeline irá carregar os dados, criar features, gerar visualizações de EDA, tratar outliers, preparar treino/teste, treinar modelos comparativos com `GridSearchCV`, salvar artefatos em `models/` e exportar gráficos para `outputs/figures/`.

```bash
python main.py
```

---

## 🛠️ Tecnologias Utilizadas / Tech Stack
*   **Language:** Python 3.x
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Statistics:** SciPy, Statsmodels
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Hyperparameter Tuning:** GridSearchCV, KFold
*   **Model Persistence:** Pickle

---

## 📊 Métricas de Avaliação / Evaluation Metrics
*   **R²:** coeficiente de determinação para medir a capacidade explicativa do modelo.
*   **RMSE:** erro quadrático médio na escala original da tarifa.
*   **MAE:** erro absoluto médio, útil para interpretar o erro médio em dólares.

---

## 📌 Observações / Notes
*   O dataset não é versionado diretamente neste repositório por ser um arquivo externo do Kaggle.
*   Os diretórios `models/` e `outputs/figures/` são utilizados para armazenar artefatos gerados durante a execução.
