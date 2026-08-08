# 📊 E-Commerce Customer Churn Prediction & Intelligence System

An end-to-end Machine Learning project that predicts customer churn using the **Olist Brazilian E-Commerce Dataset**. The project combines data preprocessing, feature engineering, exploratory data analysis, predictive modeling, explainable AI (SHAP), and deployment through an interactive Streamlit dashboard.

---

# 📌 Project Overview

Customer retention is one of the biggest challenges faced by e-commerce businesses. Acquiring a new customer is significantly more expensive than retaining an existing one. This project aims to identify customers who are likely to churn based on their historical purchasing behavior, allowing businesses to take proactive retention measures.

The project follows a complete machine learning pipeline:

- Data Collection
- Data Cleaning & Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Machine Learning Model Development
- Model Evaluation
- Explainable AI using SHAP
- Streamlit Web Application Deployment

---

# 🎯 Objectives

- Predict customer churn using historical purchase data.
- Identify the most important factors influencing churn.
- Compare multiple machine learning algorithms.
- Provide interpretable predictions using SHAP.
- Develop an interactive dashboard for business users.

---

# 📂 Dataset

**Dataset:** Olist Brazilian E-Commerce Public Dataset

The project uses multiple relational datasets including:

- Customers
- Orders
- Order Items
- Order Payments

These tables were merged to generate customer-level behavioral features for predictive modeling.

---

# 🏗️ Project Workflow

```text
Raw Data
     │
     ▼
Data Cleaning & Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
SHAP Explainability
     │
     ▼
Streamlit Deployment
```

---

# ⚙️ Feature Engineering

The following customer-level features were created:

| Feature | Description |
|----------|-------------|
| Recency | Days since last purchase |
| Total Orders | Number of completed orders |
| Total Spending | Total money spent |
| Average Order Value | Average spending per order |
| Average Items | Average number of items purchased |
| Total Freight | Total shipping cost |
| Average Installments | Average payment installments |
| Customer Tenure | Days between first and last purchase |
| Customer State | One-hot encoded customer location |

These features summarize customer purchasing behavior and serve as inputs to the machine learning models.

---

# 📈 Exploratory Data Analysis

The EDA focused on understanding customer purchasing behavior and identifying patterns related to churn.

The analysis includes:

- Churn Distribution
- Class Imbalance Analysis
- RFM (Recency, Frequency, Monetary) Analysis
- Distribution of Numerical Features
- Customer Tenure Analysis
- Correlation Heatmap
- Geographic Analysis by State
- Top Cities Analysis
- Purchase Frequency Analysis
- Pair Plot Analysis

Key insights were documented after each visualization to support business interpretation.

---

# 🤖 Machine Learning Models

Three supervised learning algorithms were trained and compared.

| Model | Purpose |
|--------|----------|
| Logistic Regression | Baseline model |
| Random Forest | Ensemble learning model |
| XGBoost | Gradient Boosting model |

The dataset was divided into training and testing sets using **Stratified Train-Test Split** to preserve the churn distribution.

---

# 📊 Model Evaluation

Models were evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

The best-performing model was selected based on overall predictive performance.

---

# 🔍 Explainable AI (SHAP)

To improve model transparency, SHAP (SHapley Additive exPlanations) was used.

The explainability analysis includes:

- SHAP Summary Plot
- SHAP Feature Importance
- SHAP Waterfall Plot

These visualizations explain how individual features influence churn predictions, making the model more interpretable for business stakeholders.

---

# 💻 Streamlit Dashboard

The trained XGBoost model was deployed using Streamlit.

The dashboard provides:

- Customer behavioral input form
- Real-time churn prediction
- Churn probability score
- Risk level indicator
- Business recommendations
- SHAP explainability
- Customer behavior radar chart

The application enables business users to assess churn risk without requiring machine learning knowledge.

---

# 📁 Repository Structure

```text
churn-prediction/
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── random_forest.pkl
│   └── logistic_regression.pkl
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_business_report.ipynb
│
├── outputs/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/churn-prediction.git
```

Move into the project folder:

```bash
cd churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# 💼 Business Impact

This solution helps businesses:

- Identify customers likely to churn.
- Prioritize retention campaigns.
- Improve customer lifetime value.
- Reduce customer acquisition costs.
- Support data-driven marketing decisions.

---

# 🔮 Future Improvements

Potential enhancements include:

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Integration with real-time customer databases.
- Automated customer segmentation.
- Cloud deployment using AWS or Azure.
- Email notification system for high-risk customers.
- Interactive business dashboards using Power BI or Tableau.

---

# 📷 Application Screenshots

Include screenshots of:

- Dashboard Home Page
- Prediction Result
- SHAP Explainability
- Business Recommendations
- Radar Chart

---

# 📜 License

This project is developed for educational and portfolio purposes.

---

# 👩‍💻 Author

**Shreyasi**

Machine Learning | Data Science | Python
