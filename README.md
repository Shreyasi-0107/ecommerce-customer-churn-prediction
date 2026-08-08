# 📊 E-Commerce Customer Churn Prediction & Intelligence System

An end-to-end **Machine Learning, Explainable AI (XAI), and Business Intelligence** solution built using the **Olist Brazilian E-Commerce Dataset**.

The project predicts customer churn, explains the key behavioral factors behind each prediction using **SHAP**, and provides actionable customer-retention recommendations through an interactive **Streamlit dashboard**.

---

## 🚀 Live Demo

### 🌐 Deployed Streamlit Application

[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://ecommerce-customer-churn-prediction-s0125.streamlit.app/)

The deployed application provides:

- 🔮 Customer churn prediction
- 🚦 Churn risk classification
- 🧠 SHAP-based prediction explanations
- 📊 Customer behavioral analysis
- 📈 Model performance visualization
- 💡 Business retention recommendations

---

## 🎯 Project Overview

Customer churn is a major challenge for e-commerce businesses because acquiring a new customer is often more expensive than retaining an existing one.

This project develops a complete customer churn prediction pipeline that transforms raw transactional data into:

**Customer Behavioral Data → Machine Learning Prediction → Explainable AI → Business Action**

The system identifies customers who are likely to churn, explains **why** they are at risk, and provides recommendations for improving customer retention.

---

## 💼 Business Problem

The primary business objective is:

> **Identify customers who are likely to churn and determine the behavioral factors contributing to their churn risk so that targeted retention strategies can be applied.**

Instead of simply answering:

> **Will this customer churn?**

the system also answers:

> **Why is this customer likely to churn?**

and:

> **What can the business do about it?**

---

## 🎯 Project Objectives

The project aims to:

1. Integrate multiple e-commerce transaction tables.
2. Clean and prepare customer-level data.
3. Define customer churn using behavioral purchase patterns.
4. Engineer meaningful customer-level features.
5. Perform exploratory data analysis.
6. Compare multiple machine learning algorithms.
7. Tune an XGBoost classification model.
8. Evaluate model performance using out-of-time validation.
9. Explain predictions using SHAP.
10. Convert model predictions into business recommendations.
11. Build an interactive Streamlit application.
12. Deploy the application as a publicly accessible web application.

---

## 📂 Dataset

The project uses the **Olist Brazilian E-Commerce Dataset**, which contains approximately 100k e-commerce orders from Brazil.

The project integrates multiple relational tables including:

- Customers
- Orders
- Order Items
- Payments
- Order Reviews

The tables are connected using customer and order identifiers to construct a customer-level behavioral dataset.

---

## 🔗 Data Pipeline

```text
Raw Olist Tables
       │
       ▼
Data Cleaning
       │
       ▼
Table Integration
       │
       ▼
Order-Level Aggregation
       │
       ▼
Customer-Level Feature Engineering
       │
       ▼
Churn Label Creation
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Time-Based Train/Test Split
       │
       ▼
Model Training
       │
       ├── Logistic Regression
       ├── Random Forest
       └── XGBoost
       │
       ▼
Model Evaluation
       │
       ▼
SHAP Explainability
       │
       ▼
Business Recommendations
       │
       ▼
Streamlit Dashboard
       │
       ▼
Cloud Deployment
```

---

## ⚠️ Data Leakage Prevention

A major focus of the project was preventing **look-ahead bias and data leakage**.

A strict **90-day snapshot cutoff** was applied before the end of the dataset observation period.

This ensures that customer features are generated only from information that would have been available at the prediction point.

This is important because using future transactions or future customer behavior would artificially improve model performance and make the model unsuitable for real-world deployment.

---

## 🧮 Feature Engineering

Customer-level behavioral features were created from the transactional data.

### RFM Features

| Feature | Description |
|---|---|
| `recency` | Number of days since the customer's latest purchase |
| `frequency` | Number of completed orders |
| `monetary` | Total customer expenditure |

RFM analysis provides a strong representation of customer purchasing behavior.

### Customer Tenure

| Feature | Description |
|---|---|
| `tenure` | Elapsed time between the customer's first and latest purchase |

This helps distinguish newer customers from longer-term customers.

### Order Behavior

| Feature | Description |
|---|---|
| `avg_order_value` | Average spending per order |
| `avg_items` | Average number of items purchased per order |
| `total_freight` | Total shipping/freight expenditure |

### Logistics Features

| Feature | Description |
|---|---|
| `avg_delivery_time` | Average delivery duration |
| `avg_delivery_delay` | Average delivery delay |

These features capture the potential effect of fulfillment and delivery experience on churn.

### Customer Satisfaction

| Feature | Description |
|---|---|
| `avg_review_score` | Average customer review rating from 1 to 5 |

Lower review scores can indicate customer dissatisfaction and potential churn risk.

### Payment Behavior

| Feature | Description |
|---|---|
| `avg_installments` | Average customer payment installment behavior |

### Geography

Customer state information was transformed using **One-Hot Encoding**.

Example:

```text
customer_state_SP
customer_state_RJ
customer_state_MG
...
```

---

## 🔍 Exploratory Data Analysis

The project includes detailed exploratory analysis of customer behavior.

Key areas analyzed include:

- Customer churn distribution
- Recency distribution
- Purchase frequency
- Monetary value
- Average order value
- Customer tenure
- Delivery behavior
- Freight costs
- Review scores
- Geographic differences
- Feature correlations
- Pairwise relationships

Log transformations were used where appropriate to visualize highly skewed monetary and frequency distributions.

---

## 🤖 Machine Learning Models

Three classification models were evaluated.

### 1. Logistic Regression

Used as the baseline model.

Advantages:

- Simple
- Interpretable
- Fast
- Provides a useful baseline for comparison

### 2. Random Forest

A tree-based ensemble model capable of capturing nonlinear relationships and feature interactions.

### 3. XGBoost

The final model was based on **XGBoost**, a gradient boosting algorithm that performs well on structured/tabular datasets.

The XGBoost model was optimized using:

```text
RandomizedSearchCV
```

---

## ⏳ Time-Based Validation

Instead of relying only on a random train-test split, the project uses an **out-of-time validation strategy**.

This simulates a realistic deployment scenario:

```text
Past Customer Data
        ↓
     Training
        ↓
      Model
        ↓
Future Customer Data
        ↓
      Testing
```

This approach helps evaluate how well the model can generalize to future customer behavior.

---

## 📊 Model Performance

The evaluated models produced the following results:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.4% | 0.762 | 81.5% | 71.0% | 0.759 |
| Random Forest | 80.1% | 0.865 | 89.4% | 81.2% | 0.851 |
| **XGBoost** | **85.6%** | **0.910** | **85.7%** | **96.4%** | **90.7%** |

### 🏆 Final Model: XGBoost

Key performance:

- **Accuracy:** 85.6%
- **ROC-AUC:** 0.910
- **Precision:** 85.7%
- **Recall:** 96.4%
- **F1-Score:** 90.7%

The high recall is particularly useful for a churn-retention use case because missing a potentially churning customer can result in lost retention opportunities.

---

## 🧠 Explainable AI — SHAP

Machine learning predictions are not sufficient for a business decision-making system.

The project therefore uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

SHAP is used at both global and local levels.

### Global Explainability

Identifies which features are most important across the overall customer population.

### Local Explainability

Explains why a particular customer's prediction is high or low risk.

---

## 📈 SHAP Visualizations

### SHAP Bar Plot

Shows global feature importance.

```text
outputs/shap_bar.png.png
```

### SHAP Beeswarm Plot

Shows the distribution and direction of feature effects across customers.

```text
outputs/shap_beeswarm.png.png
```

### SHAP Waterfall Plot

Explains the contribution of individual features for a specific customer prediction.

```text
outputs/shap_waterfall.png.png
```

---

## 💡 Key Behavioral Drivers

The SHAP analysis identified important behavioral drivers of churn.

### 1. Customer Tenure

Shorter customer tenure is associated with higher churn risk, while longer relationships tend to indicate stronger retention.

### 2. Purchase Frequency

Customers with more repeat purchases generally demonstrate stronger engagement and lower churn risk.

### 3. Monetary Expenditure

Higher-value customers tend to demonstrate stronger purchasing engagement.

### 4. Freight Overhead

Higher shipping costs can negatively influence customer retention, particularly when shipping costs represent a significant portion of order value.

### 5. Review Score

Lower customer satisfaction scores can indicate dissatisfaction and increased churn risk.

---

## 🖥️ Streamlit Application

The project includes an interactive **Streamlit web application** for real-time customer churn prediction.

Users can enter customer behavioral information and receive a churn prediction through an interactive dashboard.

---

## ⚙️ Application Features

### 👤 Customer Behavioral Input

Users can provide information such as:

- Purchase frequency
- Customer tenure
- Monetary value
- Total freight
- Average items per order
- Delivery delay
- Average order value
- Payment installments
- Customer state
- Review rating

### 🔮 Churn Prediction

The application processes the customer information through the trained XGBoost model and produces a churn probability.

### 🚦 Risk Classification

Customers are categorized into risk levels such as:

```text
Low Risk
Medium Risk
High Risk
```

This allows business users to prioritize retention activities.

### 🧠 SHAP Customer Explanation

The application provides SHAP-based explanations showing which customer characteristics are contributing to the churn prediction.

This helps answer:

> **Why is this customer considered high risk?**

### 🕸️ Behavioral Radar Signature

The dashboard provides a radar-style representation of the customer's behavioral profile.

### 📊 Global Feature Importance

The application displays the major features influencing churn predictions across the model.

### 📈 Model Validation

The dashboard includes model evaluation visualizations such as:

- ROC curve
- Confusion matrix
- Feature importance

### 💼 Business Recommendations

The prediction system converts churn risk into actionable business recommendations.

#### High Churn Risk

Potential actions:

- Personalized retention offers
- Targeted discounts
- Customer re-engagement campaigns
- Shipping cost incentives
- Service recovery for dissatisfied customers

#### Medium Churn Risk

Potential actions:

- Engagement campaigns
- Personalized product recommendations
- Loyalty incentives
- Promotional communication

#### Low Churn Risk

Potential actions:

- Loyalty programs
- Cross-selling
- Upselling
- Referral campaigns

---

## 📁 Repository Structure

```text
ecommerce-customer-churn-prediction/
│
├── Data/
│   ├── Raw/
│   │   └── Olist raw dataset files
│   │
│   └── Processed/
│       └── customer_features.csv
│
├── models/
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── xgb_model.pkl
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_business_report.ipynb
│
├── outputs/
│   ├── shap_bar.png.png
│   ├── shap_beeswarm.png.png
│   ├── shap_waterfall.png.png
│   ├── xgb_confusion_matrix.png.png
│   └── xgb_roc_curve.png.png
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📓 Notebook Organization

### `01_data_preparation.ipynb`

Contains:

- Data loading
- Data cleaning
- Table integration
- Snapshot cutoff
- Order-level aggregation
- Customer-level feature engineering
- Churn label creation
- Processed dataset generation

### `02_eda.ipynb`

Contains:

- Churn distribution analysis
- Customer behavior analysis
- RFM analysis
- Feature distributions
- Correlation analysis
- Geographic analysis
- Pairwise analysis
- EDA findings

### `03_modeling.ipynb`

Contains:

- Feature selection
- Preprocessing
- Train/test splitting
- Time-based validation
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter tuning
- Model evaluation
- Model serialization

### `04_explainability.ipynb`

Contains:

- SHAP TreeExplainer
- Global feature importance
- SHAP beeswarm visualization
- SHAP waterfall visualization
- Individual prediction explanations

### `05_business_report.ipynb`

Contains:

- Business insights
- Churn drivers
- Retention recommendations
- Project conclusion
- Future work
- References

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning and preprocessing |
| XGBoost | Final churn prediction model |
| SHAP | Explainable AI |
| Matplotlib | Data visualization |
| Plotly | Interactive visualization |
| Streamlit | Web application |
| Joblib | Model serialization |
| Pillow | Image handling |
| Jupyter Notebook | Analysis and experimentation |
| Git & GitHub | Version control and project hosting |

---

## ⚙️ Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shreyasi-0107/ecommerce-customer-churn-prediction.git
```

Navigate into the project:

```bash
cd ecommerce-customer-churn-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
shap
plotly
pillow
```

---

## 🚀 Run the Application Locally

From the project root directory:

```bash
python -m streamlit run streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

The deployment is connected to the GitHub repository and uses:

```text
streamlit_app.py
```

as the application entry point.

Dependencies are specified in:

```text
requirements.txt
```

### 🌐 Live Application

👉 **[https://ecommerce-customer-churn-prediction-s0125.streamlit.app/](https://ecommerce-customer-churn-prediction-s0125.streamlit.app/)**

---

## 🔐 Model Artifacts

The trained models and preprocessing artifacts are stored in the `models/` directory.

Important artifacts include:

```text
xgb_model.pkl
scaler.pkl
feature_columns.pkl
```

These allow the deployed application to reproduce the same preprocessing and prediction pipeline used during model development.

---

## 📌 Key Project Outcomes

The project demonstrates a complete machine learning lifecycle:

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Analysis
       ↓
Model Development
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Time-Based Validation
       ↓
Explainable AI
       ↓
Business Recommendations
       ↓
Streamlit Application
       ↓
Cloud Deployment
```

The final system combines:

**Machine Learning + Explainable AI + Business Intelligence + Web Deployment**

---

## 🔮 Future Work

Potential improvements include:

- Automated model retraining pipelines
- Real-time customer data integration
- Customer-level database integration
- Automated email/SMS retention campaigns
- Cost-sensitive churn optimization
- Customer Lifetime Value integration
- More advanced time-series modeling
- Model monitoring and drift detection
- A/B testing of retention strategies
- Automated business reporting
- CRM system integration

---

## ⚠️ Limitations

The current system has several limitations:

1. The model is trained on historical Olist e-commerce data.
2. Churn behavior is inferred from historical purchasing patterns.
3. Business recommendations are rule-based rather than automatically optimized.
4. Customer behavior can change over time.
5. The deployed application is primarily a demonstration and decision-support system rather than a fully automated production CRM system.

---

## 📚 References

- Olist Brazilian E-Commerce Dataset
- Scikit-learn Documentation
- XGBoost Documentation
- SHAP Documentation
- Streamlit Documentation
- Pandas Documentation
- NumPy Documentation

---

## 👩‍💻 Author

### Shreyasi

**GitHub:**  
https://github.com/Shreyasi-0107

**Project Repository:**  
https://github.com/Shreyasi-0107/ecommerce-customer-churn-prediction

**Live Application:**  
https://ecommerce-customer-churn-prediction-s0125.streamlit.app/

---

## ⭐ Project Summary

> **E-Commerce Customer Churn Prediction & Intelligence System** is an end-to-end machine learning project that predicts customer churn using behavioral transaction data, explains predictions using SHAP, and transforms those predictions into actionable customer retention strategies through an interactive Streamlit dashboard.

### 🚀 Machine Learning + Explainable AI + Business Intelligence + Deployment
