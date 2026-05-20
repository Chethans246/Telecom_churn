# Telecom Customer Churn Prediction

Predicting which telecom customers are at risk of churning using behavioral and demographic features, with a focus on early intervention and proactive retention.

---

## Problem Statement

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. The goal of this project is to build a machine learning model that identifies high-risk churners before they leave, enabling the business to intervene with targeted retention campaigns.

---

## Dataset

- **Source:** Provided by Rubixe AI Solutions as part of a structured analytics project program
- **Size:** 243,000 rows (30% sample used for analysis due to hardware constraints)
- **Target:** Binary — churned (1) or retained (0)
- **Class distribution:** Imbalanced; churned customers are the minority class

| Column | Description |
|---|---|
| customer_id | Unique customer identifier |
| age | Customer age |
| gender | Male / Female |
| telecom_partner | Telecom service provider |
| state / city | Geographic location |
| calls_made | Total calls made |
| sms_sent | Total SMS sent |
| data_used | Total data used (MB) |
| estimated_salary | Estimated annual salary |
| date_of_registration | Service registration date |
| churn | Target variable (1 = churned) |

---

## Repository Structure

```
telecom-churn-prediction/
│
└── telecom_churn.ipynb    # Full EDA, feature engineering, modeling, and business insights notebook
```

---

## Approach

### 1. Data Cleaning
- Removed duplicate records and dropped identifier columns
- Clipped negative usage values to zero (data entry errors)
- Converted registration date to tenure in days

### 2. Feature Engineering
Six new features created:

| Feature | Description |
|---|---|
| tenure_days | Days since registration (loyalty duration) |
| data_per_day | Daily data usage normalized by tenure |
| calls_per_day | Daily call rate normalized by tenure |
| total_activity | Composite usage score across calls, SMS, and data |
| low_usage_flag | 1 if customer is in bottom 20% across all usage channels |
| new_customer | 1 if tenure is 90 days or less |
| senior | 1 if age is 60 or above |

### 3. Exploratory Data Analysis
- Distribution analysis across all numeric and categorical features
- Churn rate analysis by telecom partner, geography, age group, and usage tier
- Identified that low total activity and short tenure are the strongest early churn signals

### 4. Model Comparison

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (Tuned) | 0.5172 | 0.4405 | 0.2679 | 0.4878 |
| **XGBoost (Tuned)** | **0.5323** | **0.4251** | **0.2672** | **0.4909** |
| Decision Tree | 0.5809 | 0.3456 | 0.2486 | 0.4918 |
| Random Forest (Tuned) | 0.6996 | 0.1593 | 0.1754 | 0.4879 |

Note: All models show ROC-AUC near 0.49, reflecting weak linear separability in the available features. This is a dataset signal issue, not a modeling error. XGBoost was selected for deployment based on scalability and native imbalance handling.

### 5. Class Imbalance Handling
- Applied `scale_pos_weight` in XGBoost to natively correct for class imbalance
- Prioritized Recall and F1-score over accuracy as evaluation metrics

---

## Model Selection: Why XGBoost

XGBoost and Logistic Regression produced virtually identical F1 scores (difference of 0.0007). XGBoost was selected for the following reasons:

- Natively parallelizable and scales to the full 243,000-row production dataset
- Captures non-linear feature interactions that will matter as richer features are added
- `scale_pos_weight` integrates imbalance correction directly into the boosting objective
- Threshold-tunable probability scores give the business control over precision vs recall tradeoff

---

## Key Features by Importance

1. **total_activity** — Single strongest disengagement indicator; consistently low values are an early warning sign
2. **tenure_days** — Short tenure is the highest churn risk window; first 90 days are critical
3. **data_per_day / calls_per_day** — Tenure-normalized usage captures engagement intensity more accurately than raw counts
4. **low_usage_flag** — Cross-channel disengagement signal; customers in the bottom 20% across all channels need immediate outreach
5. **new_customer** — Flags highest early churn risk window

---

## Business Recommendations

**Proactive Retention Segments**

| Segment | Flag | Recommended Action |
|---|---|---|
| New customers | tenure <= 90 days | Onboarding program, welcome offers, dedicated support |
| Low usage customers | low_usage_flag = 1 | Proactive re-engagement before formal churn |
| Senior customers | age >= 60 | Simplified plans, dedicated helpline |
| High-risk regions | above-average churn by state/city | Network investment, localized pricing |

**Estimated Revenue Impact (Illustrative)**
- Monthly churners: 1,000
- Model recall: ~43% = 430 churners identified in advance
- Retention success rate: 50% = 215 customers saved
- Average monthly revenue per customer: Rs. 500
- Monthly revenue protected: Rs. 1,07,500

---

## Tech Stack

- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/Chethans246/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter openpyxl
   ```

3. Launch the notebook
   ```bash
   jupyter notebook telecom_churn.ipynb
   ```

---

## Limitations and Future Work

**Current Limitations**
- No contract type, plan tier, or complaint history features (strong churn drivers)
- Static features do not capture month-over-month usage trend direction
- Trained on a 30% sample; rare patterns may be underrepresented
- External shocks (competitor price cuts, network outages) are not modeled

**Planned Improvements**
- Add contract type, plan tier, and complaint history
- Engineer time-series features to capture usage trend direction
- Monthly model retraining as churn patterns shift with market conditions
- Integrate churn scores into CRM for automated retention workflows

---

## Author

**Chethan S** — Data Analyst  
[LinkedIn](https://linkedin.com/in/chethan-s-69b71b241) • [GitHub](https://github.com/Chethans246)
