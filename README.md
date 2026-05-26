# 🏦 Loan Amount Prediction using Linear Regression

## 📌 Problem Statement
In the BFSI (Banking, Financial Services & Insurance) sector, determining the right loan amount for a customer is a critical decision. Approving too little means losing business. Approving too much increases default risk.

This project builds a regression model to predict the loan amount a customer qualifies for based on their financial and demographic profile — the same kind of model used by banks, NBFCs, and fintech companies for credit decisioning.

---

## 📂 Dataset
**Source:** Credit Risk Dataset — Kaggle  
**Size:** 32,581 rows × 12 columns  

| Column | Description |
|---|---|
| person_age | Age of the applicant |
| person_income | Annual income |
| person_home_ownership | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | Employment length in years |
| loan_intent | Purpose of loan (MEDICAL, EDUCATION, etc.) |
| loan_grade | Risk grade assigned by lender (A to G) |
| loan_amnt | **Target — Loan amount requested** |
| loan_int_rate | Interest rate on the loan |
| loan_status | 0 = Non-default, 1 = Default |
| loan_percent_income | Loan amount as % of income |
| cb_person_default_on_file | Historical default on record (Y/N) |
| cb_person_cred_hist_length | Length of credit history in years |

---

## ⚙️ Project Workflow
> Raw Data → Cleaning → EDA → Feature Engineering → Scaling → Modelling → Evaluation

1. **Data Cleaning** — removed impossible ages (>100) and employment lengths (>60), filled missing values with median
2. **EDA** — correlation heatmap, distribution plots, boxplots by category
3. **Feature Engineering** — log transforms on skewed features, interaction feature, one-hot encoding, dropped multicollinear column
4. **Modelling** — trained Linear Regression, Ridge, and Lasso
5. **Evaluation** — R², Adjusted R², RMSE, residual analysis, coefficient interpretation

---

## 📊 Results

| Model | R² | Adjusted R² | RMSE |
|---|---|---|---|
| Linear Regression | 0.8183 | 0.8176 | ₹2,704 |
| Ridge (alpha=1.0) | 0.8183 | 0.8176 | ₹2,704 |
| Lasso (alpha=0.1) | 0.8183 | 0.8176 | ₹2,704 |

All three models performed identically — confirming no overfitting in the baseline model. Regularisation had nothing to correct.

---

## 🔍 Key Business Insights

- **loan_percent_income** is the strongest predictor (coefficient: +5450) — banks size loans primarily based on the ratio of loan to income
- **log_income** is the second strongest (coefficient: +4814) — higher earners consistently qualify for larger loans
- **Riskier loan grades (D, E, F)** are associated with larger loan amounts — these borrowers take more and pay higher interest rates
- **Loan intent** has minimal impact on loan amount — financial profile matters far more than the stated purpose

---

## ⚠️ Limitations

- **Heteroscedasticity** detected in residuals — model errors grow for higher loan amounts, violating a core linear regression assumption
- Loan amounts cluster at round numbers (₹5,000 / ₹10,000 / ₹35,000) creating multimodal patterns that linear regression cannot fully capture
- A tree-based model (Random Forest / XGBoost) would likely outperform on this dataset

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/loan-regression-project.git
cd loan-regression-project

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Download credit_risk_dataset.csv from Kaggle and place in data/ folder

# 5. Open notebook
# Open notebooks/loan_regression.ipynb in VS Code or Jupyter
```

---

## 📁 Project Structure
loan_regression_project/
│
├── data/
│   └── credit_risk_dataset.csv
├── notebooks/
│   └── loan_regression.ipynb
├── scripts/
│   └── loan_regression_model.pkl
│   └── scaler.pkl
├── requirements.txt
└── README.md

---

## 🛠️ Libraries Used

- pandas, numpy — data manipulation
- matplotlib, seaborn — visualisation
- scikit-learn — modelling and evaluation

---

## 🔮 Next Steps

- [ ] Try Random Forest and XGBoost to handle heteroscedasticity
- [ ] Hyperparameter tune Ridge/Lasso alpha using cross-validation
- [ ] Use the same dataset for classification (predicting loan_status) using Logistic Regression
