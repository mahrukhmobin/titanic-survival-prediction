# 🚢 Titanic Survival Prediction – From EDA to Model Building

An end-to-end machine learning project on the Titanic dataset — covering data exploration, preprocessing pipelines, model training, hyperparameter tuning, and feature importance analysis.

---

## 📌 Project Overview

This project walks through the complete ML workflow:

**EDA → Preprocessing → Pipelines → Model Building → Evaluation → Hyperparameter Tuning**

The goal is to predict whether a passenger survived the Titanic disaster based on features like age, sex, fare, and passenger class.

---

## 📊 Key Findings from EDA

- Females had significantly higher survival rates than males
- 1st class passengers survived more than 2nd and 3rd class
- Children had better survival chances
- **Most influential features:** `Sex`, `Fare`, `Age`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML pipelines & models |

---

## 🤖 Models Trained

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80% |
| Decision Tree | ~80% |
| Random Forest (default) | ~82% |
| **Random Forest (tuned)** | **~82%** ✅ |

> Random Forest with GridSearchCV tuning achieved the best performance.

---

## ⚙️ ML Pipeline

```
Raw Data
   ↓
Drop irrelevant columns (PassengerId, Name, Ticket, Cabin)
   ↓
ColumnTransformer
   ├── Numeric: Median Imputation → Standard Scaling
   └── Categorical: Mode Imputation → One-Hot Encoding
   ↓
Model Training (LR / DT / RF)
   ↓
Cross Validation (5-Fold Stratified)
   ↓
GridSearchCV Tuning
   ↓
Final Evaluation
```

---

## 📈 Results

**Feature Importance (Tuned Random Forest):**
- `Sex_male` — highest importance
- `Sex_female`, `Fare`, `Age` — top predictors
- `Pclass`, `SibSp`, `Embarked` — moderate impact

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── Titanic.py        # Main script (EDA + pipeline + models)
├── train.csv         # Dataset
└── README.md         # Project documentation
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/mahrukhmobin/titanic-survival-prediction.git

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run the script
python Titanic.py
```

---

## 📚 Learnings

- Building end-to-end ML pipelines with `sklearn.pipeline`
- Using `ColumnTransformer` for mixed data types
- Model comparison and evaluation with cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Interpreting feature importance from tree-based models

---

*Built by [Mahrukh Mobin](https://github.com/mahrukhmobin) — Computer Engineering Student @ UET Lahore*
