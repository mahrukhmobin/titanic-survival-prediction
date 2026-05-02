import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

df = pd.read_csv("TITAN_IC/Train.csv")
df.info()
print(df.head())
print(df.describe(include="all"))
print(df.columns.tolist())
print(df.dtypes)
print(df.shape)
print(df.isnull().sum().sort_values(ascending=False))
print(df.duplicated().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

sns.countplot(x="Survived", data=df)
plt.title("Survival Count (0 = No, 1 = Yes)", fontsize=14)
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Sex", fontsize=14)
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class", fontsize=14)
plt.show()

cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

y = df['Survived']
X = df.drop(columns=['Survived'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]
numeric_features = [c for c in numeric_features if c in X_train.columns]
categorical_features = [c for c in categorical_features if c in X_train.columns]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

clf_default = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
clf_default.fit(X_train, y_train)
y_pred = clf_default.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

clf_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))

clf_dt = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=6, random_state=42))
])
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

plt.figure(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Survived", "Survived"],
            yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout(); plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models_for_cv = {"LogisticRegression": clf_default, "RandomForest": clf_rf, "DecisionTree": clf_dt}
print("5-Fold CV Accuracy:")
for name, pipe in models_for_cv.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5, 10]
}
grid = GridSearchCV(clf_rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("Tuned RandomForest Accuracy:", accuracy_score(y_test, y_pred_best_rf))

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "RandomForest (default)", "RandomForest (tuned)"],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_best_rf)
    ]
})
print(results)

plt.figure(figsize=(8, 5))
sns.barplot(x="Accuracy", y="Model", data=results)
plt.title("Model Comparison")
plt.tight_layout()
plt.show()

importances = best_rf.named_steps["classifier"].feature_importances_
feature_names = (
    numeric_features +
    list(best_rf.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .named_steps['onehot']
         .get_feature_names_out(categorical_features))
)
feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance (RandomForest (Tuned))")
plt.show()

best_model = results.loc[results["Accuracy"].idxmax()]
print(f"Best model: {best_model['Model']} with accuracy {best_model['Accuracy']:.3f}")






#means first is EDA and all graphs/visualizations, then remove the extra columns, then select the target (X and y). After that, do the pipeline and ColumnTransformer part, and then use different models like Logistic Regression, Decision Tree, and Random Forest to find their accuracy. Then pick the one with the highest accuracy and plot the graph for our convenience.
#After that, do cross validation and GridSearchCV to adjust the parameters for the best accuracy, and then find the best model. Next, create a DataFrame for the accuracy of all models and plot the graph of model accuracy. Then check the feature importance (which column has how much impact) and plot that graph.
#At the end, find the most important column.
