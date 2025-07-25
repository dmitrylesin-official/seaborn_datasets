# 🐧 Penguin Species Classification
📥 The dataset is loaded directly from the Seaborn library using the built-in penguins dataset.

This project demonstrates a clean and efficient machine learning pipeline to classify penguin species based on physical measurements and categorical features like island and sex. We use a Logistic Regression model combined with proper preprocessing, including imputation and encoding.

---

## 📂 Dataset
The penguins dataset contains the following features:

**• Numerical:** bill length, bill depth, flipper length, body mass

**• Categorical:** island, sex

**• Target variable:** species (Adelie, Chinstrap, Gentoo)

The dataset is loaded via Seaborn:
```python
import seaborn as sns
df = sns.load_dataset('penguins')
```

---

## 🧹 Data Preprocessing
• Features (X) and target (y) are separated

• Train/test split with 80/20 ratio and stratification on target variable

• Missing values handled with median imputation (numerical) and most frequent imputation (categorical)

• Categorical features one-hot encoded with unseen categories ignored

• Numerical features scaled using StandardScaler

---

🧠 Model Pipeline
We use a Scikit-learn Pipeline combining preprocessing and a Logistic Regression classifier:
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=10000))
])
```
----

## 📊 Evaluation
After training, model performance is evaluated on the test set with accuracy:
```python
from sklearn.metrics import accuracy_score

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
```
Additionally, 5-fold cross-validation is performed to assess stability:
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model_pipeline, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

## 🚀 Results Summary
✅ Test Accuracy: 1.0

📈 Cross-validation accuracy confirms model stability across splits

The Logistic Regression model with careful preprocessing provides a solid baseline for species classification.

---

## 🛠 Technologies Used
• Python

• Pandas

• Seaborn

• Scikit-learn

---

## 📬 Author
Telegram: @dmitrylesin

Email: dmitrylesin_official@gmail.com

© 2025 Dmitry Lesin. All rights reserved.
