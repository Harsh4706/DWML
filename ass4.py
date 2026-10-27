
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 2. Create dataset
np.random.seed(10)
n = 300

data = pd.DataFrame({
    'Income': np.random.randint(20000, 150000, n),
    'Credit_Score': np.random.randint(550, 850, n),
    'Employment_Status': np.random.choice(
        ['Salaried', 'Self-Employed', 'Unemployed'], n
    ),
    'Loan_Amount': np.random.randint(50000, 1000000, n),
    'Loan_Term': np.random.choice([12, 24, 36, 48, 60], n),
    'Past_Default': np.random.choice(['Yes', 'No'], n)
})

# 3. Generate loan approval status
score = (
    (data['Credit_Score'] >= 700).astype(int) * 3 +
    (data['Income'] >= 60000).astype(int) * 2 +
    (data['Employment_Status'] != 'Unemployed').astype(int) +
    (data['Loan_Amount'] <= 500000).astype(int) * 2 +
    (data['Past_Default'] == 'No').astype(int) * 3 +
    (data['Loan_Term'] <= 48).astype(int)
)

data['Loan_Status'] = (score >= 7).astype(int)

# 4. Explore dataset
print("Dataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nLoan status counts:\n", data['Loan_Status'].value_counts())
print("\nFirst five rows:\n", data.head())

# 5. Separate features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# 6. Encode categorical columns
le = LabelEncoder()
X['Employment_Status'] = le.fit_transform(X['Employment_Status'])
X['Past_Default'] = le.fit_transform(X['Past_Default'])

# 7. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 9. Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# 10. Evaluate both models
models = {
    'Decision Tree': dt_pred,
    'Naive Bayes': nb_pred
}

results = []

for name, pred in models.items():
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("Classification Report:\n",
          classification_report(y_test, pred, zero_division=0))

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1 Score': f1_score(y_test, pred, zero_division=0)
    })

# 11. Compare models
results = pd.DataFrame(results)

print("\nModel Comparison:")
print(results.round(3))

# 12. Plot accuracy comparison
results.plot(
    x='Model', y='Accuracy',
    kind='bar', legend=False,
    title='Loan Approval Model Accuracy'
)

plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 13. Display final result
best_model = results.loc[results['F1 Score'].idxmax(), 'Model']
print("\nBest model based on F1 Score:", best_model)
