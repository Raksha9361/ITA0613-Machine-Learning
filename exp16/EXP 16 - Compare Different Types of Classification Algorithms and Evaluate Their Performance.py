
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []

for name, model in models.items():

   
    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average='weighted', zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average='weighted', zero_division=0
    )

    results.append([
        name, accuracy, precision, recall, f1
    ])

df = pd.DataFrame(
    results,
    columns=[
        "Algorithm", "Accuracy", "Precision",
        "Recall", "F1 Score"
    ]
)

print("\nClassification Algorithm Comparison")
print(df.round(3).to_string(index=False))

plt.figure(figsize=(10, 5))

plt.bar(df["Algorithm"], df["Accuracy"])

plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy")
plt.title("Performance Comparison of Classification Algorithms")

plt.xticks(rotation=30)
plt.ylim(0, 1.1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
