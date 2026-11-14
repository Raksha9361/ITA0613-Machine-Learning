


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Perceptron(max_iter=1000, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Perceptron Based Iris Classification")
print("Accuracy:", accuracy * 100, "%")


print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))


new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction:",
      iris.target_names[prediction[0]])


plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_pred, cmap='viridis'
)

plt.xlabel("Scaled Sepal Length")
plt.ylabel("Scaled Sepal Width")
plt.title("Perceptron Based Iris Classification")
plt.show()
