
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

y = np.array([6, 12, 22, 35, 52, 72, 98, 128, 162, 202])


linear_model = LinearRegression()
linear_model.fit(X, y)

linear_pred = linear_model.predict(X)


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_pred = poly_model.predict(X_poly)


linear_score = r2_score(y, linear_pred)
poly_score = r2_score(y, poly_pred)

print("Linear Regression R2 Score:", round(linear_score, 4))
print("Polynomial Regression R2 Score:", round(poly_score, 4))


plt.figure(figsize=(10, 5))

plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, linear_pred, color='red',
         label='Linear Regression')

plt.plot(X, poly_pred, color='green',
         label='Polynomial Regression')

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear vs Polynomial Regression")

plt.legend()
plt.grid()
plt.show()
