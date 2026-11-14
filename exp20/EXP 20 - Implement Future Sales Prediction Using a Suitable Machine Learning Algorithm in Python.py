import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [10000, 12000, 15000, 18000, 20000,
              22000, 25000, 28000, 30000, 33000]
}


df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']


model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({'Month': [11, 12, 13, 14, 15]})
predictions = model.predict(future_months)


print("Historical Sales Data:")
print(df)


print("\nFuture Sales Prediction:")
for month, sales in zip(future_months['Month'], predictions):
    print("Month", month, "Predicted Sales: Rs.", round(sales, 2))


plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(future_months, predictions,
            color='green', label='Predicted Sales')

plt.xlabel("Month")
plt.ylabel("Sales Amount (Rs.)")
plt.title("Future Sales Prediction Using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
