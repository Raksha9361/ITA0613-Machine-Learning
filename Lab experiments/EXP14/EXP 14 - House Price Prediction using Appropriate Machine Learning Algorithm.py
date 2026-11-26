
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1500, 1800, 2000, 2500,
             1200, 2200, 3000, 1600, 2800],

    'Bedrooms': [2, 3, 3, 4, 4, 2, 4, 5, 3, 5],

    'Bathrooms': [1, 2, 2, 3, 3, 1, 3, 4, 2, 4],

    'Age': [10, 5, 8, 4, 2, 15, 3, 1, 6, 2],

    'Price': [3000000, 4500000, 5000000, 6500000,
              8000000, 3200000, 7000000, 10000000,
              4800000, 9500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']


model = LinearRegression()

model.fit(X, y)


print("Enter House Details")

area = int(input("Enter house area (sq.ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
age = int(input("Enter house age: "))

new_house = pd.DataFrame(
    [[area, bedrooms, bathrooms, age]],
    columns=['Area', 'Bedrooms', 'Bathrooms', 'Age']
)

predicted_price = model.predict(new_house)


print("\nPredicted House Price: Rs.",
      round(predicted_price[0], 2))
