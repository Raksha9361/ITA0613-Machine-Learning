
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Year': [2018, 2017, 2019, 2016, 2020,
             2015, 2021, 2018, 2019, 2017],

    'Kilometers_Driven': [30000, 40000, 20000, 50000,
                          15000, 60000, 10000, 35000,
                          25000, 45000],

    'Fuel_Type': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],

    'Transmission': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],

    'Price': [600000, 450000, 750000, 350000, 900000,
              300000, 1100000, 650000, 700000, 400000]
}


df = pd.DataFrame(data)


X = df[['Year', 'Kilometers_Driven',
        'Fuel_Type', 'Transmission']]

y = df['Price']


model = LinearRegression()


model.fit(X, y)


print("Enter Car Details")

year = int(input("Enter manufacturing year: "))
km = int(input("Enter kilometers driven: "))

print("Fuel Type: 1-Petrol, 0-Diesel")
fuel = int(input("Enter fuel type: "))

print("Transmission: 1-Automatic, 0-Manual")
transmission = int(input("Enter transmission: "))

new_car = pd.DataFrame(
    [[year, km, fuel, transmission]],
    columns=['Year', 'Kilometers_Driven',
             'Fuel_Type', 'Transmission']
)

predicted_price = model.predict(new_car)

print("\nPredicted Car Price: Rs.",
      round(predicted_price[0], 2))
