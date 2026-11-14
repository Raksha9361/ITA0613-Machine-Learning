

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


data = {
    'RAM': [2, 3, 4, 6, 8, 12, 4, 6, 8, 12,
            2, 3, 6, 8, 12],

    'Battery': [2000, 2500, 3000, 3500, 4000,
                5000, 3200, 3800, 4500, 5000,
                1800, 2200, 3600, 4200, 5200],

    'Memory': [16, 32, 32, 64, 128, 256, 64,
               128, 128, 256, 16, 32, 64, 128, 256],

    'Camera': [5, 8, 12, 16, 20, 48, 13, 16,
               32, 48, 5, 8, 16, 32, 64],

    'Screen_Size': [5.0, 5.2, 5.5, 5.8, 6.0,
                    6.5, 5.6, 6.0, 6.2, 6.7,
                    4.8, 5.1, 5.7, 6.1, 6.8],

    'Price_Range': [0, 0, 1, 1, 2, 3, 1, 2,
                    2, 3, 0, 0, 1, 2, 3]
}

df = pd.DataFrame(data)


X = df[['RAM', 'Battery', 'Memory',
        'Camera', 'Screen_Size']]

y = df['Price_Range']


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X, y)


print("Enter Mobile Specifications")

ram = int(input("Enter RAM (GB): "))
battery = int(input("Enter Battery Power (mAh): "))
memory = int(input("Enter Internal Memory (GB): "))
camera = int(input("Enter Camera (MP): "))
screen = float(input("Enter Screen Size (inches): "))


new_mobile = pd.DataFrame(
    [[ram, battery, memory, camera, screen]],
    columns=['RAM', 'Battery', 'Memory',
             'Camera', 'Screen_Size']
)

prediction = model.predict(new_mobile)

price_categories = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("\nPredicted Mobile Price Range:",
      price_categories[prediction[0]])
