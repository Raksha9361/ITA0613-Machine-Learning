
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    'Income': [50000, 30000, 70000, 25000, 60000,
               40000, 80000, 20000, 55000, 35000],

    'Loan': [10000, 20000, 15000, 25000, 12000,
             18000, 10000, 30000, 14000, 22000],

    'Debt': [5000, 15000, 3000, 20000, 6000,
             12000, 2000, 25000, 7000, 18000],

    'Credit_Score': ['Good', 'Poor', 'Good', 'Poor',
                     'Good', 'Standard', 'Good', 'Poor',
                     'Standard', 'Poor']
}

df = pd.DataFrame(data)


X = df[['Income', 'Loan', 'Debt']]
y = df['Credit_Score']

le = LabelEncoder()
y = le.fit_transform(y)

model = DecisionTreeClassifier(random_state=42)


model.fit(X, y)


new_customer = [[65000, 12000, 4000]]

prediction = model.predict(new_customer)


result = le.inverse_transform(prediction)

print("Customer Financial Details:")
print("Income:", new_customer[0][0])
print("Loan:", new_customer[0][1])
print("Debt:", new_customer[0][2])

print("\nPredicted Credit Score:", result[0])
