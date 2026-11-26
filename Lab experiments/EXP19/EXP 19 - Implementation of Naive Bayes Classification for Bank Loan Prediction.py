
# Bank Loan Prediction using Naive Bayes

import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Step 1: Create sample bank loan dataset
data = {
    'Age': [25, 35, 45, 30, 50, 28, 40, 55, 32, 48,
            26, 38, 52, 29, 42],

    'Income': [25000, 50000, 70000, 30000, 80000,
               20000, 60000, 90000, 35000, 75000,
               22000, 55000, 85000, 28000, 65000],

    'Loan_Amount': [100000, 200000, 300000, 150000,
                    250000, 120000, 180000, 350000,
                    140000, 280000, 110000, 190000,
                    320000, 130000, 220000],

    'Credit_Score': [650, 750, 800, 620, 780, 580,
                     720, 810, 640, 790, 590, 730,
                     820, 610, 760],

    'Employment': [1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
                   0, 1, 1, 0, 1],

    'Loan_Status': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                    0, 1, 1, 0, 1]
}

# Step 2: Convert dataset into DataFrame
df = pd.DataFrame(data)

# Step 3: Separate input and output
X = df[['Age', 'Income', 'Loan_Amount',
        'Credit_Score', 'Employment']]

y = df['Loan_Status']

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train the model
model.fit(X, y)

# Step 6: Get applicant details
print("Enter Applicant Details")

age = int(input("Enter Age: "))
income = int(input("Enter Monthly Income: "))
loan = int(input("Enter Loan Amount: "))
credit = int(input("Enter Credit Score: "))

print("Employment: 1-Employed, 0-Unemployed")
employment = int(input("Enter Employment Status: "))

# Step 7: Predict loan status
new_applicant = pd.DataFrame(
    [[age, income, loan, credit, employment]],
    columns=['Age', 'Income', 'Loan_Amount',
             'Credit_Score', 'Employment']
)

prediction = model.predict(new_applicant)

# Step 8: Display result
if prediction[0] == 1:
    print("\nLoan Status: Loan Approved")
else:
    print("\nLoan Status: Loan Rejected")
