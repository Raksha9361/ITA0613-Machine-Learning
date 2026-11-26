import csv
import os

filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training_data.csv"
)

training_data = [
    ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"],
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(training_data)

print("CSV file created successfully.")




with open(filename, "r") as file:
    data = list(csv.reader(file))


attributes = data[0][:-1]

examples = data[1:]

n = len(attributes)


S = ["Ø"] * n

G = [["?"] * n]

def covers(h, example):

    for i in range(n):

        if h[i] != "?" and h[i] != example[i]:
            return False

    return True

def generalize_S(S, example):

    for i in range(n):

        if S[i] == "Ø":
            S[i] = example[i]

        elif S[i] != example[i]:
            S[i] = "?"

    return S

for example in examples:

    x = example[:-1]
    label = example[-1]


    if label == "Yes":

        S = generalize_S(S, x)

        G = [g for g in G if covers(g, x)]



    else:

        new_G = []

        for g in G:

            if covers(g, x):

                for i in range(n):

                    if g[i] == "?":

                        values = set(row[i] for row in examples)

                        for value in values:

                            if value != x[i] and value != S[i]:

                                new_h = g.copy()
                                new_h[i] = value

                                new_G.append(new_h)

            else:

                new_G.append(g)

        G = new_G


final_G = []

for g in G:

    if g not in final_G:
        final_G.append(g)

G = final_G

print("\n========================================")
print("TRAINING DATA")
print("========================================")

for row in data:
    print(row)

print("\n========================================")
print("ATTRIBUTES")
print("========================================")

print(attributes)

print("\n========================================")
print("SPECIFIC BOUNDARY (S)")
print("========================================")

print(S)

print("\n========================================")
print("GENERAL BOUNDARY (G)")
print("========================================")

for g in G:
    print(g)

print("\n========================================")
print("CANDIDATE-ELIMINATION COMPLETED")
print("========================================")

print("Specific Boundary:")
print(S)

print("\nGeneral Boundary:")
for g in G:
    print(g)

print("\nCSV file location:")
print(filename)
