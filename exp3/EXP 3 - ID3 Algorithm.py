import math
import pandas as pd




def entropy(data):

    target = data.iloc[:, -1]

    values = target.value_counts()

    total = len(target)

    ent = 0

    for count in values:

        probability = count / total

        ent -= probability * math.log2(probability)

    return ent




def information_gain(data, attribute):

    total_entropy = entropy(data)

    values = data[attribute].unique()

    weighted_entropy = 0

    for value in values:

        subset = data[data[attribute] == value]

        weight = len(subset) / len(data)

        weighted_entropy += weight * entropy(subset)

    gain = total_entropy - weighted_entropy

    return gain




def id3(data):

    target = data.columns[-1]

    if len(data[target].unique()) == 1:

        return data[target].iloc[0]

    if len(data.columns) == 1:

        return data[target].mode()[0]

    attributes = data.columns[:-1]

    gains = {}

    for attribute in attributes:

        gains[attribute] = information_gain(data, attribute)

    best_attribute = max(gains, key=gains.get)

    tree = {best_attribute: {}}

    for value in data[best_attribute].unique():

        subset = data[data[best_attribute] == value]

        subset = subset.drop(columns=[best_attribute])

        tree[best_attribute][value] = id3(subset)

    return tree



data = pd.DataFrame({

    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain", "Rain",
        "Rain", "Overcast", "Sunny", "Sunny", "Rain",
        "Sunny", "Overcast", "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild", "Cool",
        "Cool", "Cool", "Mild", "Cool", "Mild",
        "Mild", "Mild", "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High", "Normal",
        "Normal", "Normal", "High", "Normal", "Normal",
        "Normal", "High", "Normal", "High"
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Weak", "Weak",
        "Strong", "Strong", "Weak", "Strong"
    ],

    "PlayTennis": [
        "No", "No", "Yes", "Yes", "Yes",
        "No", "Yes", "No", "Yes", "Yes",
        "Yes", "Yes", "Yes", "No"
    ]
})




print("\n==============================")
print("PLAY TENNIS DATASET")
print("==============================")

print(data)



print("\n==============================")
print("INFORMATION GAIN")
print("==============================")

for attribute in data.columns[:-1]:

    gain = information_gain(data, attribute)

    print(attribute, "=", round(gain, 4))



tree = id3(data)

print("\n==============================")
print("DECISION TREE")
print("==============================")

print(tree)





def classify(tree, sample):

    if not isinstance(tree, dict):

        return tree

    attribute = next(iter(tree))

    value = sample[attribute]

    subtree = tree[attribute][value]

    return classify(subtree, sample)



new_sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}



result = classify(tree, new_sample)

print("\n==============================")
print("NEW SAMPLE")
print("==============================")

print(new_sample)

print("\nPredicted Class:", result)
