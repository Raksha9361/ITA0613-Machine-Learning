import numpy as np
from collections import Counter


X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 7],
    [7, 8],
    [8, 8]
])

y = ['A', 'A', 'A', 'B', 'B', 'B']


test = np.array([2, 2])


distances = np.sqrt(np.sum((X - test) ** 2, axis=1))


k = 3
indices = np.argsort(distances)[:k]


nearest = [y[i] for i in indices]
prediction = Counter(nearest).most_common(1)[0][0]


print("Test Data:", test)
print("Nearest Neighbours:", nearest)
print("Predicted Class:", prediction)
