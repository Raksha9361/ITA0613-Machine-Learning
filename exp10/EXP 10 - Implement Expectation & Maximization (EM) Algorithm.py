
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X = np.array([
    [1, 2], [1.5, 1.8], [2, 2.5],
    [8, 8], [8.5, 9], [9, 8.5],
    [4, 5], [4.5, 5.5], [5, 4.5]
])

gmm = GaussianMixture(n_components=3, random_state=42)


gmm.fit(X)


labels = gmm.predict(X)


print("Cluster Labels:")
print(labels)

print("\nCluster Means:")
print(gmm.means_)

print("\nCluster Covariances:")
print(gmm.covariances_)

plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0], X[:, 1],
    c=labels, cmap='viridis', s=100
)

plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    color='red', marker='X', s=200,
    label='Cluster Centers'
)

plt.title("Expectation Maximization Algorithm")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.show()
