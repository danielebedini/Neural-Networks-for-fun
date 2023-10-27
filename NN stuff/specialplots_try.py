import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification

# Genera un set di dati non linearmente separabile con 4 feature
X, y = make_classification(n_samples=30, n_features=4, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
X_test,y_test = make_classification(n_samples=10, n_features=4, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Crea una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotta tutte e 4 le feature
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], label="Classe 0", marker='o')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], label="Classe 1", marker='x')

# Etichette degli assi
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

plt.legend()
plt.show()