import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Genera un set di dati non linearmente separabile con 4 feature
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Visualizza il set di dati
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Classe 0", marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Classe 1", marker='x')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Ora hai un set di dati con 4 feature e 1 output di classificazione che è non linearmente separabile.
