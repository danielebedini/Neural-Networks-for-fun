import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Genera un set di dati non linearmente separabili a forma di luna
X, y = make_moons(n_samples=200, noise=0.2, random_state=50)
X_test,y_test = make_moons(n_samples=50, noise=0.2, random_state=50)

# Visualizza il set di dati
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Classe 0", marker='o')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Classe 1", marker='x')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Ora hai un set di dati non linearmente separabili con 4 feature (usando solo le prime 2 feature) e 1 output di classificazione.
