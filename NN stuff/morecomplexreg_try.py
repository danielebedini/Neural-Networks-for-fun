import numpy as np
import matplotlib.pyplot as plt

# Genera dati di esempio per la regressione non lineare
np.random.seed(0)
X = 5 * np.random.rand(100, 2)  # Due feature
y = 2 * X[:, 0] + 3 * X[:, 1]**2 + 1 + np.random.randn(100)  # Variabile dipendente (target)

# Visualizza i dati
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, marker='o')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Variabile Dipendente")
plt.title("Set di Dati per la Regressione Non Lineare")
plt.show()

# Ora hai un set di dati di esempio per la regressione con due feature e una relazione non lineare tra le feature e la variabile dipendente.
