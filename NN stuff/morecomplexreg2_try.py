import numpy as np
import matplotlib.pyplot as plt

# Genera dati di esempio per la regressione non lineare con 2 feature
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Dati di test per la validation
X_test = np.sort(10 * np.random.rand(80, 1), axis=0)
y_test = np.sin(X_test).ravel() + np.random.normal(0, 0.1, X_test.shape[0])

# Aggiungi una seconda feature, elevando al quadrato la feature originale
X2 = X**2

# Concatena le due feature per ottenere un set di dati con 2 feature
X_combined = np.concatenate((X, X2), axis=1)

# Visualizza i dati
plt.scatter(X_combined[:, 0], y, color='darkorange', label='Dati reali')
plt.xlabel("Feature 1")
plt.ylabel("Variabile Dipendente")
plt.title("Set di Dati per la Regressione Non Lineare con 2 Feature")
plt.legend()
plt.show()

# Ora hai un set di dati con 2 feature: X_combined[:, 0] e X_combined[:, 1].
