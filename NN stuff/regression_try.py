import numpy as np
import matplotlib.pyplot as plt

# Genera dati di esempio per la regressione lineare
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Variabile indipendente tra 0 e 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Variabile dipendente (target)

# Visualizza i dati
plt.scatter(X, y)
plt.xlabel("Variabile Indipendente")
plt.ylabel("Variabile Dipendente")
plt.title("Set di Dati per la Regressione Lineare")
plt.show()

# Ora hai un set di dati di esempio per la regressione lineare con una variabile indipendente e una variabile dipendente.
