import numpy as np

# Genera dati di esempio
# Sintassi: np.array([feature_1, feature_2, ..., feature_n, classe])
data = np.array([
    [0.2, 0.3, 0.4, 0.1, 0],  # Esempio 1 con etichetta di classe 0
    [0.1, 0.5, 0.3, 0.2, 1],  # Esempio 2 con etichetta di classe 1
    [0.4, 0.2, 0.6, 0.3, 0],  # Esempio 3 con etichetta di classe 0
    [0.7, 0.8, 0.5, 0.4, 1],  # Esempio 4 con etichetta di classe 1
    # Aggiungi altri esempi con le relative etichette di classe
])

# Mischia i dati in modo casuale
np.random.shuffle(data)

# Dividi i dati in feature e etichette di classe
X = data[:, :-1]  # Feature
y = data[:, -1]   # Etichette di classe

# Ora hai i dati di addestramento (X) e le etichette di classe (y) pronte per l'addestramento.
