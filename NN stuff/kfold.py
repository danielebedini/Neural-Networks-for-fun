import numpy as np

# Creazione del dataset
# ...

# Numero di fold per la cross-validation
k = 5

# Calcola le dimensioni dei fold
fold_size = len(data) // k
remainder = len(data) % k

# Suddividi il dataset in fold
fold_indices = []
start_index = 0
for i in range(k):
    end_index = start_index + fold_size + (1 if i < remainder else 0)
    fold_indices.append((start_index, end_index))
    start_index = end_index

# Esegui la cross-validation
for i in range(k):
    val_indices = fold_indices[i]
    val_set = data[val_indices[0]:val_indices[1]]
    
    train_indices = [idx for j, idx in enumerate(fold_indices) if j != i]
    train_set = np.concatenate([data[idx[0]:idx[1]] for idx in train_indices])

    # Addestramento del modello su train_set e valutazione su val_set
    # ...
