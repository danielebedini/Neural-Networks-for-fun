import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Genera dati di esempio
data = np.random.rand(100, 64)  # Esempio di 100 campioni con 64 feature ciascuno

# Costruzione del modello
input_dim = 64
encoding_dim = 32

# Definizione dell'autoencoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)

# Compilazione del modello
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Lista per memorizzare i valori dell'errore durante l'addestramento
loss_history = []

# Addestramento dell'autoencoder
for epoch in range(50):
    history = autoencoder.fit(data, data, epochs=1, batch_size=32, verbose=0)
    loss = history.history['loss'][0]
    loss_history.append(loss)
    print(f'Epoch {epoch + 1}/{50} - Loss: {loss}')

# Ora il modello è addestrato e puoi tracciare l'andamento dell'errore
plt.plot(loss_history)
plt.title('Andamento dell\'errore durante l\'addestramento')
plt.xlabel('Epoca')
plt.ylabel('Errore')
plt.show()
