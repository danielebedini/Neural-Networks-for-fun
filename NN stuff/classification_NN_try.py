# Importa le librerie necessarie
import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

#from data_try import X,y, X_test,y_test
from makemoons_try import X,y,X_test,y_test
#from moref_try import X,y
# from specialplots_try import X,y, X_test,y_test

# Crea il modello della rete neurale
model = keras.Sequential()

# Aggiungi uno strato di input con 10 neuroni e specifica la forma dell'input (ad esempio, 5 feature)
model.add(layers.Input(shape=(2,)))

# Per i pesi
# - tutti zero: kernel_initializer = 'zeros'
# - valori piccoli: weight_init = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)

# Inizializzazione dei pesi con valori casuali molto piccoli (es. deviazione standard 0.001)
# weight_init = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)

# Per le activation functions: 
# - 'relu'
# - 'sigmoid'
# - 'tanh'
# - 'softmax'
# - 'linear'
# - 'elu'
# - 'leaky_relu'
# - 'softplus'
# - 'selu'
# - 'swish'
# - 'hard_sigmoid'
# - 'hard_swish'

# Aggiungi uno strato nascosto con n neuroni e funzione di attivazione ReLU
model.add(layers.Dense(8, activation='relu'))

# Aggiungi uno strato di output con 1 neurone e funzione di attivazione sigmoide per la classificazione binaria
model.add(layers.Dense(1, activation='sigmoid'))


# Compila il modello specificando la funzione di perdita, l'ottimizzatore e le metriche
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Stampa un riepilogo del modello
model.summary()

# Comando per fittare i dati di input X con gli output y dei vari file
# model.fit(X,y, epochs = 50, verbose = 0)

# Predict

'''
threshold = 0.5  
predicted_classes = (model.predict(X_test) > threshold).astype(int)
'''
'''
predictions = model.predict(X)
errors = y - predictions
'''
'''
predictions = model.predict(X)
mean_prediction = predictions.mean()
std_deviation = predictions.std()
max_value = predictions.max()
min_value = predictions.min()
'''
'''
predictions = model.predict(X)
plt.plot(X, predictions, label='Previste')
plt.plot(X, y, label='Reali')
plt.legend()
plt.show()
'''
'''
predictions = model.predict(X)
window = np.ones(10) / 10  # Finestra di media mobile
smoothed_predictions = np.convolve(predictions, window, 'same')
plt.plot(X, smoothed_predictions, label='Previste (Smoothed)')
plt.plot(X, y, label='Reali')
plt.legend()
plt.show()

'''
'''
predictions = model.predict(X_test)
sampled_indices = range(0, len(X_test), 10)  # Mostra un punto ogni 10
plt.plot(X_test[sampled_indices], predictions[sampled_indices], label='Previste')
plt.plot(X_test, y_test, label='Reali')
plt.legend()
plt.show()
'''

# Questo comando valuta le prestazioni del modello utilizzando dati di test. 
# Devi specificare i dati di input (X) e i target di test (y). 
# Restituisce i valori della funzione di perdita e delle metriche specificate durante la compilazione.

# model.evaluate(X_test,y_test) 

# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Funzione di Perdita: {loss:.4f}')
# print(f'Accuratezza: {accuracy * 100:.2f}%')

# Lista per memorizzare i valori dell'errore durante l'addestramento
loss_history = []

# Addestramento della rete
for epoch in range(50):
    #history = model.fit(data, data, epochs=1, batch_size=32, verbose=0)
    history = model.fit(X,y, epochs = 1, verbose = 0)
    loss = history.history['loss'][0]
    loss_history.append(loss)
    print(f'Epoch {epoch + 1}/{50} - Loss: {loss}')

# Ora il modello è addestrato e puoi tracciare l'andamento dell'errore
plt.plot(loss_history)
plt.title('Andamento dell\'errore durante l\'addestramento')
plt.xlabel('Epoca')
plt.ylabel('Errore')
plt.show()
