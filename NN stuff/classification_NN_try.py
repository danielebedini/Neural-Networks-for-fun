# Importa le librerie necessarie
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from data_try import X,y
#from makemoons_try import X,y
#from moref_try import X,y
#from specialplots_try import X,y
from regression_try2 import X,y

# Crea il modello della rete neurale
model = keras.Sequential()

# Aggiungi uno strato di input con 10 neuroni e specifica la forma dell'input (ad esempio, 5 feature)
model.add(layers.Input(shape=(2,)))

# Per i pesi
# - tutti zero: kernel_initializer = 'zeros'
# - valori piccoli: weight_init = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)

# Inizializzazione dei pesi con valori casuali molto piccoli (es. deviazione standard 0.001)
weight_init = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)

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
model.add(layers.Dense(2, activation='tanh', kernel_initializer = weight_init))
# model.add(layers.Dense(10, activation='relu', kernel_initializer='zeros'))

# Aggiungi uno strato di output con 1 neurone e funzione di attivazione sigmoide per la classificazione binaria
model.add(layers.Dense(1, activation='tanh', kernel_initializer = weight_init))


# Compila il modello specificando la funzione di perdita, l'ottimizzatore e le metriche
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Stampa un riepilogo del modello
model.summary()

# Comando per fittare i dati di input X con gli output y di data_try.py
# model.fit(X, y, epochs=30, batch_size = 4) 

# Comando per fittare i dati di input X con gli output y dei vari file
model.fit(X,y, epochs=50)

# Utilizza questo comando per fare previsioni utilizzando il modello addestrato. 
# Passa i dati di input (X) e otterrai le previsioni corrispondenti
model.predict(X)

# Questo comando valuta le prestazioni del modello utilizzando dati di test. 
# Devi specificare i dati di input (X) e i target di test (y). 
# Restituisce i valori della funzione di perdita e delle metriche specificate durante la compilazione.
model.evaluate(X,y) 