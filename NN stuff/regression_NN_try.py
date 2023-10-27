# Importa le librerie necessarie
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from regression_try import X,y
from morecomplexreg_try import X,y

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
model.add(layers.Dense(15, activation='selu', kernel_initializer = weight_init))
model.add(layers.Dense(15, activation='selu', kernel_initializer = weight_init))

# Aggiungi modello di output 
model.add(layers.Dense(1, activation='selu'))

# Compila il modello specificando la funzione di perdita, l'ottimizzatore e le metriche
model.compile(loss='mean_squared_error', optimizer='adam')
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])

# Stampa un riepilogo del modello
model.summary()

# Comando per fittare i dati di input X con gli output y dei vari file
model.fit(X,y, epochs=50)