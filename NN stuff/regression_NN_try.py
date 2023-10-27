# Importa le librerie necessarie
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

# from regression_try import X,y
from morecomplexreg2_try import X,y,X_test,y_test

# Crea il modello della rete neurale
model = keras.Sequential()

# Aggiungi uno strato di input con 10 neuroni e specifica la forma dell'input (ad esempio, 5 feature)
model.add(layers.Input(shape=(1,)))

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
model.add(layers.Dense(15, activation='selu', kernel_initializer=RandomUniform(seed=42)))

# Aggiungi modello di output 
model.add(layers.Dense(1, activation='selu'))

# Compila il modello specificando la funzione di perdita, l'ottimizzatore e le metriche

custom_optimizer = Adam(learning_rate=0.001)

model.compile(loss='mean_squared_error', optimizer=custom_optimizer, metrics=['mae', 'mse'])

# Comandi possibili nel compile:
''' 
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy', 'mae', 'mse'],
    loss_weights=None,
    weighted_metrics=None,
    sample_weight_mode=None,
    target_tensors=None,
    distribute=None,
    steps_per_execution=1,
    experimental_steps_per_execution=1,
    learning_rate=0.001
)
'''

# Stampa un riepilogo del modello
model.summary()

# Comando per fittare i dati di input X con gli output y dei vari file
model.fit(X,y, epochs=50, verbose=0)

# Utilizza questo comando per fare previsioni utilizzando il modello addestrato. 
# Passa i dati di input (X) e otterrai le previsioni corrispondenti

model.predict(X_test)

# Questo comando valuta le prestazioni del modello utilizzando dati di test. 
# Devi specificare i dati di input (X) e i target di test (y). 
# Restituisce i valori della funzione di perdita e delle metriche specificate durante la compilazione.
model.evaluate(X_test,y_test) 