import numpy as np

import numpy as np

# Funzione di attivazione e la sua derivata
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Creazione del dataset di esempio
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inizializzazione dei pesi in modo casuale
input_size = 2
hidden_size = 4
output_size = 1

weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Tasso di apprendimento
learning_rate = 0.1

# Numero di epoche di addestramento
epochs = 10000

# Addestramento del modello
for epoch in range(epochs):
    # Fase di forward
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Calcolo dell'errore
    error = y - predicted_output

    # Fase di backward (backpropagation)
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Aggiornamento dei pesi
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

# Stampiamo l'output previsto dopo l'addestramento
print("Output previsto dopo l'addestramento:")
print(predicted_output)
