import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing dei dati
x_train = x_train / 255.0
x_test = x_test / 255.0

# Creazione del modello
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilazione del modello
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Valutazione del modello
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
