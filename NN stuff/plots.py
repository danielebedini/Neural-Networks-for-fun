import numpy as np
import matplotlib.pyplot as plt


'''
# Tangente iperbolica

# Genera una serie di valori x
x = np.linspace(-5, 5, 100)

# Calcola i valori corrispondenti di tanh(x)
y = np.tanh(x)

# Crea il grafico
plt.plot(x, y, label='tanh(x)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('Grafico della Tangente Iperbolica (tanh)')
plt.grid(True)
plt.legend()
plt.show()
'''

# Sigmodal

# Genera una serie di valori x
x = np.linspace(-5, 5, 100)

# Calcola i valori corrispondenti di sigmoid(x)
# default: a = 1
a = 0.00000001
y = 1 / (1 + np.exp(-a*x))

# Crea il grafico
plt.plot(x, y, label='Sigmoid(x)')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.title('Grafico della Funzione Sigmoidale')
plt.grid(True)
plt.legend()
plt.show()

#'''