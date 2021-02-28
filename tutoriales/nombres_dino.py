import numpy as np
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend as keras

# Lectura de los datos
nombres = open("nombres_dinosaurios.txt", "r").read()
nombres = nombres.lower()

# Diccionario one hot para alfabeto
alfabeto = list(set(nombres))
