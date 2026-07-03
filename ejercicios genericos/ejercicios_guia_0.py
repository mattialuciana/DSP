# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:17:44 2026

@author: dell_
"""

# Ejercicio 4
# Se requiere imprimir en consola cada medición de 20 muestras que está dentro de la matriz mediciones, 
# las cuales corresponden a las filas de la matriz.


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio

fs = 44100
t = np.linspace(0, 1, fs)
f = 440
A = 0.5

audio = A * np.sin(2*np.pi*f*t)

# Graficar
plt.plot(t[:1000], audio[:1000])
plt.show()

# Guardar
sf.write("audio.wav", audio, fs)

# Reproducir
Audio(audio, rate=fs)