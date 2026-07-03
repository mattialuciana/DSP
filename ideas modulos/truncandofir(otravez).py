import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from modulos import filtro_fir, analisis_filtros, graficar_analisis

coeficientes_fir = np.load(r'Datos\archivos_tp1c2026\fir_hamming_1000Hz.npy')
punto_medio = np.argmax(coeficientes_fir)
fs2 = 44100
fir_og = filtro_fir(N=3000,path='Datos\\archivos_tp1c2026\\fir_hamming_1000Hz.npy')
fir_1000 = fir_og[punto_medio-500:punto_medio+500]
fir_500 = fir_og[punto_medio-250:punto_medio+250]
fir_200 = fir_og[punto_medio-100:punto_medio+100]
fir_50 = fir_og[punto_medio-25:punto_medio+25]
freq_fir_1000, mod_fir_1000, fase_fir_1000 = analisis_filtros(fir_1000, fs2) 
graficar_analisis(freq_fir_1000, mod_fir_1000, fase_fir_1000 , etiquetas='FIR 1000 coeficientes', titulo='Caracterización de Filtro FIR (1000 coeficientes)')
freq_fir_500, mod_fir_500, fase_fir_500 = analisis_filtros(fir_500, fs2)
graficar_analisis(freq_fir_500, mod_fir_500, fase_fir_500 , etiquetas='FIR 500 coeficientes', titulo='Caracterización de Filtro FIR (500 coeficientes)')
freq_fir_200, mod_fir_200, fase_fir_200 = analisis_filtros(fir_200, fs2)
graficar_analisis(freq_fir_200, mod_fir_200, fase_fir_200 , etiquetas='FIR 200 coeficientes', titulo='Caracterización de Filtro FIR (200 coeficientes)')
freq_fir_50, mod_fir_50, fase_fir_50 = analisis_filtros(fir_50, fs2)
graficar_analisis(freq_fir_50, mod_fir_50, fase_fir_50 , etiquetas='FIR 50 coeficientes', titulo='Caracterización de Filtro FIR (50 coeficientes)')



"""
n = np.arange(len(coeficientes_fir))

plt.figure(figsize=(10, 4))
plt.stem(n, coeficientes_fir)
plt.title("Coeficientes FIR")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()
print(len(coeficientes_fir))
print(np.argmax(coeficientes_fir))
"""