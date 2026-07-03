import numpy as np
import matplotlib.pyplot as plt

# Parámetro global de la ventana Kaiser según la consigna
BETA = 5.48

def calcular_y_graficar(L, N, ax):
    """
    Función auxiliar para calcular la señal, su FFT y graficarla.
    L: Longitud de la ventana / señal
    N: Número de puntos para la FFT (Zero-padding si N > L)
    ax: Eje de matplotlib donde se dibujará la gráfica
    """
    
    # 1. Definir el vector de tiempo discreto n (de 0 hasta L-1)
    n = np.arange(L)
    
    # 2. Generar la ventana de Kaiser de longitud L
    w = np.kaiser(L, BETA)
    
    # 3. Generar la señal x[n] según la ecuación dada
    # La señal original multiplicada por la ventana w[n]
    x = w * np.cos(2 * np.pi * n / 14) + 0.75 * w * np.cos(4 * np.pi * n / 15)
    
    # 4. Calcular la Transformada Discreta de Fourier (FFT)
    # Al especificar 'n=N', numpy automáticamente rellena con ceros (zero-padding) si N > L
    X = np.fft.fft(x, n=N)
    
    # 5. Calcular la magnitud del espectro
    mag_X = np.abs(X)
    
    # Como la señal es real, el espectro es simétrico. 
    # Solo nos interesa la primera mitad (de frecuencia 0 a pi).
    half_N = N // 2
    mag_X = mag_X[:half_N]
    
    # 6. Crear el eje de frecuencias (en radianes/muestra, de 0 a pi)
    omega = np.linspace(0, np.pi, half_N, endpoint=False)
    
    # 7. Graficar los resultados
    # Usamos marcadores para ver los puntos discretos reales calculados por la FFT
    ax.plot(omega, mag_X, '-o', markersize=4, linewidth=1.5)
    ax.set_title(f'L = {L}, N = {N}')
    ax.set_xlabel('Frecuencia $\omega$ (rad/muestra)')
    ax.set_ylabel('Magnitud |X|')
    ax.grid(True, linestyle='--', alpha=0.7)


# ==========================================
# CASO A: L y N aumentan simultáneamente
# ==========================================
fig_a, axes_a = plt.subplots(1, 3, figsize=(15, 4))
fig_a.suptitle('Caso A: Efecto de incrementar la longitud (L) sin zero-padding (N=L)', fontsize=14)

casos_a = [(32, 32), (64, 64), (128, 128)]
for i, (L, N) in enumerate(casos_a):
    calcular_y_graficar(L, N, axes_a[i])

plt.tight_layout()
plt.show()

# ==========================================
# CASO B: L fijo, N aumenta (Zero-Padding)
# ==========================================
fig_b, axes_b = plt.subplots(2, 2, figsize=(12, 8))
fig_b.suptitle('Caso B: Efecto del zero-padding (L constante, N aumenta)', fontsize=14)

casos_b = [(32, 32), (32, 64), (32, 128), (32, 1024)]
axes_b = axes_b.flatten() # Aplanar el array de ejes para iterar fácilmente

for i, (L, N) in enumerate(casos_b):
    calcular_y_graficar(L, N, axes_b[i])

plt.tight_layout()
plt.show()

# ==========================================
# CASO C: N alto fijo, L aumenta
# ==========================================
fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4))
fig_c.suptitle('Caso C: Mejora de la resolución espectral real (N=1024, L aumenta)', fontsize=14)

casos_c = [(32, 1024), (42, 1024), (64, 1024)]
for i, (L, N) in enumerate(casos_c):
    calcular_y_graficar(L, N, axes_c[i])

plt.tight_layout()
plt.show()