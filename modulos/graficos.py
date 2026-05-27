
# GRAFICAR FUNCIONES EN EL TIEMPO. revisar los rastros de copilot

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def graficar_t(fs, señales, etiquetas=None, titulo='Señales en el Tiempo', xlim=None):
    """
    Grafica una o varias señales en función del tiempo calculando 
    la duración automáticamente y permitiendo ajustar los límites del eje X.
    
    Parámetros
    ----------
    fs : int
        Frecuencia de muestreo en Hz de la señal.
    señales : NumPy array o lista de NumPy arrays
        Datos de la o las señales generadas.
    etiquetas : str o lista de str, opcional
        Etiquetas para identificar cada señal en la leyenda.
    titulo : str, opcional
        Título del gráfico.
    xlim : tuple o list, opcional
        Límites para el eje X, ej: (0, 0.01). Si es None, muestra el tiempo completo.
    """
    # Si se recibe una única señal, la convertimos en lista
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    if etiquetas is None:
        etiquetas = [f'Señal {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Recorremos cada señal
    for s, etiqueta in zip(señales, etiquetas):
        muestras = len(s) 
        duracion = muestras / fs 
        t = np.arange(0, duracion, 1/fs)
        
        ax.plot(t[:muestras], s, label=etiqueta)
    
    # Si el usuario definió xlim, lo aplicamos al eje X
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.legend(loc='upper right') 
    ax.grid(True)
    plt.show()
    


# GRAFICAR FUNCIONES EN FRECUENCIA. revisar los rastros de copilot. cuidado con la escala logarítmica  (límites?)
def graficar_f(fs, señales, etiquetas=None, titulo='Espectro de Amplitud (Fourier)', xlim=None):
    """
    Calcula y grafica el espectro de magnitud de Fourier en escala logarítmica.
    
    Parámetros
    ----------
    fs : int
        Frecuencia de muestreo en Hz.
    señales : NumPy array o lista de NumPy arrays
        Datos de la o las señales de entrada.
    etiquetas : str o lista de str, opcional
        Etiquetas para identificar cada espectro en la leyenda.
    titulo : str, opcional
        Título del gráfico.
    xlim : tuple, opcional
        Límites del eje X (fmin, fmax). Si es None o no se especifica, 
        se usa por defecto (20, 20000) Hz.
    """
    # 1. Normalización de las señales de entrada
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    # 2. Normalización de las etiquetas
    if etiquetas is None:
        etiquetas = [f'Espectro {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Procesamiento y cálculo de FFT
    for s, etiqueta in zip(señales, etiquetas):
        N = len(s)
        
        # FFT para señales reales y sus frecuencias correspondientes
        fft_vals = np.fft.rfft(s)
        frecuencias = np.fft.rfftfreq(N, 1/fs)
        
        # Magnitud normalizada para recuperar la amplitud real
        magnitud = np.abs(fft_vals) * (2.0 / N)
        
        # Corrección para componentes límites (Continua y Nyquist)
        magnitud[0] = magnitud[0] / 2.0
        if N % 2 == 0:
            magnitud[-1] = magnitud[-1] / 2.0

        ax.plot(frecuencias, magnitud, label=etiqueta)
        
    # 4. Configuración del eje X y escala logarítmica
    ax.set_xscale('log')
    
    # CONTROL DE LÍMITES: Si es None, asigna el rango clásico de audio (20 - 20kHz)
    if xlim is None:
        xlim = (20, 20000)
    ax.set_xlim(xlim)
    
    # 5. Configuración dinámica de Ticks según los límites elegidos
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    
    # Filtrar ticks para que solo se muestren los que entran en el rango elegido por el usuario
    ticks_filtrados = [t for t in ticks_audio if xlim[0] <= t <= xlim[1]]
    labels_filtrados = [l for t, l in zip(ticks_audio, labels_audio) if xlim[0] <= t <= xlim[1]]
    
    # Si el usuario eligió un rango muy específico donde no caen ticks estándar, 
    # dejamos que matplotlib los ponga automáticamente para no dejar el eje vacío.
    if len(ticks_filtrados) > 0:
        ax.set_xticks(ticks_filtrados)
        ax.set_xticklabels(labels_filtrados)
    
    # 6. Estética final del gráfico
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.legend(loc='upper right')
    
    plt.show()

def graficar_analisis(frecuencias, modulos, fases_rad, etiquetas=None, titulo='Caracterización'):
    
    """Función para graficar módulo y fase en radianes usando subplots."""
    if isinstance(modulos, np.ndarray) and modulos.ndim == 1:
        modulos = [modulos]
    if isinstance(fases_rad, np.ndarray) and fases_rad.ndim == 1:
        fases_rad = [fases_rad]
    if etiquetas is None:
        if len(modulos) == 1:
            etiquetas_modulo = ['Módulo']
            etiquetas_fase = ['Fase']
        else:
            etiquetas_modulo = [f'Filtro {i+1}' for i in range(len(modulos))]
            etiquetas_fase = etiquetas_modulo.copy()
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        etiquetas_modulo = etiquetas
        etiquetas_fase = etiquetas
    else:
        # Si sólo hay un filtro pero se pasaron dos etiquetas, la primera
        # se usa para módulo y la segunda para fase.
        if len(modulos) == 1 and len(etiquetas) == 2:
            etiquetas_modulo = [etiquetas[0]]
            etiquetas_fase = [etiquetas[1]]
        else:
            etiquetas_modulo = etiquetas
            etiquetas_fase = etiquetas
            if len(etiquetas_fase) < len(modulos):
                etiquetas_fase = etiquetas_modulo

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for mod, f_rad, etiqueta_mod, etiqueta_fase in zip(modulos, fases_rad, etiquetas_modulo, etiquetas_fase):
        fase_rad_continua = np.unwrap(f_rad)
        freqs_recortadas = frecuencias[:len(mod)]
        
        ax1.plot(freqs_recortadas, mod, label=etiqueta_mod, lw=2)
        ax2.plot(freqs_recortadas, fase_rad_continua, label=etiqueta_fase, lw=1.5, color='red')
        
    # Estética Módulo
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')
    ax1.set_title(titulo, fontsize=14)
    ax1.set_ylabel("|H(w)|")
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Estética Fase
    ax2.spines['left'].set_position('zero')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Fase (rad)")
    ax2.set_ylim(-np.pi, np.pi)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()



"""
# Definir tasa de muestreo y número de muestras
fs = 44100
samples = 1000
t = np.arange(samples) / fs

# Señales sinusoidales de ejemplo
signal_1 = np.sin(2 * np.pi * 500 * t)
signal_2 = np.sin(2 * np.pi * 1000 * t)
"""
"""
signal_1, fs= sf.read("C:\\Users\\Familia\\Documents\\DONELLA 2026\\Graba iii\\BVox 1 RX_Melodyne.wav")
signal_2, fs= sf.read("C:\\Users\\Familia\\Documents\\DONELLA 2026\\Graba iii\\tp edicion\\PerdidosNaZonaSul_MeuBem_Full\\17_ElecGtr1.wav")


graficar_f(fs, [signal_1, signal_2], ["señal 1", "señal 2"], "espectros")
"""



