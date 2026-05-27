# GRAFICAR FUNCIONES EN EL TIEMPO. revisar los rastros de copilot

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

# --- FUNCIONES DE GENERACION DE SENALES ---
def generar_tono_puro(amplitud, duracion, fs, frecuencia):
    """
    Genera un tono sinusoidal puro.
    Parámetros
    ----------
    frecuencia : int
        Frecuencia de oscilación de la señal.
    duracion: float
        Duración en segundos de la señal.
    fs : int
        Frecuencia de muestreo en Hz.
    amplitud:
        Valor del pico de la señal sinusoidal.
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    tono_puro = amplitud * np.sin(2 * np.pi * frecuencia * t)
    return tono_puro

def generar_ruido_blanco(amplitud, duracion, fs):
    """
    Genera ruido blanco, bajo el metodo de dispersion normal.
    Parámetros
    ----------
    amplitud: float
        Valor medio de dispersión del ruido blanco.
    duracion: float
        Duración en segundos de la señal.
    fs : int
        Frecuencia de muestreo en Hz.
    """
    longitud = int(fs * duracion)
    ruido_blanco_random = np.random.normal(loc=0, scale=1.0, size=longitud)     # Genera un ruido con media 0, desviación estandar 1.
    
    valor_pico = np.max(np.abs(ruido_blanco_random))                            # Calcula el pico de la señal de ruido
    ruido_blanco_normalizado = (ruido_blanco_random / valor_pico)               # Normaliza el ruido con el valor absoluto del pico     
    ruido_blanco_final = ruido_blanco_normalizado * amplitud                    # Multiplica el ruido por el valor de amplitud buscado
    return ruido_blanco_final

def sumar_senales(*senales):
    """
    Suma n cantidad de señales, rellenando las más cortas con ceros para obtener un largo total de la señal mas larga.
    Parámetros
    ----------
    senales: array
    """
    if not senales:
        return np.array([])
  
    largo_max = max(len(s) for s in senales)                                    # Encuentra el largo de la señal más larga
  
    resultado = np.zeros(largo_max)                                             # Crea un array de resultados con ceros
  
    for s in senales:                                                           # Rellena cada señal con ceros hasta el maximo y la suma al resultado
        s_pad = np.pad(s, (0, largo_max - len(s)), mode='constant')             # np.pad agrega ceros al final hasta llegar a 'largo_max'
        resultado += s_pad
        
    return resultado

def generar_arpegio(duracion_arpegio, fs, *frecuencias):
    duracion_nota = duracion_arpegio / len(frecuencias)                         # Calcula la duración de cada nota para que el total sume los segundos esperados
    notas_audio = []
    
    for f in frecuencias:                                                       # Genera cada tono y lo guarda en la lista
        tono = generar_tono_puro(1, duracion_nota, fs, f)
        notas_audio.append(tono)

    audio_final = np.concatenate(notas_audio)                                   # Concatena todas las notas en una sola señal larga
    
    return audio_final, fs

# --- FUNCIONES DE GRAFICACION ---
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
    
    plt.subplots_adjust()
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
    
    plt.subplots_adjust()
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

    plt.subplots_adjust()
    plt.show()


def respuesta_f(entrada, salida, fs):
    """
    Calcula la respuesta en frecuencia de un sistema dado su señal de entrada y salida.
    
    Parámetros
    ----------
    entrada : NumPy array
        Señal de entrada al sistema.
    salida : NumPy array
        Señal de salida del sistema.
    fs : int
        Frecuencia de muestreo en Hz.
        
    Retorna
    -------
    freqs : NumPy array
        Frecuencias correspondientes a la respuesta en frecuencia.
    H : NumPy array
        Respuesta en frecuencia del sistema (magnitud y fase).
    """
    # Convertir a float64 para evitar problemas con FFT
    entrada = np.asarray(entrada, dtype=np.float64)
    salida = np.asarray(salida, dtype=np.float64)
    
    # Aseguramos que ambas señales tengan la misma longitud con zero padding
    max_len = max(len(entrada), len(salida))
    entrada_padded = np.pad(entrada, (0, max_len - len(entrada)), mode='constant')
    salida_padded = np.pad(salida, (0, max_len - len(salida)), mode='constant')
    
    # Calculamos la FFT de ambas señales
    N = len(entrada_padded)
    E = np.fft.fft(entrada_padded)
    S = np.fft.fft(salida_padded)
    
    # Evitamos divisiones por cero
    E_magnitude = np.abs(E)
    E_magnitude[E_magnitude == 0] = 1e-10  # Pequeño valor para evitar división por cero
    
    # Calculamos la respuesta en frecuencia H(f) = S(f) / E(f)
    H = S / E
    
    magnitud_H = np.abs(H)
    fase_H = np.angle(H)
    
    # Calculamos las frecuencias correspondientes a cada bin de la FFT
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    return freqs, magnitud_H, fase_H


# FUNCIONES DE FILTROS
def filtros_media_movil(M,N):
    """
    Genera las respuestas al impulso para un filtro de media móvil
    de 1, 2 y 3 pasadas.
    
    Parámetros:
    M (int): El largo de la ventana del filtro.
    N (int): La longitud de la respuesta al impulso.
    
    Retorna:
    tuple: (h1, h2, h3) que son los arrays (numpy arrays) de las respuestas al impulso.
    """
    
    delta = np.zeros(N)
    delta[0] = 1

    # 1. Primera pasada: h[n] = 1/M
    # Creamos un arreglo de tamaño M donde cada valor es 1/M
    p1 = np.ones(M) / M
    
    # 2. Segunda pasada: Convolución de h1 consigo misma
    p2 = np.convolve(p1, p1)
    
    # 3. Tercera pasada: Convolución de h2 con h1
    p3 = np.convolve(p2, p1)

    h1 = np.convolve(delta, p1)[:N]
    h2 = np.convolve(delta, p2)[:N]
    h3 = np.convolve(delta, p3)[:N]
    
    return h1, h2, h3

def filtro_peine(b0, b1, b2, N):
    """
    Genera la respuesta al impulso h[n] para un filtro peine FIR.
    
    Parámetros
    ----------
    b0, b1, b2 : float
        Coeficientes constantes del filtro.
        
    Retorna
    -------
    h : NumPy array
        Vector con la respuesta al impulso h[n].
    """

    
    delta = np.zeros(N)
    delta[0] = 1

    filtro = np.array([b0,b1,b2])
    h = np.convolve(delta, filtro)[:N]

    return h 


def filtro_fir(N, path):
    #esto funciona solamente para mi compu pero bueno
    delta = np.zeros(N)
    delta[0] = 1
    
    coeficientes_fir = np.load(path)
    h = np.convolve(delta, coeficientes_fir)[:N]

    return h

def analisis_filtros(filtro, fs):
    """
    Toma la respuesta al impulso de un filtro, calcula su FFT real 
    y devuelve el vector de frecuencias, el módulo y la fase en radianes.
    
    Parámetros
    ----------
    filtro : NumPy array
        Respuesta al impulso del filtro (coeficientes).
    fs : int
        Frecuencia de muestreo en Hz.
        
    Retorna
    -------
    freqs : NumPy array (Vector de frecuencias en Hz)
    modulo : NumPy array (Magnitud o ganancia del filtro)
    fase_rad : NumPy array (Fase del filtro en radianes)
    """
    N = len(filtro)
    H_w = np.fft.rfft(filtro)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    modulo = np.abs(H_w)
    fase_rad = np.angle(H_w)

    return freqs, modulo, fase_rad


def filtrar_temporal(h_filtro, señal):
    """
    agregar docstring
    """
    señal_filtrada = np.convolve(señal, h_filtro, mode='same')

    return señal_filtrada

def filtrar_frecuencial(h_filtro, señal, fs):
    """
    agregar docstring
    """
    H_w = np.fft.rfft(h_filtro, n=len(señal))
    X_w = np.fft.rfft(señal)
    Y_w = H_w * X_w

    señal_filtrada =np.fft.irfft(Y_w, n=len(señal))

    return señal_filtrada

# EXTRA
def descargar_wav_normalizado(audio, fs, nombre_archivo):
    valor_pico = np.max(np.abs(audio))
    audio_normalizado = audio / valor_pico
    sf.write(f'{nombre_archivo}.wav', audio_normalizado, fs)
    
def leer_audio (nombre_archivo, carpeta_de_archivos):
    ruta_completa = os.path.join(carpeta_de_archivos, nombre_archivo)
    audio, fs = sf.read(ruta_completa)
    
    return audio, fs

        

