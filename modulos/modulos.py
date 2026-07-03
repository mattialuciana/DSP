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
        
    Retorna
    ----------
    tono_puro : NumPy array
        Valores en tiempo correspondientes a la señal sinusoidal.
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
        
    Retorna
    ----------
    ruido_blanco_final: NumPy array
        Valores en tiempo correspondientes al ruido blanco.
    """
    
    longitud = int(fs * duracion)
    
    # Genera un ruido con media 0, desviación estandar 1.
    ruido_blanco_random = np.random.normal(loc=0, scale=1.0, size=longitud)     
    
    # Calcula el pico de la señal de ruido
    valor_pico = np.max(np.abs(ruido_blanco_random))                

    # Normaliza el ruido con el valor absoluto del pico              
    ruido_blanco_normalizado = (ruido_blanco_random / valor_pico)            

    # Multiplica el ruido por el valor de amplitud buscado      
    ruido_blanco_final = ruido_blanco_normalizado * amplitud                    
    return ruido_blanco_final


def sumar_senales(*senales):
    """
    Suma n cantidad de señales, rellenando las más cortas con ceros para obtener un largo total de la señal mas larga.
    
    Parámetros
    ----------
    senales: lista de NumPy array.
        La cantidad de arrays por separado que se deseen superponer. (Pueden ser de largos distintos)
    
    Retorna
    ----------
    resultado: NumPy array
        Valores en tiempo correspondientes a la señal compuesta.
    """
    
    if not senales:
        return np.array([])
    
    # Encuentra el largo de la señal más larga
    largo_max = max(len(s) for s in senales)                                    
  
    # Crea un array de resultados con ceros
    resultado = np.zeros(largo_max)                                             
  
    # Rellena cada señal con ceros hasta el maximo y la suma al resultado
    # np.pad agrega ceros al final hasta llegar a 'largo_max'
    for s in senales:                                                          
        s_pad = np.pad(s, (0, largo_max - len(s)), mode='constant')             
        resultado += s_pad
        
    return resultado


def generar_arpegio(duracion_arpegio, fs, *frecuencias):
    """
    Recibe una duración total y valores de frecuencias particulares para armar una concatenación temporal de esas frecuencias.
    Cada frecuencia suelta durará lo mismo en la señal total.
    
    Parámetros
    ----------
    duracion_arpegio: float.
        Cantidad de segundos totales que va a durar la señal final.
    fs: int
        Frecuencia de muestreo de las operaciones.
    frecuencias: float
        Valores sueltos de las frecuencias que serán parte de la señal, en el orden en el que se deseen.
    
    Retorna
    ----------
    audio_final: NumPy array
        Valores en tiempo correspondientes a la señal compuesta.
    """
    
    # Calcula la duración de cada nota para que el total sume los segundos esperados
    duracion_nota = duracion_arpegio / len(frecuencias)                         
    notas_audio = []
    
    # Genera cada tono y lo guarda en la lista
    for f in frecuencias:                                                       
        tono = generar_tono_puro(1, duracion_nota, fs, f)
        notas_audio.append(tono)
        
    # Concatena todas las notas en una sola señal larga
    audio_final = np.concatenate(notas_audio)                                   
    
    return audio_final


# --- FUNCIONES DE GRAFICACION ---
def graficar_t(fs, señales, etiquetas=None, titulo='Señales en el Tiempo', xlim=None, modo=None):
    """
    Grafica una o varias señales en función del tiempo calculando 
    la duración automáticamente. Permitiendo ajustar los límites del eje X y la visualización en modo discreto.
    
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
    modo : str, opcional
        Si se establece a 'discreto', grafica la señal discretamente usando
        marcadores y líneas tipo "stem". Por defecto (None) grafica en
        forma continua con plot().
        
    Retorna
    ----------
    Muestra el gráfico compuesto en tiempo.
    """
    
    # Si se recibe una única señal, la convertimos en lista
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    if etiquetas is None:
        etiquetas = [f'Señal {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Recorremos cada señal
    for i, (s, etiqueta) in enumerate(zip(señales, etiquetas)):
        muestras = len(s) 
        duracion = muestras / fs 
        t = np.arange(0, duracion, 1/fs)
        color = colores[i % len(colores)]
        
        if modo == 'discreto':
            markerline, stemlines, baseline = ax.stem(
                t[:muestras], s, label=etiqueta, linefmt=color, markerfmt='o', basefmt=" "
            )
            markerline.set_color(color)
            stemlines.set_color(color)
            baseline.set_color(color)
        else:
            ax.plot(t[:muestras], s, label=etiqueta, color=color)
    
    # Si el usuario definió xlim, lo aplicamos al eje X
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_title(titulo)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.legend(loc='upper right') 
    ax.grid(True)
    plt.show()  
    
    
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
    
    Retorna
    ----------
    Muestra el gráfico compuesto en frecuencia.
    """
    
    # Normaliza las señales de entrada y las convierte en lista.
    if isinstance(señales, np.ndarray) and señales.ndim == 1:
        señales = [señales]
        
    # Normaliza las etiquetas en caso de no haber especificado.
    if etiquetas is None:
        etiquetas = [f'Espectro {i+1}' for i in range(len(señales))]
    elif isinstance(etiquetas, str):
        etiquetas = [etiquetas]
        
    fig, ax = plt.subplots(figsize=(10, 6))

    # Procesamiento y cálculo de FFT
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
        
    # Configuración del eje X y escala logarítmica
    ax.set_xscale('log')
    
    # Control de límites: Si es None, asigna el rango clásico de audio (20 - 20kHz)
    if xlim is None:
        xlim = (20, fs/2)
    ax.set_xlim(xlim)
    
    # Configura los marcadores típicos de audio según los límites elegidos
    ticks_audio = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    labels_audio = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
    
    # Filtra los ticks para que solo se muestren los que entran en el rango elegido por el usuario
    ticks_filtrados = [t for t in ticks_audio if xlim[0] <= t <= xlim[1]]
    labels_filtrados = [l for t, l in zip(ticks_audio, labels_audio) if xlim[0] <= t <= xlim[1]]
    
    # Si el usuario eligió un rango muy específico donde no caen ticks estándar, 
    # dejamos que matplotlib los ponga automáticamente para no dejar el eje vacío.
    if len(ticks_filtrados) > 0:
        ax.set_xticks(ticks_filtrados)
        ax.set_xticklabels(labels_filtrados)
    
    # Estética final del gráfico
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    ax.set_title(titulo)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.legend(loc='upper right')
    
    plt.subplots_adjust()
    plt.show()


def graficar_analisis(frecuencias, modulos, fases_rad, etiquetas=None, titulo='Caracterización', modo='log'):
    """
    Función para graficar módulo (azul) y fase (rojo) en radianes.
    Mantiene la fase acotada entre -pi y pi.
    
    Parámetros
    ----------
    frecuencias : NumPy array
        Array con las muestras de frecuencias del eje x.
    modulos : NumPy array
        Datos del modulo de la señal de entrada.
    etiquetas : str, opcional
        Etiquetas de la señal a graficar.
    titulo : str, opcional
        Título del gráfico.
    modo : str, opcional
        Escala del eje x: 'log' para logarítmica o 'lin' para lineal.
        
    Retorna
    ----------
    Muestra dos gráficos indicando módulo y fase del espectro de la señal.
    """
    
    # Validaciones para asegurar que modulos y fases sean listas/iterables
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
        if len(modulos) == 1 and len(etiquetas) == 2:
            etiquetas_modulo = [etiquetas[0]]
            etiquetas_fase = [etiquetas[1]]
        else:
            etiquetas_modulo = etiquetas
            etiquetas_fase = etiquetas
            if len(etiquetas_fase) < len(modulos):
                etiquetas_fase = etiquetas_modulo

    # Creación de los subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Bucle de graficación
    for i, (mod, f_rad, etiqueta_mod, etiqueta_fase) in enumerate(zip(modulos, fases_rad, etiquetas_modulo, etiquetas_fase)):
        freqs_recortadas = frecuencias[:len(mod)]
        
        # Variamos la transparencia (alpha) si hay múltiples curvas para poder distinguirlas
        alpha_val = 1.0 - (i * 0.2) if len(modulos) > 1 else 1.0
        
        # Gráfico de Módulo en AZUL
        ax1.plot(freqs_recortadas, mod, label=etiqueta_mod, lw=2, color='blue', alpha=max(alpha_val, 0.4))
        
        # Gráfico de Fase en ROJO (directo, SIN np.unwrap)
        ax2.plot(freqs_recortadas, f_rad, label=etiqueta_fase, lw=1.5, color='red', alpha=max(alpha_val, 0.4))
        
    # -- Estética del Módulo --
    # Dibujamos ejes cruzados visuales en cero.
    ax1.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax1.axvline(0, color='black', linewidth=0.8, zorder=1)
    
    # Ocultamos solo los bordes superior y derecho para dar un aspecto de plano cartesiano limpio
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    if modo == 'log':
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    elif modo == 'lin':
        ax1.set_xscale('linear')
        ax2.set_xscale('linear')
    else:
        raise ValueError("modo debe ser 'lin' o 'log'")

    # Ajuste de límites para que la escala lineal no quede forzada por el eje compartido
    if modo == 'lin':
        xlim = (min(frecuencias), max(frecuencias))
        ax1.set_xlim(xlim)
        ax2.set_xlim(xlim)

    ax1.set_title(titulo, fontsize=14)
    ax1.set_ylabel(r"$|H(\omega)|$", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # -- Estética de la Fase --
    ax2.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax2.axvline(0, color='black', linewidth=0.8, zorder=1)
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    ax2.set_xlabel("Frecuencia (Hz)", fontsize=12)
    ax2.set_ylabel("Fase (rad)", fontsize=12)
    
    # Acotamos el eje Y, sumando un pequeño margen
    ax2.set_ylim(-np.pi - 0.5, np.pi + 0.5)
    
    # Configuramos los ticks del eje Y para mostrar múltiplos de Pi limpios
    ax2.set_yticks([-np.pi, 0, np.pi])
    ax2.set_yticklabels([r'$-\pi$', '0', r'$\pi$'], fontsize=11)
    
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout() 
    plt.show()

# --- FUNCIONES DE ARMADO DE FILTROS ---
def filtros_media_movil(M,N):
    """
    Genera las respuestas al impulso para un filtro de media móvil
    de 1, 2 y 3 pasadas.
    
    Parámetros
    ----------
    M: int
        El largo de la ventana del filtro.
    N: int
        La longitud de la respuesta al impulso.
    
    Retorna
    ----------
    h1: NumPy array
        Respuesta al impulso de la primera pasada del filtro.
    h2: NumPy array
        Respuesta al impulso de la segunda pasada del filtro.
    h3: NumPy array
        Respuesta al impulso de la tercera pasada del filtro.
    """
    
    delta = np.zeros(N)
    delta[0] = 1

    # Primera pasada: h[n] = 1/M
    # Creamos un arreglo de tamaño M donde cada valor es 1/M
    p1 = np.ones(M) / M
    
    # Segunda pasada: Convolución de h1 consigo misma
    p2 = np.convolve(p1, p1)
    
    # Tercera pasada: Convolución de h2 con h1
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
    ----------
    h : NumPy array
        Vector con la respuesta al impulso h[n].
    """

    delta = np.zeros(N)
    delta[0] = 1

    filtro = np.array([b0,b1,b2])
    h = np.convolve(delta, filtro)[:N]

    return h 


def filtro_fir(N, path):
    """
    Genera la respuesta al impulso h[n] para un filtro FIR, importando parametros de un archivo .py externo.
    
    Parámetros
    ----------
    N: cantidad de muestras.
    path: camino relativo donde se encuentran los coeficientes del filtro.
        
    Retorna
    ----------
    h : NumPy array
        Vector con la respuesta al impulso h[n].
    """
    
    delta = np.zeros(N)
    delta[0] = 1
    
    coeficientes_fir = np.load(path)
    h = np.convolve(delta, coeficientes_fir)[:N]

    return h


# --- FUNCIONES DE ANÁLISIS --- 
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
    ----------
    freqs : NumPy array
        Frecuencias correspondientes a la respuesta en frecuencia.
    magnitud_H : NumPy array
        Magnitud de la respuesta en frecuencia del sistema.
    fase_H: NumPy array
        Fase de la respuesta en frecuencia del sistema.
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


def analisis_filtros(filtro, fs):
    """
    Toma la respuesta al impulso de un filtro, calcula su FFT real 
    y devuelve el vector de frecuencias, el módulo y la fase acotada entre -pi y pi.
    
    Parámetros
    ----------
    filtro: NumPy array.
        Valores de muestras de la respuesta al impulso del filtro.
        fs: frecuencia de muestreo a la que se tomaron esos valores.
        
    Retorna
    ----------
    freqs: NumPy array. 
        Valores de frecuencias a la que le corresponderán un módulo y una fase ordenadas.
    modulo: NumPy array
        Valores ordenados de modulo
    fase_rad: NumPy array
        Valores ordenados de fase
    """
    
    N = len(filtro)
    H_w = np.fft.rfft(filtro)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    modulo = np.abs(H_w)
    
    # Obtenemos la fase base
    fase_raw = np.angle(H_w)
    
    # Forzamos matemáticamente a que esté en el rango [-pi, pi]
    fase_rad = (fase_raw + np.pi) % (2 * np.pi) - np.pi

    return freqs, modulo, fase_rad


# --- FUNCIONES DE FILTRADO --- 
def filtrar_temporal(h_filtro, señal):
    """
    Por método de convolución, filtra una señal.
    
    Parámetros
    ----------
    h_filtro: NumPy array.
        Valores en muestras temporales de la respuesta al impulso del filtro.
    señal: NumPy array.
        Valores en muestras temporales temporales de una señal.
        
    Retorna
    ----------
    señal_filtrada: NumPy array
        Valores en muestras temporales de la señal filtrada.
    """
    
    señal_filtrada = np.convolve(señal, h_filtro, mode='same')

    return señal_filtrada


def filtrar_frecuencial(h_filtro, señal, fs):
    """
    Por un lado realiza la FFT para convertir señales temporales en espectros frecuenciales.
    Luego por método de la multiplicación en frecuencias, filtra una señal.
    
    Parámetros
    ----------
    h_filtro: NumPy array.
        Valores en muestras temporales de la respuesta al impulso del filtro.
    señal: NumPy array.
        Valores en muestras temporales temporales de una señal.
    fs: int
        frecuencia de muestreo a la que se realizan los pasajes.
        
    Retorna
    ----------
    señal_filtrada: NumPy array
        Valores en muestras temporales de la señal filtrada.
    """
    
    H_w = np.fft.rfft(h_filtro, n=len(señal))
    X_w = np.fft.rfft(señal)
    Y_w = H_w * X_w

    señal_filtrada =np.fft.irfft(Y_w, n=len(señal))

    return señal_filtrada


# --- EXTRA ---
def descargar_wav_normalizado(audio, fs, nombre_archivo):
    """
    Descarga un archivo .wav, normalizando los valores entre 1 y -1 para cuidar los equipos de reproducción.
    
    Parámetros
    ----------
    audio: NumPy array.
        Valores en muestras temporales de la señal de audio.
    fs: int
        frecuencia de muestreo a la que se realizan los pasajes.
    nombre_archivo: str
        nombre con el que se desea guardar el archivo (va sin .wav)
        
    Retorna
    ----------
    El archivo .wav se descarga en el directorio donde se este trabajando.
    """
    
    valor_pico = np.max(np.abs(audio))
    audio_normalizado = audio / valor_pico
    sf.write(f'{nombre_archivo}.wav', audio_normalizado, fs)
    
    
def leer_audio (nombre_archivo, carpeta_de_archivos):
    """
    Lee un archivo .wav y guarda sus valores y su fs.
    
    Parámetros
    ----------
    carpeta_de_archivos: str
        Ruta completa de la carpeta donde se encuentre el archivo.
    nombre_archivo: str
        nombre comleto del archivo que se desea leer.
        
    Retorna
    ----------
    audio: NumPy array
        Valores muestreados en tiempo de la señal de audio.
    fs: int
        Frecuencia de muestreo a la que estaba el audio original.
    """
    
    ruta_completa = os.path.join(carpeta_de_archivos, nombre_archivo)
    audio, fs = sf.read(ruta_completa)
    
    return audio, fs

        

