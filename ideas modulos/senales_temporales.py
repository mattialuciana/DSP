# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:54:10 2026

@author: dell_
"""

import soundfile as sf
import numpy as np
import os

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

def descargar_wav_normalizado(audio, fs, nombre_archivo):
    valor_pico = np.max(np.abs(audio))
    audio_normalizado = audio / valor_pico
    sf.write(f'{nombre_archivo}.wav', audio_normalizado, fs)
    
def leer_audio (nombre_archivo, carpeta_de_archivos):
    ruta_completa = os.path.join(carpeta_de_archivos, nombre_archivo)
    audio, fs = sf.read(ruta_completa)
    
    return audio, fs

        
        