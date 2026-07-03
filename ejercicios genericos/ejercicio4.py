import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

x, fs = sf.read("archivo_ejc3.wav")
print(fs)

#t = np.arange(len(x)) / fs
t = np.arange(start = 0,stop = len(x)/fs,step=1/fs)

plt.figure(figsize=(10, 6))
plt.plot(t, x, color="black")
plt.title("Señal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show() 

inicio = np.argmax(x)

print("inicio ", t[inicio])


x_nueva = x[inicio:inicio+int(0.005*fs)]
t_nueva = np.arange(len(x_nueva)) / fs

plt.figure(figsize=(10, 6))
plt.plot(t_nueva, x_nueva, color="black")
plt.title("Señal nueva")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.show() 

x_nueva_fft = np.fft.rfft(x_nueva)
w_fft = np.arange(start = 0,stop = fs//2,step=(fs//2)/len(x_nueva_fft))
plt.plot(w_fft, np.abs(x_nueva_fft), color="black")
plt.title("FFT")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.show() 