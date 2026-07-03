import numpy as np


x = [1, 1j, -1, -1j]
x_real =np.real(x)
x_imag = np.imag(x)
X = np.fft.fft(x_imag)
print("x(n):", x_imag)
print("X(k):", X)