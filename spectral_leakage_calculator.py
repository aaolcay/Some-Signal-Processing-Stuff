import numpy as np
import matplotlib.pyplot as plt

# Generate a 25 Hz sine wave sampled at 100 Hz
f = 25
fs = 100
N = 200
resolution = fs/N
if f % resolution == 0:
    print("There is no spectral leakage :)")
else:
     print("There is spectral leakage :(, please use tapered shape window")
t = np.arange(0, N/fs, 1/fs)
x = np.sin(2*np.pi*f*t)
plt.figure()
plt.plot(t, x)
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.show()


# Compute the DFT of the signal
X = np.fft.fft(x)

# Plot the magnitude spectrum of the DFT
freq = np.fft.fftfreq(len(x), 1/fs)
mag = np.abs(X)
plt.figure()
plt.stem(freq, mag)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()