import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Analog signal (assumed to be)
sr, signal = wavfile.read('speech.wav')

# rescale the analog signal's amplitude between -100 and 100
norm = 1.0/max(np.abs([min(signal),max(signal)]))
# Using 8 bits per audio sample means audio samples can take the integer values
# between -100 and 100: 1 0 0 0 0 0 0 0 -> 127
signal_analog = 100.0*signal*norm # ranging between -100 and 100
 
# Apply quantization, as digital signal takes only integer values
signal_digital = np.round(signal_analog)

error = signal_analog-signal_digital
# we expect to see maximum error as 0.5

plt.figure(figsize=(10,5))
plt.plot(signal_analog)
plt.title('Analog Signal')
plt.xlabel('samples')
plt.ylabel('Analog Signal')

plt.figure(figsize=(10,5))
plt.plot(signal_digital)
plt.title('Digital Signal')
plt.xlabel('samples')
plt.ylabel('Digital Signal')

plt.figure(figsize=(10,5))
plt.plot(error)
plt.title('Quantization Error')
plt.xlabel('samples')
plt.ylabel('error')

# Compute the SNR (signal-to-noise-ratio)
# 1) If you do not have the noise signal, but noisy signal
def calculate_SNR_1(noisy_signal, original_signal):
    # find error (i.e., noise)
    err = original_signal-noisy_signal
    # find the power of error signal
    pow_err = np.linalg.norm(err)**2
    # np.linalg.norm(err) computes the euclidean distance (i.e., RMS value), so
    # we need to take its squared value
    
    # find the power of original signal
    pow_signal = np.linalg.norm(original_signal)**2
    
    return 10*np.log10(pow_signal/pow_err)

# 2) If you have the noise you added to the original signal
def calculate_SNR_2(noise, original_signal):
    
    # find the power of error signal (noise) that can be used to estimate 
    # the variance of the noise
    pow_err = np.linalg.norm(noise)**2
    # np.linalg.norm(err) computes the euclidean distance (i.e., RMS value), so
    # we need to take its squared value
    
    # find the power of original signal
    pow_signal = np.linalg.norm(original_signal)**2
    
    return 10*np.log10(pow_signal/pow_err)


# In our case, we have our noisy signal, so we use the first SNR computation
# The SNR value below represents the signal-to-noise-ratio between analog signal
# and the digital signal acquired after quantization
SNR1 = calculate_SNR_1(noisy_signal=signal_digital, original_signal=signal_analog)
print(f"SNR computed with the first function: {SNR1}")
# We can also compute this SNR value as we already calculated the error signal
# that is basically the noise
SNR2 = calculate_SNR_2(noise=error, original_signal=signal_analog)
print(f"SNR computed with the second function: {SNR2}")
