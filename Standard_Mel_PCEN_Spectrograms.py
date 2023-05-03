import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
y, sr = librosa.load(librosa.ex('robin'))

#############################################################################################
#############################################################################################
#Compute Standard Spectrogram

X_STFT = librosa.stft(y, n_fft = 2048, hop_length=512)
X_log_scale = librosa.power_to_db(np.abs(X_STFT)**2)

#############################################################################################
#############################################################################################
# Compute mel-spectrogram
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
# The mel-spectrogram computed by librosa.feature.melspectrogram() is already a power 
# spectrogram (as it is default power=2.0), so there is no need to square it before computing
# the PCEN spectrogram
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
#############################################################################################
#############################################################################################
# Compute PCEN spectrogram
pcen_spectrogram = librosa.pcen(spectrogram*(2**31), sr=sr)
#############################################################################################
#############################################################################################

# Plot standard spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(X_log_scale, x_axis='time', y_axis='linear', sr=sr, n_fft = 2048, hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('Standard Spectrogram\n(log of squared magnitude)')
plt.tight_layout()

# Plot Mel-spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_spectrogram, x_axis='time', y_axis='mel', sr=sr, n_fft = 2048, hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram')
plt.tight_layout()

# Plot PCEN spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(pcen_spectrogram, x_axis='time', y_axis='mel', sr=sr, n_fft = 2048, hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('PCEN Spectrogram')
plt.tight_layout()
