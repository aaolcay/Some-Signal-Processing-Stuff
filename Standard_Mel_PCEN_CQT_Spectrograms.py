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
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
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

# Compute Constant Q-Transform spectrogram
nbins = 12*np.log2((sr/2)/32.70)
cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=int(nbins)+1, fmin=32.70, bins_per_octave=12) 
# f_min-> Defaults to C1 ~= 32.70 Hz
# bins_per_octave (b)-> Defaults to 12
# n_bins number of frequency bins, starting at f_min-> Defaults to 84 
# 84/12 = 7 -> 7 Octaves we have (see below)
#############################################################################################
#############################################################################################

# Display the CQT
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),
                         sr=sr, x_axis='time', y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q Transform (CQT)')
plt.tight_layout()
plt.show()

"""
###########################################################
############## Computing Number of Octaves ################
######################## in CQT ###########################
###########################################################
To determine the number of octaves that can be visualized with the given 
parameters, we need to calculate the frequency range covered by the Constant-Q 
Transform (CQT).

The number of octaves can be calculated using the formula:

number of octaves = (log2(f_max) - log2(f_min)) / log2(2)

In this case, we are given the following parameters:

f_min = C1 = 32.70 Hz
bins_per_octave (b) = 12
n_bins = 84
To find f_max, we can use the formula:

f_max = f_min * 2^(n_bins/b * log2(2))

Let's calculate the values:

f_max = 32.70 * 2^(84/12 * log2(2))
= 32.70 * 2^7
= 32.70 * 128
= 4185.60 Hz

Now, we can calculate the number of octaves:

number of octaves = (log2(4185.60) - log2(32.70)) / log2(2)
= (12.977 - 5) / 1
= 7.977

So, with the given parameters, you can visualize approximately 7.977 octaves.

#######################################################################################
#######################################################################################
## say we are looking for the value of n_bins in the case we know the value of f_max ##
#######################################################################################
#######################################################################################
f_max = fs/2 = 44100/2 = 22050
f_min = C1 = 32.70 Hz
bins_per_octave (b) = 12
#######################################################
#######################################################
###### n_bins = bins_per_octave*log2(f_max/f_min)######
#######################################################
#######################################################
n_bins = 12*log2(22050/32.70) ~= 112
"""