clc; clear all; close all;
fs = 600;
T  = 3;
N  = T * fs;
clicks_olcay  = zeros(N,1);
[b,a] = butter(9, 150/(fs/2), 'high');
num_clicks   = 3;
click_len    = 0.05;
interval_click = 0.9;

i0 = randi([1, N - round(2*fs)]); % I want to have random starting point (not exceeding 1 second)
i1 = i0 + round(click_len*fs)-1;

for k = 1:num_clicks
    if k==1
        noise_olcay= randn(i1-i0+1,1);
        pulse  = filtfilt(b, a, noise_olcay);
        pulse  = pulse.*hann(length(pulse));
        clicks_olcay(i0:i1) = clicks_olcay(i0:i1) + pulse;
    else
        i0_next = i1 + round(interval_click*fs);
        i1_next = i0_next + round(click_len*fs) - 1;

        noise_olcay = randn(i1_next-i0_next+1,1);
        pulse  = filtfilt(b, a, noise_olcay);
        pulse  = pulse.*hann(length(pulse));
        clicks_olcay(i0_next:i1_next) = clicks_olcay(i0_next:i1_next) + pulse;
        i1 = i1_next;
    end
end
x = clicks_olcay; % + 0.008.*randn(length(clicks_olcay),1);
figure;
spectrogram(x,128, 120, 512, 600, 'yaxis');