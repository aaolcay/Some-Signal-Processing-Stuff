clc; clear all; close all;
fs = 600;
T  = 3;
N  = T * fs;
clicks_Yuang  = zeros(N,1);
[b,a] = butter(9, 150/(fs/2), 'high');
num_clicks   = 3;
click_len    = 0.05;
interval_click = 0.9;

i0 = randi([1, N - round(2*fs)]); % I want to have random starting point (not exceeding 1 second)
i1 = i0 + round(click_len*fs)-1;

for k = 1:num_clicks
    if k==1
        noise_Yuang= randn(i1-i0+1,1);
        pulse  = filtfilt(b, a, noise_Yuang);
        pulse  = pulse.*hann(length(pulse));
        clicks_Yuang(i0:i1) = clicks_Yuang(i0:i1) + pulse;
    else
        i0_next = i1 + round(interval_click*fs);
        i1_next = i0_next + round(click_len*fs) - 1;

        noise_Yuang = randn(i1_next-i0_next+1,1);
        pulse  = filtfilt(b, a, noise_Yuang);
        pulse  = pulse.*hann(length(pulse));
        clicks_Yuang(i0_next:i1_next) = clicks_Yuang(i0_next:i1_next) + pulse;
        i1 = i1_next;
    end
end
x = clicks_Yuang; % + 0.008.*randn(length(clicks_Yuang),1);
figure;
spectrogram(x,128, 120, 512, 600, 'yaxis');