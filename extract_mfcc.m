function mfcc_features = extract_mfcc(audio_file)
% Read the audio file
[signal, fs] = audioread(audio_file);

% Parameters
frame_size = 0.03; % 30 ms
hop_size = 0.01;   % 10 ms
num_mel_filters = 26; % Number of Mel filters
num_mfcc_coefs = 13; % Number of MFCC coefficients

% Frame segmentation
frame_length = round(frame_size * fs);
hop_length = round(hop_size * fs);
signal_length = length(signal);
num_frames = floor((signal_length - frame_length) / hop_length) + 1;

% Apply Hamming window
window = hamming(frame_length);

% Preallocate storage for MFCC features
mfcc_features = zeros(num_frames, num_mfcc_coefs); % Preallocate the array

for i = 1:num_frames
    % Frame signal
    frame_start = (i - 1) * hop_length + 1;
    frame_end = frame_start + frame_length - 1;
    if frame_end > signal_length
        break;
    end
    frame = signal(frame_start:frame_end) .* window;

    % FFT and Power Spectrum
    fft_spectrum = abs(fft(frame, 512)).^2;
    power_spectrum = fft_spectrum(1:257); % Keep all positive frequencies
    power_spectrum = power_spectrum(:); % Ensure it's a column vector

    % Mel Filterbank
    mel_filterbank = mel_filterbank_matrix(num_mel_filters, 512, fs);

    % Multiply Mel filterbank with power spectrum
    mel_energies = mel_filterbank * power_spectrum;

    % Log-Mel Energies
    log_mel_energies = log(mel_energies + eps);

    % Discrete Cosine Transform (DCT)
    mfcc_frame = dct(log_mel_energies);
    mfcc_frame = mfcc_frame(1:num_mfcc_coefs); % Retain first 'num_mfcc_coefs' coefficients

    % Store the MFCC frame in the preallocated matrix
    mfcc_features(i, :) = mfcc_frame';
end
end

function mel_filterbank = mel_filterbank_matrix(num_filters, fft_size, fs)
% Mel scale filterbank
low_freq_mel = 0;
high_freq_mel = 2595 * log10(1 + (fs / 2) / 700);
mel_points = linspace(low_freq_mel, high_freq_mel, num_filters + 2);
hz_points = 700 * (10.^(mel_points / 2595) - 1);
bin = floor((fft_size + 1) * hz_points / fs);

% Ensure bin indices are valid
bin(bin < 1) = 1; % Replace any zero or negative values with 1
bin(bin > fft_size / 2 + 1) = floor(fft_size / 2 + 1);

% Filterbank matrix
mel_filterbank = zeros(num_filters, floor(fft_size / 2 + 1));
for i = 1:num_filters
    mel_filterbank(i, bin(i):bin(i+1)) = linspace(0, 1, bin(i+1) - bin(i) + 1);
    mel_filterbank(i, bin(i+1):bin(i+2)) = linspace(1, 0, bin(i+2) - bin(i+1) + 1);
end
end