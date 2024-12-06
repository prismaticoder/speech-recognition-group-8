% Step 1
% Before starting, rename 'EEEM030cw2-DevelopmentSet-2024' to 'DevSet'
function extract_mfcc_for_dataset(dataset_dir, output_file)
    % This function processes all audio files in the dataset directory,
    % extracts MFCC features, and saves them into a .mat file.
    %
    % Inputs:
    % - dataset_dir: Folder containing audio files.
    % - output_file: Name of the .mat file to save the results.

    % Get the list of all .mp3 files in the specified directory
    audio_files = dir(fullfile(dataset_dir, '*.mp3')); % https://www.mathworks.com/help/matlab/ref/dir.html
    num_files = length(audio_files); % https://www.mathworks.com/help/matlab/ref/length.html

    % Preallocate storage for MFCC features and file names
    all_mfcc_features = cell(num_files, 1); % https://www.mathworks.com/help/matlab/ref/cell.html
    file_names = cell(num_files, 1); 

    % Loop through each audio file
    for i = 1:num_files
        file_path = fullfile(audio_files(i).folder, audio_files(i).name); % https://www.mathworks.com/help/matlab/ref/fullfile.html
        fprintf('Processing file: %s\n', file_path); % https://www.mathworks.com/help/matlab/ref/fprintf.html

        try
            % Extract MFCC features using the helper function
            mfcc_features = extract_mfcc(file_path);
            all_mfcc_features{i} = mfcc_features; % Save the features
            file_names{i} = audio_files(i).name; % Save the file name
        catch ME
            % If something goes wrong, display the error message
            fprintf('Error processing file: %s\n', file_path); % https://www.mathworks.com/help/matlab/ref/fprintf.html
            fprintf('%s\n', ME.message);
        end
    end

    % Combine all extracted MFCC frames for global statistics
    all_frames = cell2mat(all_mfcc_features); % https://www.mathworks.com/help/matlab/ref/cell2mat.html
    global_mean = mean(all_frames, 1); % https://www.mathworks.com/help/matlab/ref/mean.html
    global_variance = var(all_frames, 0, 1); % https://www.mathworks.com/help/matlab/ref/var.html

    % Save everything into a .mat file
    save(output_file, 'all_mfcc_features', 'file_names', 'global_mean', 'global_variance'); % https://www.mathworks.com/help/matlab/ref/save.html
    fprintf('Processing complete. Results saved to %s.\n', output_file); 
end

function mfcc_features = extract_mfcc(audio_file)
    % This helper function calculates the MFCC features for a single audio file.
    %
    % Input:
    % - audio_file: Path to the .mp3 file.
    %
    % Output:
    % - mfcc_features: Matrix containing MFCC features for each frame.

    % Read the audio file into memory
    [signal, fs] = audioread(audio_file); % https://www.mathworks.com/help/matlab/ref/audioread.html

    % Set up parameters for frame segmentation and MFCC computation
    frame_size = 0.03; % 30 milliseconds
    hop_size = 0.01; % 10 milliseconds
    num_mel_filters = 26; % Number of Mel filters
    num_mfcc_coefs = 13; % Number of MFCC coefficients to keep

    % Frame segmentation: Calculate frame and hop lengths in samples
    frame_length = round(frame_size * fs); % https://www.mathworks.com/help/matlab/ref/round.html
    hop_length = round(hop_size * fs);
    signal_length = length(signal); % https://www.mathworks.com/help/matlab/ref/length.html
    num_frames = floor((signal_length - frame_length) / hop_length) + 1; % https://www.mathworks.com/help/matlab/ref/floor.html

    % Create a Hamming window to apply to each frame
    window = hamming(frame_length); % https://www.mathworks.com/help/signal/ref/hamming.html

    % Preallocate space for the MFCC features
    mfcc_features = zeros(num_frames, num_mfcc_coefs); % https://www.mathworks.com/help/matlab/ref/zeros.html

    for i = 1:num_frames
        % Define the start and end of the current frame
        frame_start = (i - 1) * hop_length + 1;
        frame_end = frame_start + frame_length - 1;

        % Ensure we don't go beyond the signal's length
        if frame_end > signal_length
            break;
        end

        % Extract the frame and apply the Hamming window
        frame = signal(frame_start:frame_end) .* window;

        % Compute the power spectrum using the FFT
        fft_spectrum = abs(fft(frame, 512)).^2; % https://www.mathworks.com/help/matlab/ref/fft.html
        power_spectrum = fft_spectrum(1:257); % Retain the positive frequencies

        % Apply the Mel filterbank
        mel_filterbank = mel_filterbank_matrix(num_mel_filters, 512, fs);
        mel_energies = mel_filterbank * power_spectrum;

        % Take the log of the Mel energies to reduce dynamic range
        log_mel_energies = log(mel_energies + eps); % https://www.mathworks.com/help/matlab/ref/log.html

        % Perform the DCT to get MFCCs and keep only the first coefficients
        mfcc_frame = dct(log_mel_energies); % https://www.mathworks.com/help/matlab/ref/dct.html
        mfcc_features(i, :) = mfcc_frame(1:num_mfcc_coefs)';
    end
end

function mel_filterbank = mel_filterbank_matrix(num_filters, fft_size, fs)
    % This function creates the Mel filterbank matrix for a given FFT size and sample rate.
    %
    % Inputs:
    % - num_filters: Number of Mel filters