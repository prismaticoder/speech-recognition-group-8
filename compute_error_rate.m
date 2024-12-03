function error_rate = compute_error_rate(dataset_dir, hmm_models)
% Validate inputs
if ~isfolder(dataset_dir)
    error('Invalid dataset directory path');
end

if ~all(arrayfun(@(x) isa(x, 'HmmModel'), hmm_models))
    error('All models must be instances of HmmModel class');
end

% Initialize counters
correctWords = 0;
totalWords = 0;

% Get all mp3 files in the directory
audio_files = dir(fullfile(dataset_dir, '*.mp3'));

% Process each audio file
for i = 1:length(audio_files)
    % Increment total words counter
    totalWords = totalWords + 1;

    % Get file path and actual word
    file_path = fullfile(audio_files(i).folder, audio_files(i).name);
    actual_word = extract_word_from_filename(audio_files(i).name);

    % Extract MFCC features
    mfcc_features = extract_mfcc(file_path);

    % Find the most likely word
    maxLikelihood = -inf;
    wordWithMaxLikelihood = '';

    % Test against each HMM model
    for j = 1:length(hmm_models)
        likelihood = hmm_models(j).estimateLikelihood(mfcc_features);

        if likelihood > maxLikelihood
            maxLikelihood = likelihood;
            wordWithMaxLikelihood = hmm_models(j).word;
        end
    end

    % Check if prediction was correct
    if strcmp(wordWithMaxLikelihood, actual_word)
        correctWords = correctWords + 1;
    end
end

% Calculate and return error rate (3 decimal places)
error_rate = round((1 - correctWords/totalWords) * 1000) / 1000;
end

function word = extract_word_from_filename(filename)
% Extract word from filename format: sp01a_w03_head.mp3
parts = split(filename, '_');
word = extractBefore(parts{3}, '.mp3');
end
