function [error_rate, word_labels] = compute_error_rate(dataset_dir, hmm_models)
% Validate inputs
if ~isfolder(dataset_dir)
    error('Invalid dataset directory path');
end

% if ~all(arrayfun(@(x) isa(x, 'HmmModel'), hmm_models))
%     error('All models must be instances of HmmModel class');
% end

% Initialize counters
correctWords = 0;
totalWords = 0;

% creates 2 dimensional array for word labels of actual and predicted for
% confusion matrix. first dimension is true, second is predicted
word_labels = [];

% Get all mp3 files in the directory
audio_files = dir(fullfile(dataset_dir, '*.mp3'));

% Process each audio file
for i = 1:length(audio_files)
    % Increment total words counter
    totalWords = totalWords + 1;

    % Add a progress bar
    % progress = round((i / length(audio_files)) * 100);
    % fprintf('Processing file %d/%d (%.2f%%) - %s\n', i, length(audio_files), progress, audio_files(i).name);

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
        current_hmm = hmm_models{j};
        likelihood = current_hmm.estimateLikelihood(mfcc_features);

        if likelihood > maxLikelihood
            maxLikelihood = likelihood;
            wordWithMaxLikelihood = current_hmm.word;
        end
    end

    % Check if prediction was correct
    if strcmp(wordWithMaxLikelihood, actual_word)
        correctWords = correctWords + 1;
    end

    % append max likelihood word and actual word to word labels 
    word_labels = [word_labels, [string(actual_word); string(wordWithMaxLikelihood)]];
end

% Calculate and return error rate (3 decimal places)
error_rate = round((1 - (correctWords/totalWords)) * 1000) / 1000;
end
