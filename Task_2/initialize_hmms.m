% Step 2 (part 1): Initialize HMMs with word-specific parameters
function initialize_hmms(dev_features_file, output_hmm_file)
    % Load the MFCC features and file names from the dataset
    % 'all_mfcc_features' contains feature matrices for all audio files
    % 'file_names' is a list of the corresponding file names
    load(dev_features_file, 'all_mfcc_features', 'file_names'); 

    % Get the word labels for each file by extracting them from the file names
    word_labels = extract_word_labels(file_names);

    % Parameters for the HMMs
    num_states = 8; % Number of states for each HMM
    num_features = 13; % Number of MFCC features per frame
    transition_prob = 0.8; % Probability of staying in the same state
    forward_prob = 0.2; % Probability of moving to the next state

    % Number of unique words in the dataset
    vocab_size = 11; % We know there are 11 words
    hmms = cell(vocab_size, 1); % To store the HMM for each word

    % Loop through all words and initialize an HMM for each one
    for word_idx = 1:vocab_size
        % Collect all MFCC features for this specific word
        word_features = vertcat(all_mfcc_features{word_labels == word_idx});

        % Calculate the mean and variance for the features of this word
        word_mean = mean(word_features, 1); % Mean across all frames
        word_variance = var(word_features, 0, 1); % Variance across all frames

        % Create the transition matrix
        A_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);

        % Set up the HMM parameters using the HmmModel class
        initial_prob = [1; zeros(num_states + 1, 1)]; % Initial state distribution
        hmm = HmmModel(sprintf('word_%d', word_idx), A_matrix, ...
               repmat(word_mean, num_states, 1), ...
               repmat(sqrt(word_variance), num_states, 1), ...
               initial_prob);

        % Save the HMM into the cell array
        hmms{word_idx} = hmm;
    end

    % Save the HMMs to a file so we can use them later
    save(output_hmm_file, 'hmms');
    fprintf('Initialized HMMs saved to %s\n', output_hmm_file);
end

% Function to extract the word labels from the file names
function labels = extract_word_labels(file_names)
    % Input: file_names is a list of file names (e.g., 'sp01a_w01_heed.mp3')
    % Output: labels is an array of word labels (e.g., [1, 2, 3, ...])
    
    labels = zeros(length(file_names), 1); % Preallocate space for the labels
    for i = 1:length(file_names)
        % Use regex to get the word number after '_w' in the file name
        tokens = regexp(file_names{i}, '_w(\d+)_', 'tokens');
        if ~isempty(tokens) && ~isempty(tokens{1})
            labels(i) = str2double(tokens{1}{1}); % Convert the label to a number
        else
            % Throw an error if the file name format is wrong
            error('File name "%s" doesnt match the expected format (e.g., "sp01a_w01_heed.mp3").', file_names{i});
        end
    end
end

% Function to create the transition matrix for the HMM
function A = create_transition_matrix(num_states, self_prob, forward_prob)
    % Create an (N+2)x(N+2) transition matrix (including entry and exit states)
    A = zeros(num_states + 2); % Add 2 for the entry and exit states

    % Loop through each emitting state
    for i = 2:num_states+1
        A(i, i) = self_prob; % Self-loop probability
        if i < num_states+1
            A(i, i+1) = forward_prob; % Forward transition probability
        else
            % Last emitting state transitions to the exit state
            A(i, i+1) = forward_prob; 
        end
    end

    % Transitions for entry and exit states
    A(1, 2) = 1; % Entry to the first state
    A(num_states+2, num_states+2) = 1; % Exit state self-loop
end

% Commands to run the function:
% dev_features_file = 'dev_set_mfcc_features.mat'; % File containing MFCC features
% output_hmm_file = 'prototype_hmms.mat'; % File where the HMMs will be saved
% initialize_hmms(dev_features_file, output_hmm_file);
