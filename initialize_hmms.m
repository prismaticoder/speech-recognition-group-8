% Step 2 (part 1): Initialize HMMs with word-specific parameters
function initialize_hmms(dev_features_file, output_hmm_file)
    % Load the MFCC features and file names from the dataset
    load(dev_features_file, 'all_mfcc_features', 'file_names'); 

    % Get the word labels for each file by extracting them from the file names
    word_labels = extract_word_labels(file_names);

    % Parameters for the HMMs
    num_states = 8; % Number of emitting states for each HMM
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

        % Extract the word name for the HMM
        word_files = file_names(word_labels == word_idx);
        word_name = extract_word_from_filename(word_files{1}); % Get the actual word name

        % Calculate the mean and variance for the features of this word
        word_mean = mean(word_features, 1); % Mean across all frames
        word_variance = max(var(word_features, 0, 1), 1e-4); % Variance with floor for stability

        % Create the 9x9 transition matrix
        A_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);

        % Set up the HMM parameters using the HmmModel class
        initial_prob = [1; zeros(num_states, 1)]; % Initial state distribution (9 states)
        hmm = HmmModel(word_name, A_matrix, ...
               repmat(word_mean, num_states + 1, 1), ... % 9x13 mean matrix
               repmat(sqrt(word_variance), num_states + 1, 1), ... % 9x13 variance matrix
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
    labels = zeros(length(file_names), 1); % Preallocate space for the labels
    for i = 1:length(file_names)
        tokens = regexp(file_names{i}, '_w(\d+)_', 'tokens');
        if ~isempty(tokens) && ~isempty(tokens{1})
            labels(i) = str2double(tokens{1}{1}); % Convert the label to a number
        else
            error('File name "%s" doesn''t match the expected format.', file_names{i});
        end
    end
end

% Function to extract the word from the file name
function word = extract_word_from_filename(file_name)
    tokens = regexp(file_name, '_w\d+_(\w+)\.', 'tokens'); % Extract word after '_w<index>_'
    if ~isempty(tokens) && ~isempty(tokens{1})
        word = tokens{1}{1};
    else
        error('File name "%s" doesn''t match the expected format.', file_name);
    end
end

% Function to create a 9x9 transition matrix
function A = create_transition_matrix(num_states, self_prob, forward_prob)
    % Add an entry and exit state to create a 9x9 matrix
    A = zeros(num_states + 1);

    % Fill in the transition probabilities
    for i = 1:num_states
        A(i, i) = self_prob; % Self-loop probability
        if i < num_states
            A(i, i + 1) = forward_prob; % Forward transition probability
        else
            A(i, i + 1) = forward_prob; % Last emitting state transitions to exit
        end
    end

    % Ensure the exit state transitions only to itself
    A(num_states + 1, num_states + 1) = 1; % Exit state self-loop
end

% Commands to run the function:
% dev_features_filce = 'dev_set_mfcc_features.mat'; % File containing MFCC features
% output_hmm_file = 'prototype_hmms.mat'; % File where the HMMs will be saved
% initialize_hmms(dev_features_file, output_hmm_file);
