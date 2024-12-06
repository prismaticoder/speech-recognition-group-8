% Step 2 (part 1): Initialize HMMs with word-specific parameters
function initialize_hmms(dev_features_file, output_hmm_file)
    % This function initializes Hidden Markov Models (HMMs) for each unique word
    % in the dataset. It uses word-specific parameters calculated from MFCC
    % features to define the HMMs.
    %
    % Inputs:
    % - dev_features_file: The .mat file containing MFCC features and file names.
    % - output_hmm_file: The .mat file where the initialized HMMs will be saved.

    % Load the extracted MFCC features and corresponding file names
    load(dev_features_file, 'all_mfcc_features', 'file_names'); % https://www.mathworks.com/help/matlab/ref/load.html

    % Get the word labels by parsing the file names
    word_labels = extract_word_labels(file_names); 

    % Define parameters for the HMM
    num_states = 8; % Number of states excluding entry/exit states
    num_features = 13; % MFCC features per frame
    transition_prob = 0.8; % Probability of staying in the current state
    forward_prob = 0.2; % Probability of moving to the next state

    % Known vocabulary size (number of unique words in the dataset)
    vocab_size = 11; % Assumes we know the dataset has 11 unique words
    hmms = cell(vocab_size, 1); % Preallocate a cell array to store the HMMs

    % Loop through each unique word to initialize its HMM
    for word_idx = 1:vocab_size
        % Extract all MFCC features corresponding to this word
        word_features = vertcat(all_mfcc_features{word_labels == word_idx}); % https://www.mathworks.com/help/matlab/ref/vertcat.html

        % Retrieve the word name for this HMM using one of its file names
        word_files = file_names(word_labels == word_idx);
        word_name = extract_word_from_filename(word_files{1}); 

        % Calculate the mean and variance of the word's MFCC features
        word_mean = mean(word_features, 1); % https://www.mathworks.com/help/matlab/ref/mean.html
        word_variance = max(var(word_features, 0, 1), 1e-4); % https://www.mathworks.com/help/matlab/ref/var.html
        % Apply a floor value (1e-4) to avoid numerical issues with near-zero variance

        % Create a transition matrix for the HMM
        A_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);

        % Define the HMM parameters
        initial_prob = [1; zeros(num_states, 1)]; % Initial probability (entry state is 1)
        hmm = HmmModel(word_name, A_matrix, ...
               repmat(word_mean, num_states + 1, 1), ... % Replicate mean for all states https://www.mathworks.com/help/matlab/ref/repmat.html
               repmat(sqrt(word_variance), num_states + 1, 1), ... % Variance matrix
               initial_prob);

        % Store the HMM in the cell array
        hmms{word_idx} = hmm;
    end

    % Save the HMMs to a file for later use
    save(output_hmm_file, 'hmms'); % https://www.mathworks.com/help/matlab/ref/save.html
    fprintf('Initialized HMMs saved to %s\n', output_hmm_file); % https://www.mathworks.com/help/matlab/ref/fprintf.html
end

% Function to extract word labels from file names
function labels = extract_word_labels(file_names)
    % This function generates numeric labels for words based on file names.
    %
    % Inputs:
    % - file_names: Cell array of file names.
    %
    % Outputs:
    % - labels: Numeric labels corresponding to word indices in the dataset.

    labels = zeros(length(file_names), 1); % Preallocate label array https://www.mathworks.com/help/matlab/ref/zeros.html
    for i = 1:length(file_names)
        tokens = regexp(file_names{i}, '_w(\d+)_', 'tokens'); % https://www.mathworks.com/help/matlab/ref/regexp.html
        if ~isempty(tokens) && ~isempty(tokens{1})
            labels(i) = str2double(tokens{1}{1}); % Convert label to numeric https://www.mathworks.com/help/matlab/ref/str2double.html
        else
            error('File name "%s" doesn''t match the expected format.', file_names{i}); % https://www.mathworks.com/help/matlab/ref/error.html
        end
    end
end

% Function to extract the word name from a file name
function word = extract_word_from_filename(file_name)
    % This function extracts the word name from a file name based on its format.
    %
    % Input:
    % - file_name: A single file name as a string.
    %
    % Output:
    % - word: Extracted word name as a string.

    tokens = regexp(file_name, '_w\d+_(\w+)\.', 'tokens'); % Extract word after '_w<index>_' https://www.mathworks.com/help/matlab/ref/regexp.html
    if ~isempty(tokens) && ~isempty(tokens{1})
        word = tokens{1}{1};
    else
        error('File name "%s" doesn''t match the expected format.', file_name);
    end
end

% Function to create a transition matrix for the HMM
function A = create_transition_matrix(num_states, self_prob, forward_prob)
    % This function creates a transition matrix for an HMM with entry and exit states.
    %
    % Inputs:
    % - num_states: Number of emitting states.
    % - self_prob: Probability of staying in the same state.
    % - forward_prob: Probability of moving to the next state.
    %
    % Output:
    % - A: Transition matrix of size (num_states + 1) x (num_states + 1).

    A = zeros(num_states + 1); % Preallocate the matrix https://www.mathworks.com/help/matlab/ref/zeros.html

    for i = 1:num_states
        A(i, i) = self_prob; % Self-transition
        if i < num_states
            A(i, i + 1) = forward_prob; % Forward transition
        else
            A(i, i + 1) = forward_prob; % Transition from last emitting state to exit
        end
    end

    A(num_states + 1, num_states + 1) = 1; % Exit state only transitions to itself
end