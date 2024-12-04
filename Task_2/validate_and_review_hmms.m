% Step 2 (part 2)
function validate_and_review_hmms(dev_features_file, output_hmm_file)
    % Load MFCC features from the development set
    load(dev_features_file, 'all_mfcc_features', 'file_names'); % Contains 'all_mfcc_features' and 'file_names'

    % Extract word labels from file names
    word_labels = extract_word_labels(file_names);

    % Initialize and review HMMs
    fprintf('Initializing HMMs...\n');
    num_states = 8; % Number of states
    vocab_size = 11; % Number of words in vocabulary
    num_features = 13; % Dimensionality of MFCC features
    transition_prob = 0.8; % Self-loop probability
    forward_prob = 0.2; % Forward transition probability

    hmms = cell(vocab_size, 1); % To store HMMs for each word
    for word_idx = 1:vocab_size
        % Extract features for the current word
        word_features = vertcat(all_mfcc_features{word_labels == word_idx});

        % Compute mean and variance for this word's features
        word_mean = mean(word_features, 1); % Mean specific to the current word
        word_variance = var(word_features, 0, 1); % Variance specific to the current word

        % Initialize HMM parameters
        hmm.mean_vectors = repmat(word_mean, num_states, 1); % Mean vector for each state
        hmm.variance_vectors = repmat(word_variance, num_states, 1); % Variance vector for each state
        hmm.transition_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);

        % Validate HMM dimensions
        fprintf('Validating HMM for word %d...\n', word_idx);
        assert(size(hmm.mean_vectors, 1) == num_states, 'Mean vectors row mismatch');
        assert(size(hmm.mean_vectors, 2) == num_features, 'Mean vectors column mismatch');
        assert(size(hmm.variance_vectors, 1) == num_states, 'Variance vectors row mismatch');
        assert(size(hmm.variance_vectors, 2) == num_features, 'Variance vectors column mismatch');
        assert(size(hmm.transition_matrix, 1) == num_states + 2, 'Transition matrix size mismatch');
        assert(size(hmm.transition_matrix, 2) == num_states + 2, 'Transition matrix size mismatch');

        hmms{word_idx} = hmm; % we store the validated HMM
    end

    % Save the initialized HMMs
    save(output_hmm_file, 'hmms');
    fprintf('HMMs validated and saved to %s.\n', output_hmm_file);
end

function labels = extract_word_labels(file_names)
    % Function to extract the numeric word labels from file names
    labels = zeros(length(file_names), 1);
    for i = 1:length(file_names)
        % And use a regular expression to extract the number after "_w"
        tokens = regexp(file_names{i}, '_w(\d+)_', 'tokens');
        if ~isempty(tokens) && ~isempty(tokens{1})
            labels(i) = str2double(tokens{1}{1}); % Convert the numeric word label to a number
        else
            error('File name "%s" does not match the expected format (e.g., "sp01a_w01_heed.mp3").', file_names{i});
        end
    end
end

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

% Commands to run: 
% dev_features_file = 'dev_set_mfcc_features.mat'; % From Task 1
% output_hmm_file = 'validated_hmms.mat';
% validate_and_review_hmms(dev_features_file, output_hmm_file);