% Step 2 (part 2)
function validate_and_review_hmms(dev_features_file, output_hmm_file)
    % Load MFCC features from the development set
    load(dev_features_file, 'all_mfcc_features'); % Contains 'all_mfcc_features'

    % Compute global mean and variance
    fprintf('Computing global mean and variance...\n');
    all_features = [];
    for i = 1:length(all_mfcc_features)
        all_features = [all_features; all_mfcc_features{i}];
    end
    global_mean = mean(all_features, 1); % Mean across all frames and files
    global_variance = var(all_features, 0, 1); % Variance across all frames and files

    % Display the global mean and variance
    fprintf('Global Mean (13 dimensions):\n');
    disp(global_mean);
    fprintf('Global Variance (13 dimensions):\n');
    disp(global_variance);

    % Initialize and review HMMs
    fprintf('Initializing HMMs...\n');
    num_states = 8; % Number of states
    vocab_size = 11; % Number of words in vocabulary
    num_features = 13; % Dimensionality of MFCC features
    transition_prob = 0.8; % Self-loop probability
    forward_prob = 0.2; % Forward transition probability

    hmms = cell(vocab_size, 1); % To store HMMs for each word
    for word_idx = 1:vocab_size
        hmm.mean_vectors = repmat(global_mean, num_states, 1); % Mean vector for each state
        hmm.variance_vectors = repmat(global_variance, num_states, 1); % Variance vector for each state
        hmm.transition_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);

        % Validate HMM dimensions
        fprintf('Validating HMM for word %d...\n', word_idx);
        assert(size(hmm.mean_vectors, 1) == num_states, 'Mean vectors row mismatch');
        assert(size(hmm.mean_vectors, 2) == num_features, 'Mean vectors column mismatch');
        assert(size(hmm.variance_vectors, 1) == num_states, 'Variance vectors row mismatch');
        assert(size(hmm.variance_vectors, 2) == num_features, 'Variance vectors column mismatch');
        assert(size(hmm.transition_matrix, 1) == num_states + 2, 'Transition matrix size mismatch');
        assert(size(hmm.transition_matrix, 2) == num_states + 2, 'Transition matrix size mismatch');

        hmms{word_idx} = hmm; % Store the validated HMM
    end

    % Save the initialized HMMs
    save(output_hmm_file, 'hmms');
    fprintf('HMMs validated and saved to %s.\n', output_hmm_file);
end

function A = create_transition_matrix(num_states, self_prob, forward_prob)
    % Create an (N+2)x(N+2) transition matrix (including entry and exit states)
    A = zeros(num_states + 2);
    for i = 2:num_states+1
        A(i, i) = self_prob; % Self-loop
        if i < num_states+1
            A(i, i+1) = forward_prob; % Forward transition
        end
    end
    % Add entry and exit transitions
    A(1, 2) = 1; % Entry to first state
    A(num_states+1, num_states+2) = 1; % Final state to exit
end

% Commands to run: 
% dev_features_file = 'dev_set_mfcc_features.mat'; % From Task 1
% output_hmm_file = 'validated_hmms.mat';
% validate_and_review_hmms(dev_features_file, output_hmm_file);