% Step 2 (part 1)
function initialize_hmms(dev_features_file, output_hmm_file)
    % Load MFCC features from the development set
    load(dev_features_file, 'all_mfcc_features'); % Contains 'all_mfcc_features'

    % Compute global mean and variance
    all_features = []; 
    for i = 1:length(all_mfcc_features)
        all_features = [all_features; all_mfcc_features{i}];
    end
    global_mean = mean(all_features, 1); % Mean across all frames and files
    global_variance = var(all_features, 0, 1); % Variance across all frames and files

    % Define HMM parameters
    num_states = 8; % Number of states
    num_features = 13; % Dimensionality of MFCC features
    transition_prob = 0.8; % Self-loop probability
    forward_prob = 0.2; % Forward transition probability

    % Initialize HMMs for each word
    vocab_size = 11; % Number of words in vocabulary
    hmms = cell(vocab_size, 1); % To store HMMs for each word

    for word_idx = 1:vocab_size
        hmm.mean_vectors = repmat(global_mean, num_states, 1); % Mean vector for each state
        hmm.variance_vectors = repmat(global_variance, num_states, 1); % Variance vector for each state
        hmm.transition_matrix = create_transition_matrix(num_states, transition_prob, forward_prob);
        hmms{word_idx} = hmm;
    end

    % Save the initialized HMMs
    save(output_hmm_file, 'hmms');
    fprintf('Initialized HMMs saved to %s\n', output_hmm_file);
end

function A = create_transition_matrix(num_states, self_prob, forward_prob)
    % Create an (N+2)x(N+2) transition matrix
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

% Commands to run
% dev_features_file = 'dev_set_mfcc_features.mat'; % From Task 1
% output_hmm_file = 'prototype_hmms.mat';
% initialize_hmms(dev_features_file, output_hmm_file);