classdef HmmModel < handle
    properties
        word            % Word that this model represents (string)
        A_matrix        % Transition matrix (NxN)
        mu             % Mean vectors for each state (NxM)
        sigma          % Covariance matrices for each state (MxMxN)
        N              % Number of states
        M              % Number of observation symbols (e.g., MFCC dimensions)
        pi             % Initial state distribution (Nx1)
        eta            % Exit state distribution (Nx1)
    end

    methods
        % Constructor to initialize the model with word, A, mu, sigma
        function obj = HmmModel(word, A_matrix, mu, sigma, pi, eta)
            obj.word = word;
            obj.A_matrix = A_matrix;
            obj.mu = mu;
            obj.sigma = sigma;
            obj.N = size(A_matrix, 1);  % Number of states
            obj.M = size(mu, 2);  % Number of observation symbols (MFCC dimensions)
            obj.pi = pi;
            obj.eta = eta;
        end

        % Method to compute observation probability using multivariate Gaussian
        function prob = compute_observation_probability(obj, feature_vector, state)
            % Debug prints
            fprintf('Size of mu: %dx%d\n', size(obj.mu));
            fprintf('Attempting to access state: %d\n', state);

            mu_state = obj.mu(state, :);
            sigma_state = obj.sigma(:, :, state);

            % Compute multivariate Gaussian PDF
            diff = feature_vector - mu_state;
            exponent = -0.5 * diff * (sigma_state \ diff');
            normalizer = sqrt((2*pi)^obj.M * det(sigma_state));
            prob = exp(exponent) / normalizer;
        end

        % Method to estimate likelihood using Viterbi
        function likelihood = estimateLikelihood(obj, mfcc_features)
            % Debug prints
            fprintf('Size of A_matrix: %dx%d\n', size(obj.A_matrix));
            fprintf('Number of states (N): %d\n', obj.N);
            fprintf('Size of mfcc_features: %dx%d\n', size(mfcc_features));

            T = size(mfcc_features, 1);
            N = obj.N;

            % Initialize viterbi and backpointer matrices
            viterbi = -inf(N, T);
            backpointer = zeros(N, T);

            % Initialization (t=1) using pi
            for s = 1:N
                viterbi(s, 1) = log(obj.pi(s)) + ...
                    log(obj.compute_observation_probability(mfcc_features(1,:), s));
            end

            % Recursion using A_matrix directly
            for t = 2:T
                for s = 1:N
                    prev_probs = zeros(N, 1);
                    for prev_s = 1:N
                        prev_probs(prev_s) = viterbi(prev_s, t-1) + log(obj.A_matrix(prev_s, s));
                    end
                    [max_val, best_prev_state] = max(prev_probs);
                    viterbi(s, t) = max_val + ...
                        log(obj.compute_observation_probability(mfcc_features(t,:), s));
                    backpointer(s, t) = best_prev_state;
                end
            end

            % Termination using eta
            final_probs = zeros(N, 1);
            for s = 1:N
                final_probs(s) = viterbi(s, T) + log(obj.eta(s));
            end
            [likelihood, ~] = max(final_probs);
        end
    end
end
