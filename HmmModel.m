classdef HmmModel < handle
    properties
        word           % Word that this model represents (string)
        logA           % Transition matrix (NxN) in log space
        mu             % Mean vectors for each state (NxM)
        sigma          % Covariance matrices for each state (NxM)
        N              % Number of states
        M              % Number of observation symbols (e.g., MFCC dimensions)
        logPi          % Initial state distribution (Nx1) in log space
        logEta         % Exit state distribution (Nx1) in log space
    end

    methods
        % Constructor to initialize the model with word, A, mu, sigma
        function obj = HmmModel(word, A_matrix, mu, sigma, pi, eta)
            obj.word = word;
            obj.logA = log(A_matrix);
            obj.mu = mu;
            obj.sigma = sigma;
            obj.N = size(A_matrix, 1);  % Number of states
            obj.M = size(mu, 2);  % Number of observation symbols (MFCC dimensions)
            obj.logPi = log(pi);
            obj.logEta = log(eta);
        end

        function logProb = computeLogEmission(obj, x, state)
            % Compute the log of the emission probability for a single observation and state
            % Inputs:
            %   x - Single observation vector (1xD)
            %   mu - Mean vector for a specific state (1xD)
            %   sigma - Standard deviation vector for a specific state (1xD)
            % Outputs:
            %   logProb - Logarithm of the emission probability (scalar)
            stateMu = obj.mu(state, :);
            stateSigma = obj.sigma(state, :);
            D = length(x);
            normalization = -0.5 * D * log(2 * pi) - sum(log(stateSigma)); % Gaussian normalization
            exponent = -0.5 * sum(((x - stateMu) ./ stateSigma).^2); % Exponent term
            logProb = normalization + exponent; % Log probability
        end

        % Method to estimate likelihood using Viterbi
        function likelihood = estimateLikelihood(obj, mfcc_features)
            % Debug prints
            fprintf('Size of A_matrix: %dx%d\n', size(obj.logA));
            fprintf('Number of states (N): %d\n', obj.N);
            fprintf('Size of mfcc_features: %dx%d\n', size(mfcc_features));

            T = size(mfcc_features, 1);
            N = obj.N;

            % Initialize viterbi and backpointer matrices
            viterbi = -inf(N, T);
            backpointer = zeros(N, T);

            % Initialization (t=1) using pi
            for s = 1:N
                viterbi(s, 1) = obj.logPi(s) + ...
                    obj.computeLogEmission(mfcc_features(1,:), s);
                    % log(obj.compute_observation_probability(mfcc_features(1,:), s));
            end

            % Recursion using A_matrix directly
            for t = 2:T
                for s = 1:N
                    prev_probs = zeros(N, 1);
                    for prev_s = 1:N
                        prev_probs(prev_s) = viterbi(prev_s, t-1) + obj.logA(prev_s, s);
                    end
                    [max_val, best_prev_state] = max(prev_probs);
                    viterbi(s, t) = max_val + ...
                        obj.computeLogEmission(mfcc_features(t,:), s);
                        % log(obj.compute_observation_probability(mfcc_features(t,:), s));
                    backpointer(s, t) = best_prev_state;
                end
            end

            % Termination using eta
            final_probs = zeros(N, 1);
            for s = 1:N
                final_probs(s) = viterbi(s, T) + obj.logEta(s);
            end
            [likelihood, ~] = max(final_probs);
        end
    end
end
