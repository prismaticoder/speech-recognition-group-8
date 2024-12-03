classdef HmmModel < handle
    properties
        word            % Word that this model represents (string)
        A_matrix        % Transition matrix (NxN)
        B_matrix        % Observation matrix (NxM)
        pi              % Initial state distribution (1xN)
        N               % Number of states
        M               % Number of observation symbols (e.g., MFCC dimensions)
    end

    methods
        % Constructor to initialize the model with word, A, B, pi
        function obj = HmmModel(word, A_matrix, B_matrix, pi)
            obj.word = word;
            obj.A_matrix = A_matrix;
            obj.B_matrix = B_matrix;
            obj.pi = pi;
            obj.N = size(A_matrix, 1);  % Number of states
            obj.M = size(B_matrix, 2);  % Number of observation symbols (MFCC dimensions)
        end

        % Method to estimate the likelihood of the model given MFCC features using the Viterbi algorithm
        function likelihood = estimateLikelihood(obj, mfcc_features)
            % Viterbi Algorithm Implementation to calculate the likelihood of observing mfcc_features
            T = size(mfcc_features, 1);  % Number of time steps (frames)
            N = obj.N;  % Number of states
            alpha = zeros(N, T);  % Alpha values for Viterbi algorithm
            backtrack = zeros(N, T);  % Backtracking table

            % Initialization (First frame)
            for s = 1:N
                alpha(s, 1) = log(obj.pi(s)) + log(obj.B_matrix(s, :)) * mfcc_features(1, :)';  % Observation likelihood
            end

            % Recursion (Subsequent frames)
            for t = 2:T
                for s = 1:N
                    [max_val, prev_state] = max(alpha(:, t-1) + log(obj.A_matrix(:, s)));  % Max of previous states
                    alpha(s, t) = max_val + log(obj.B_matrix(s, :)) * mfcc_features(t, :)';  % Add observation likelihood
                    backtrack(s, t) = prev_state;  % Store the best previous state
                end
            end

            % Termination (Final step)
            [max_val, final_state] = max(alpha(:, T));  % Find the most likely final state
            likelihood = max_val;  % The maximum likelihood of the entire observation sequence

            % Optional: backtracking to get the best state sequence if needed:
            % state_sequence = zeros(1, T);
            % state_sequence(T) = final_state;
            % for t = T-1:-1:1
            %     state_sequence(t) = backtrack(state_sequence(t+1), t+1);
            % end
        end
    end
end
