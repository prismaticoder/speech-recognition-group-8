classdef HmmModel < handle
    properties
        word           % Word that this model represents (string)
        logA           % Transition matrix (NxN) in log space
        mu             % Mean vectors for each state (NxM)
        sigma          % Covariance matrices for each state (NxM)
        N              % Number of states
        M              % Number of observation symbols (e.g., MFCC dimensions)
        logPi          % Initial state distribution (Nx1) in log space
    end
    
    methods (Static)
        function logGamma = calculateOccupationLikelihoods(logAlpha, logBeta)
            % Calculate occupation likelihoods (gamma) in the log domain
            % Inputs:
            %   logAlpha - Logarithmic forward probabilities (NxT)
            %   logBeta  - Logarithmic backward probabilities (NxT)
            % Outputs:
            %   logGamma - Logarithmic occupation likelihoods (NxT)
        
            [N, T] = size(logAlpha);
        
            logGamma = -inf(N, T); % Use -inf for log(0)
        
            for t = 1:T
                logNorm = -inf;
                for i = 1:N
                    logNorm = logSumExp(logNorm, logAlpha(i, t) + logBeta(i, t));
                end
        
                for i = 1:N
                    logGamma(i, t) = logAlpha(i, t) + logBeta(i, t) - logNorm;
                end
            end
        end
    end

    methods
        % Constructor to initialize the model with word, A, mu, sigma
        function obj = HmmModel(word, A_matrix, mu, sigma, pi)
            obj.word = word;
            obj.logA = log(A_matrix);
            obj.mu = mu;
            obj.sigma = sigma;
            obj.N = size(A_matrix, 1);  % Number of states
            obj.M = size(mu, 2);  % Number of observation symbols (MFCC dimensions)
            obj.logPi = log(pi);
        end

        function logProb = computeLogEmission(model, x, state)
            % Compute the log of the emission probability for a single observation and state
            % Inputs:
            %   model — HmmModel
            %   x - Single observation vector (1xM)
            % Outputs:
            %   logProb - Logarithm of the emission probability (scalar)
            stateMu = model.mu(state, :);
            stateSigma = model.sigma(state, :);
            M = length(x);
            normalization = -0.5 * M * log(2 * pi) - sum(log(stateSigma)); % Gaussian normalization
            exponent = -0.5 * sum(((x - stateMu) ./ stateSigma).^2); % Exponent term
            logProb = normalization + exponent; % Log probability
        end

        % ===== TRAINING =====
        function train(model, observations, numIterations)
            % Train an HMM using the Baum-Welch algorithm in log-space
            % Inputs:
            %   model — HmmModel
            %   observations - Sequence of observed vectors (TxM)
            %   numIterations - Number of training iterations
        
            % Store log-likelihoods for monitoring convergence
            logLikelihoods = zeros(1, numIterations);
           
            for iter = 1:numIterations
                fprintf('Iteration %d:\n', iter);
        
                logAlpha = model.forwardProcedureLog(observations);
                logBeta = model.backwardProcedureLog(observations);
                logGamma = model.calculateOccupationLikelihoods(logAlpha, logBeta);
                logXi = model.calculateTransitionLikelihoods(logAlpha, logBeta, observations);
                        
                [logA, mu, sigma] = model.reestimateModelLog(logXi, logGamma, observations);
                logA(isinf(logA)) = -inf;
        
                model.logA = logA;
                model.mu = mu;
                model.sigma = sigma;
        
                logLikelihood = model.estimateLikelihood(observations);
                logLikelihoods(iter) = logLikelihood;
                fprintf('Log-Likelihood after iteration %d: %.6f\n', iter, logLikelihood);
        
                if iter > 1
                    improvement = abs(logLikelihood - logLikelihoods(iter - 1)) / abs(logLikelihoods(iter - 1));
                    fprintf('Improvement: %.6f\n', improvement);
                    if improvement < 1e-4
                        fprintf('Convergence achieved at iteration %d.\n', iter);
                        break;
                    end
                end
            end
        end

        function logAlpha = forwardProcedureLog(model, observations)
            % Logarithmic Forward Procedure for HMM
            % Inputs:
            %   model — HmmModel
            %   observations - Observation sequence (TxM)
            % Outputs:
            %   logAlpha - Logarithmic forward probabilities (NxT)
            
            [T, ~] = size(observations);
            N = size(model.logA, 1);
            
            logAlpha = -inf(N, T); % Use -inf for log(0)
            
            for i = 1:N
                logEmmision = model.computeLogEmission(observations(1, :), i);
                logAlpha(i, 1) = model.logPi(i) + logEmmision;
            end
           
            for t = 2:T
                for j = 1:N
                    logSum = -inf;
                    for i = 1:N
                        logSum = logSumExp(logSum, logAlpha(i, t-1) + model.logA(i, j));
                    end
                    logEmmision = model.computeLogEmission(observations(t, :), j);
                    logAlpha(j, t) = logSum + logEmmision;
                end
            end
        end
        
        function logBeta = backwardProcedureLog(model, observations)
            % Logarithmic Backward Procedure for HMM
            % Inputs:
            %   model — HmmModel
            %   observations - Observation sequence (TxM)
            % Outputs:
            %   logBeta - Logarithmic backward probabilities (NxT)
        
            [T, ~] = size(observations);
            N = size(model.logA, 1);
        
            logBeta = -inf(N, T); % Use -inf for log(0)
        
            logBeta(:, T) = 0; % log(1) = 0 for the final step
           
            for t = T-1:-1:1
                for i = 1:N
                    logSum = -inf;
                    for j = 1:N
                        logEmmision = model.computeLogEmission(observations(t+1, :), j);
                        logSum = logSumExp(logSum, model.logA(i, j) + logEmmision + logBeta(j, t+1));
                    end
                    logBeta(i, t) = logSum;
                end
            end
        end
        
        function logXi = calculateTransitionLikelihoods(model, logAlpha, logBeta, observations)
            % Calculate transition likelihoods (xi) in the log domain
            % Inputs:
            %   model — HmmModel
            %   logAlpha - Logarithmic forward probabilities (NxT)
            %   logBeta - Logarithmic backward probabilities (NxT)
            %   observations - Observation sequence (TxM)
            % Outputs:
            %   logXi - Logarithmic transition likelihoods (NxNx(T-1))
        
            [N, T] = size(logAlpha);
        
            logXi = -inf(N, N, T-1);
        
            for t = 1:T-1
                logNorm = -inf; % Start with log(0)
                for i = 1:N
                    for j = 1:N
                        logEmmision = model.computeLogEmission(observations(t+1, :), j);
                        logNorm = logSumExp(logNorm, ...
                            logAlpha(i, t) + model.logA(i, j) + logEmmision + logBeta(j, t+1));
                    end
                end
        
                for i = 1:N
                    for j = 1:N
                        logEmmision = model.computeLogEmission(observations(t+1, :), j);
                        logXi(i, j, t) = logAlpha(i, t) + model.logA(i, j) + logEmmision + logBeta(j, t+1) - logNorm;
                    end
                end
            end
        end
        
        function [logA_new, mu_new, sigma_new] = reestimateModelLog(model, logXi, logGamma, observations)
            % Re-estimate transition and emission parameters for the HMM in log space
            % Inputs:
            %   model — HmmModel
            %   logXi - Transition likelihoods (NxNx(T-1)) in log domain
            %   logGamma - State occupation likelihoods (NxT) in log domain
            %   observations - Observation sequence (TxM)
            % Outputs:
            %   logA_new - Updated log transition probabilities (NxN)
            %   mu_new - Updated mean vectors for emissions (NxM)
            %   sigma_new - Updated standard deviations for emissions (NxM)
        
            [N, T] = size(logGamma);
            M = size(observations, 2);
        
            logA_new = -inf(N, N);
            mu_new = zeros(N, M);
            sigma_new = zeros(N, M);
        
            % Re-estimate Transition Matrix (A)
            for i = 1:N
                for j = 1:N
                    logNumerator = -inf;
                    for t = 1:(T-1)
                        logNumerator = logSumExp(logNumerator, logXi(i, j, t));
                    end
        
                    logDenominator = -inf;
                    for t = 1:(T-1)
                        logDenominator = logSumExp(logDenominator, logGamma(i, t));
                    end
        
                    if isfinite(logDenominator) % Avoid -inf - (-inf)
                        logA_new(i, j) = logNumerator - logDenominator;
                    else
                        logA_new(i, j) = -inf; % If denominator is zero, set transition probability to zero
                    end
                end
            end
        
            % Re-estimate Emission Parameters (mu and sigma)
            gamma = exp(logGamma);
            for i = 1:N
                gamma_sum = sum(gamma(i, :)); % Total gamma for state i
            
                if gamma_sum == 0
                    % If state is unused, retain previous values or initialize defaults
                    mu_new(i, :) = model.mu(i, :);
                    sigma_new(i, :) = model.sigma(i, :);
                else
                    mu_new(i, :) = (gamma(i, :) * observations) / gamma_sum;
            
                    diff = observations - mu_new(i, :);
                    sigma_new(i, :) = sqrt((gamma(i, :) * (diff.^2)) / gamma_sum);
                    
                    % force sigma to be larger than epsilon for stability
                    if any(sigma_new(i, :) < 1e-4)
                        sigma_new(i, :) = model.sigma(i, :); % alternatively you can set sigma_new(i, :) = zeros(13, 1) + 1e-4;
                    end
                end
            end
        end

        % ===== \TRAINING =====

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
                viterbi(s, 1) = obj.logPi(s) + obj.computeLogEmission(mfcc_features(1,:), s);
            end

            % Recursion using A_matrix directly
            for t = 2:T
                for s = 1:N
                    prev_probs = zeros(N, 1);
                    for prev_s = 1:N
                        prev_probs(prev_s) = viterbi(prev_s, t-1) + obj.logA(prev_s, s);
                    end
                    [max_val, best_prev_state] = max(prev_probs);
                    viterbi(s, t) = max_val + obj.computeLogEmission(mfcc_features(t,:), s);
                    backpointer(s, t) = best_prev_state;
                end
            end

            [likelihood, ~] = max(viterbi(:, T));
        end
    end
end
