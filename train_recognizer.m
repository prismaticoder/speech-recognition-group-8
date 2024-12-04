function train_recognizer(hmm_file_path, mfcc_features_file, num_epochs, dataset_dir)
% Load the MFCC features
load(mfcc_features_file, 'segmented_mfcc_features');

% Load initial HMM parameters
load(hmm_file_path, 'hmms');

% Get the vocabulary
words = fieldnames(segmented_mfcc_features);
num_words = length(words);

% Initialize storage for trained HMMs and error rates
error_rates = zeros(1, num_epochs);

% Training loop over epochs
for epoch = 1:num_epochs
    fprintf('Epoch %d/%d:\n', epoch, num_epochs);

    % Train each HMM with its corresponding data
    for i = 1:num_words
        current_word = words{i};
        word_features = segmented_mfcc_features.(current_word);

        % Train single HMM for one epoch
        hmms{i}.train(word_features, 1);
    end

    % Compute error rate using provided function
    error_rate = compute_error_rate(dataset_dir, hmms);
    error_rates(epoch) = error_rate;

    fprintf('Epoch %d complete. Error rate: %.2f%%\n', epoch, error_rate * 100);
end

% Save trained HMMs and error rates
save('trained_hmms.mat', 'hmms', 'words', 'error_rates');

% Plot error rates
figure;
plot(1:num_epochs, error_rates * 100);
xlabel('Epoch');
ylabel('Error Rate (%)');
title('Recognition Error Rate vs Training Epoch');
grid on;
savefig('error_rates.fig');
end