% Dataset directory 
% Define the dataset directory of current experiment
test_dataset = 'evaluation'; % can be changed to evaluation or recorded
evaluation_dir = fullfile('dataset', test_dataset);

% Load HMM model
load('trained_hmms.mat');

% Call the function
[error_rate, word_labels] = compute_error_rate(evaluation_dir, hmms);

% Create confusion matrix and labels. Labels from testing set are defined
% manually.
confusion_matrix = confusionmat(word_labels(1,:), word_labels(2,:) );
labels = {'heed', 'hid', 'head', 'had', 'hard', 'hud', 'hod', 'hoard', 'hood', 'whod', 'heard'};

% Plot the heatmap
heatmap(labels, labels, confusion_matrix, ...
    'XLabel', 'Predicted Labels', ...
    'YLabel', 'True Labels', ...
    'Title', 'Confusion Matrix', ...
    'Colormap', flipud(hot));

% Compute and display evaluation metrics
accuracy = 1 - error_rate;
precision = diag(confusion_matrix) ./ sum(confusion_matrix, 1)'; 
recall = diag(confusion_matrix) ./ sum(confusion_matrix, 2);   
f1 = 2*(precision .* recall) ./ (precision + recall + 1e-10); % addition of 1e-10 was added to avoid dividing by zero

% Calculate scalar average for comparison reasons. This can simply be the
% mean function as each label is equally populated
average_precision = mean(precision);
average_recall = mean(recall);
average_f1 = mean(f1);
