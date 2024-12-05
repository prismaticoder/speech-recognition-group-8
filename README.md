# Isolated Word Recognizer Using Hidden Markov Models

This project implements an isolated word recognizer using Hidden Markov Models (HMMs). The system is designed to recognize a vocabulary of 11 words based on Mel-frequency cepstral coefficients (MFCCs) extracted from audio recordings. The project demonstrates feature extraction, model initialization, training, and evaluation using techniques such as the Baum-Welch algorithm and the Viterbi algorithm.

## Vocabulary

The vocabulary includes the following 11 words:

1. `heed`
2. `hid`
3. `head`
4. `had`
5. `hard`
6. `hud`
7. `hod`
8. `hoard`
9. `hood`
10. `who’d`
11. `heard`

## Contributors

The project was developed as part of a group assignment for the Speech and Audio Processing module (EEEM030). The contributors to this project are:

- [@ArtemKar123](https://github.com/ArtemKar123)
- [@prismaticoder](https://github.com/prismaticoder)
- [@SamWattsJ](https://github.com/SamWattsJ)
- [@ZohraB612](https://github.com/ZohraB612)
- [@munavarm74](https://github.com/munavarm74)

## Module

This project is part of the coursework for the module:
**EEEM030: Speech and Audio Processing**

## Running the Project

### Prerequisites

To run the project, ensure you have the following installed:

- MATLAB (R2021a or later)
- Required toolboxes:
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

### Steps to Run the Script

#### Prerequisites

Ensure you have the following installed on your system:

- MATLAB (R2021a or later)
- Required toolboxes:
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

#### Training the Recognizer

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/prismaticoder/speech-recognition-group-8.git
   cd speech-recognition-group-8
   ```
2. Open MATLAB and navigate to the project directory:
   ```bash
   cd speech-recognition-group-8
   ```
3. Run the `train.m` script to train the recognizer on the development dataset:

   ```matlab
   train
   ```

   The train.m script performs the following steps:

   - Loads the development dataset from the `dataset/development/` directory.
   - Extracts MFCC features for each audio file.
   - Initializes and trains the HMMs for the vocabulary words using the Baum-Welch algorithm.
   - Evaluates the Word Error Rate (WER) on the development set after each epoch.

4. The training process outputs:

   - A log of training progress, including WER for each epoch.
   - The final trained HMM parameters saved to the `trained_hmms.mat` file.
   - A graph of WER vs. training epochs saved as `error_rates.fig`.

#### Evaluating the Recognizer

Once training is complete, run the `test.m` script to evaluate the recognizer on the evaluation dataset:

```matlab
test
```

The `test.m` script performs the following steps:

- Loads the evaluation dataset from the `dataset/evaluation/` directory.
- Extracts MFCC features for each audio file.
- Loads the trained HMMs stored in the `trained_hmms.mat` file.
- Computes the recognition accuracy and generates a confusion matrix.
  The evaluation process outputs:

The recognition accuracy and Word Error Rate (WER) on the evaluation set.
A confusion matrix saved as confusion_matrix.fig.

### Customizing Input

To add your own audio data:

- Place training data in the `dataset/development/` directory.
- Place evaluation data in the `dataset/evaluation/` directory.
- Ensure each file is named in the format `index_word.mp3` (e.g., 1_heed.wav for "heed"). Also ensure it's for a word in the vocabulary for the system.

## Results

The recognizer achieves a minimum Word Error Rate (WER) of 17.3% at the 6th and 7th training epochs. The confusion matrix highlights common misclassifications between acoustically similar words.

## License

This project is for academic purposes and may not be used for commercial applications without permission.
