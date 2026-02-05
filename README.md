# SemEval Task 2: Predicting Valence-Arousal Score for Longitudinal Texts


## Environment set up
1. Create a new python virtual environment on your machine and activate

2. Install PyTorch using the official instruction: https://pytorch.org/

3. Run ```pip install -r requirements.txt``` to install the rest of the libraries

## Running the inference

### Subtask 1
```[Bash]
python BiLSTM_predict.py --valence_model models/subtask1_val.pt --arousal_model models/subtask1_aro.pt --input .\data\test_subtask1.csv --output pred_subtask1.csv
```
### Subtask 2a
```[Bash]
python tft_subtask2a.py
```
### Subtask 2b
```[Bash]
python lm_rnn_subtask2b.py
```
