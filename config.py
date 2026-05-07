# 00_data/config.py
from pathlib import Path

FS = 16000
BS = 1024
ALPHA_MIN = 0.05
ALPHA_MAX = 0.95
SILENCE_THRESH = 0.025
N_TEST_TLKR = 20
N_TEST_UTT_PER_TLKR = 5
TRAIN_HR = 4.0
VAL_HR = 0.5

LIBRISPEECH_ROOT = Path("/Volumes/LPM03 storage/Datasets/Audio/LibriSpeech")
TRAIN_OUT = Path("/Volumes/LPM03 storage/Datasets/Audio/ProjectTrainValTest/speech-declipping/train")
VAL_OUT = Path("/Volumes/LPM03 storage/Datasets/Audio/ProjectTrainValTest/speech-declipping/val")
TEST_OUT = Path("/Volumes/LPM03 storage/Datasets/Audio/ProjectTrainValTest/speech-declipping/test")
STUDY_OUT = Path("/Volumes/LPM03 storage/Datasets/Audio/ProjectTrainValTest/speech-declipping/train_study")
FINAL_OUT = Path("/Volumes/LPM03 storage/Datasets/Audio/ProjectTrainValTest/speech-declipping/train_final")