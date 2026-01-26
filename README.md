# Album Cover Decade Classification (TensorFlow)

Train a computer vision model to predict an album’s release decade from its cover art. This project uses DenseNet201 (ImageNet pretrained) with two-stage training (frozen backbone then fine-tuning) and exports evaluation artifacts for easy review.

## Highlights
- DenseNet201 transfer learning plus fine-tuning
- Robust dataset cleaning (drops unused columns, removes missing images)
- Stratified train/val/test splits by decade
- Outputs: trained model, training logs, classification report, confusion matrix

## Dataset format

Expected folder layout:

```

data_root/
rock_df.csv
pop_df.csv
jazz_df.csv
classical_df.csv
electronic_df.csv
rock/ <image files...>
pop/
jazz/
classical/
electronic/

```

Each *_df.csv must contain:
- image_file (filename within the genre folder)
- decade (label, e.g. 1970s, 1980s)

Optional columns are ignored/dropped automatically.

## Installation

```

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

requirements.txt:
- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Training

Train a single genre:

```

python train_decade_classifier.py --data_root ./data_root --genre rock

```

Train all genres:

```

python train_decade_classifier.py --data_root ./data_root --genre all

```

Useful options:

```

python train_decade_classifier.py 
--data_root ./data_root 
--genre rock 
--batch_size 32 
--stage1_epochs 25 
--stage2_epochs 25 
--min_examples_per_decade 2500 
--fine_tune_last_n 10

```

## Outputs

Artifacts are written to:

```

outputs/
models/
DenseNet201_<genre>*DecadeClassifier.keras
metrics/
DenseNet201*<genre>*classification_report.csv
DenseNet201*<genre>*confusion_matrix.png
logs/
DenseNet201*<genre>*history_stage1.csv
DenseNet201*<genre>_history_stage2.csv

