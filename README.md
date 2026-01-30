# Album Cover Classifier (TensorFlow)

Train a computer vision model to predict either an album’s genre or release decade from its cover art. The training pipeline uses an ImageNet-pretrained backbone (default: DenseNet201) with two-stage transfer learning (frozen backbone → fine-tuning) and exports evaluation artifacts for easy review.

## Repository layout

```
src/
  config.py
  backbones.py
  data.py
  dataset.py
  model.py
  eval.py
scripts/
  get_album_art.py
  train_classifier.py
  test.sh
data/
  sample/
    rock_df.csv, pop_df.csv, ...
    rock/, pop/, ...   (images)
```

## Quickstart

The repo includes a small sample dataset under `data/sample/` and a simple test runner.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./scripts/test.sh
```

That runs `scripts/train_classifier.py` with a lightweight backbone and short epochs/steps so you can confirm the pipeline end-to-end.

## Tasks

`scripts/train_classifier.py` supports two tasks:

- `--task genre`  : predict genre (rock/pop/jazz/classical/electronic)
- `--task decade` : predict decade (e.g., 1970s, 1980s, ...)

Examples:

```
python scripts/train_classifier.py --task genre --data_root data/sample --out_root outputs_test
python scripts/train_classifier.py --task decade --data_root data/sample --out_root outputs_test
```

## Dataset format and organization

Expected layout under `--data_root`:

```
data_root/
  rock_df.csv
  pop_df.csv
  jazz_df.csv
  classical_df.csv
  electronic_df.csv
  rock/        (images)
  pop/
  jazz/
  classical/
  electronic/
```

Each `*_df.csv` must contain:

- `image_file` : filename (or relative path) of the image
- `decade`     : decade label (e.g. `1990s`)

Optional columns:
- `genre` or `genre_name` (normalized internally)
- extra metadata columns (dropped/ignored where appropriate)

During loading, the pipeline:
- concatenates per-genre CSVs into a unified DataFrame
- drops unused metadata columns (if present)
- resolves image paths (either `data_root/<image_file>` or `data_root/<genre>/<image_file>`)
- filters out missing images
- prints basic dataset summary + a genre×decade crosstab

## Input pipeline (tf.data)

`src/dataset.py` builds a `tf.data.Dataset` for training/validation/testing:

- reads bytes → `tf.image.decode_jpeg` → resize → float32 in [0, 1]
- optional `cache` (RAM or disk path)
- shuffle (training only)
- batch + prefetch
- optional `repeat` (training only)

Important behavior:
- `--repeat` is intended for smoke/demo runs with `--steps_per_epoch` and `--validation_steps`.
  If you repeat the training dataset, you should cap steps so epochs terminate.

## Configuration and CLI options

The script exposes configuration via CLI flags that map into `DataConfig`, `TrainingConfig`, and `CallbackConfig`.

### Core training options

- `--backbone` : backbone key from `src/backbones.py` (e.g. `densenet201`, `resnet50`, `efficientnetb0`, `mobilenetv3small`)
- `--image_size` : square resize (e.g. 128 for smoke tests, 224/250 for larger runs)
- `--batch_size`
- `--stage1_epochs`, `--stage2_epochs`
- `--stage1_lr`, `--stage2_lr`
- `--fine_tune_last_n` : number of backbone layers to unfreeze in stage 2

### Dataset and splitting

- `--min_examples_per_decade` : filter rare decades in the unified dataset
- `--test_size`, `--val_size`
- `--seed`

### tf.data controls (optional)

- `--cache` : enable `.cache()`
- `--cache_path <path>` : cache to a specific file (disk cache)
- `--repeat` : repeat the training dataset (use with `--steps_per_epoch`)

### Fit-loop caps (useful for demos / short runs)

- `--steps_per_epoch <int>`
- `--validation_steps <int>`

### Accelerator/runtime options (optional)

- `--set_memory_growth` : if GPUs are present, enable TF memory growth
- `--mixed_precision`   : enable mixed precision when supported

### Callbacks (stage-specific)

EarlyStopping:
- `--stage1_es_patience`, `--stage1_es_min_delta`
- `--stage2_es_patience`, `--stage2_es_min_delta`

ReduceLROnPlateau:
- `--stage1_rlr_patience`, `--stage1_rlr_min_delta`, `--stage1_rlr_factor`, `--stage1_rlr_min_lr`
- `--stage2_rlr_patience`, `--stage2_rlr_min_delta`, `--stage2_rlr_factor`, `--stage2_rlr_min_lr`

## Outputs

Artifacts are written under `--out_root` (default: `outputs/`):

```
outputs/
  models/
    <run_tag>.keras
    <run_tag>_best_stage1.keras
    <run_tag>_best_stage2.keras
  logs/
    <run_tag>_history_stage1.csv
    <run_tag>_history_stage2.csv
  metrics/
    <run_tag>_classification_report.csv
    <run_tag>_confusion_matrix.png
    <run_tag>_confusion_matrix.csv
    <run_tag>_summary.json
    <run_tag>_run_metadata.json
```

`<run_tag>` is derived from the task/backbone and a short hash of the configs, so runs are uniquely identifiable.

## Notes on backbones

Backbones are registered in `src/backbones.py`. The pipeline keeps images in `[0, 1]` during augmentation and applies the model-specific `preprocess_input` after augmentation (wrapped to match expected input scaling).

For quick test runs, `mobilenetv3small` is a good default.

## Bonus: dataset builder (MusicBrainz + Cover Art Archive)

`scripts/get_album_art.py` is included as a **bonus utility**. It is **not required** to train models from an existing dataset.

What it does:
- queries a local MusicBrainz PostgreSQL mirror for release groups tagged with target genres/years
- constructs Cover Art Archive URLs for front cover art
- downloads JPEGs into per-genre folders
- writes per-genre `*_df.csv` files linking metadata to local filenames

Requirements/assumptions:
- a working local MusicBrainz PostgreSQL mirror (not included in this repo)
- DB connection configured via environment variables:
  - `MB_DB_HOST`, `MB_DB_PORT`, `MB_DB_USER`, `MB_DB_PASSWORD`, `MB_DB_NAME`

See the MusicBrainz Docker mirror project for a typical local mirror setup:
https://github.com/metabrainz/musicbrainz-docker
