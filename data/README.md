# Data

**Dataset:** *Garbage Images Dataset (2000/class)*, Leonardo Cofone (Kaggle user `zlatan599`), version 5,
MIT licence – <https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification>

* ~13,900 RGB JPEG images, 256 × 256 px, six classes: cardboard, glass, metal, paper, plastic, trash.
* Archive layout: `Garbage_Dataset_Classification/images/<class>/*.jpg` and `metadata.csv` (file name, label).

## How to get it

1. Kaggle website: *Download → Download dataset as zip* (`archive.zip`, ~127 MB), or
2. Python: `import kagglehub; kagglehub.dataset_download("zlatan599/garbage-dataset-classification")`

The images are not stored in this repository (size); the notebook and `scripts/run_pipeline.py` read the
zip directly.

## Split manifest

`splits.csv` (created by the pipeline) lists every image used for modelling with its class and subset
(`train`, `val`, `test`) and its near-duplicate group, so the exact experiment can be reproduced.
