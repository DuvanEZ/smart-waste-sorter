"""Smart Waste Sorter - image classification of recyclable waste (INFO813 project).

Modules
-------
config          class names, image size, hyper-parameters
data            dataset discovery, file index, stratified group split, transforms
image_analysis  per-image statistics, perceptual hashing, duplicate detection
features        hand-crafted features (colour histogram, HOG, LBP) for baselines
models          EfficientNet-B0 (transfer learning) and a small CNN
train           two-phase training loop with early stopping
evaluate        metrics, bootstrap confidence intervals, calibration, robustness
explain         Grad-CAM heat-maps
export          ONNX export for the application
plots           every figure of the report
pipeline        the steps of the report, end to end
"""
__version__ = "1.0.0"
