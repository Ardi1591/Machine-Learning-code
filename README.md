# Machine-Learning-code

This folder contains the MATLAB code and dataset files for **Remaining Useful Life (RUL) estimation using a CNN** on the **SG001** dataset (tabular multi-channel time-series), following the workflow style of the :contentReference[oaicite:1]{index=1} Predictive Maintenance example. :contentReference[oaicite:2]{index=2}

---

## Overview

The pipeline implements an end-to-end RUL estimator:

1. **Load SG001 data** into unit-wise sequences (`localLoadDataOES`)
2. **Visualize** selected channels over time
3. **Filter near-constant channels** (low variability)
4. **Normalize** predictors with z-score statistics from training data
5. **Split train/validation by unit** (avoid leakage)
6. **Windowing**: create many samples from each run-to-failure sequence
7. **Train a causal 1-D CNN** for regression (RUL per timestep)
8. **Test with windowed prediction + stitching**
9. **Post-process predictions**: clamp, smooth, and optionally enforce monotone decreasing RUL
10. **Evaluate** with per-unit RMSE distribution and sample True vs Predicted RUL plots

This is inspired by the “Remaining Useful Life Estimation Using Convolutional Neural Network” example. :contentReference[oaicite:3]{index=3}

---

## Requirements

- MATLAB (tested with R2025b)
- **Deep Learning Toolbox** (required) :contentReference[oaicite:4]{index=4}
- **Predictive Maintenance Toolbox** (optional: only needed if you use `prognosability`) :contentReference[oaicite:5]{index=5}

---

## Files in this folder

### Data
- `train_SG001.txt` — training sequences (multiple units)
- `test_SG001.txt` — test sequences (multiple units)
- `RUL_SG001.txt` — end-of-sequence RUL per test unit (one value per unit)

### MATLAB scripts/functions
- `FD001_RUL_CNN_WebsiteStyle.m`  
  Main script (your “website-style” workflow) that trains the CNN and evaluates predictions.
  > Note: despite the name, this script is used for SG001 in your setup; you can rename it to `SG001_RUL_CNN_WebsiteStyle.m` for clarity.

- `localLoadDataOES.m`  
  Data loader: reads SG001 text files, assigns variable names, splits rows into unit-wise sequences, and generates per-timestep RUL vectors.

- `localLambdaPlot.m`  
  Utility plot: selects a test unit (random/best/worst) and plots **True vs Predicted RUL**.

---

## SG001 data format (expected)

### Predictor files (`train_SG001.txt`, `test_SG001.txt`)
A numeric matrix with columns:
1. `id` (unit identifier)
2. `timeStamp` (sequence index / cycle)
3. `Channel_1 ... Channel_N` (sensor channels)

### Response file (`RUL_SG001.txt`)
A numeric vector of length = number of unique test units:
- `RUL_SG001(i)` = RUL at the end of test unit `i` (used to reconstruct a full per-timestep RUL curve)

---

## How to run

1. Put all files in the same folder:
   - `train_SG001.txt`, `test_SG001.txt`, `RUL_SG001.txt`
   - `FD001_RUL_CNN_WebsiteStyle.m`, `localLoadDataOES.m`, `localLambdaPlot.m`

2. Open MATLAB, `cd` into this folder.

3. Run the main script:
   - Press **Run** in the editor, or type:
     ```matlab
     FD001_RUL_CNN_WebsiteStyle
     ```

4. Outputs you should see:
   - Training Progress plot (RMSE/loss)
   - RMSE histogram over test units
   - A “website-style” plot: **True vs Predicted RUL** for a random (or best/worst) test unit

---

## Key parameters option that can be tune (in the main script)

- `capRUL`  
  Maximum RUL cap (clips large early RUL values). Commonly used in RUL literature to stabilize training.
- `winLength`, `winStep`  
  Window length and stride for windowing; controls how many training samples are generated.
- `smoothK`  
  Moving-average length for smoothing predictions.
- `makeMonotonePred`  
  If `true`, forces predictions to be non-increasing over time (more physically plausible).

---

