# train_nnue.py — Configuration Settings Overview

Generated: 12 March 2026

---

## 1. Data

| Argument           | Type    | Default                        | Description                                               |
|--------------------|---------|--------------------------------|-----------------------------------------------------------|
| `--data`           | string  | `assets/training_data.bin`     | Primary training data file                                |
| `--extra-data`     | str+float | *(none)*                     | Additional dataset with sampling ratio (repeatable flag)  |
| `--max-positions`  | int     | `0` (all)                      | Limit training to first N positions; 0 = use all          |
| `--chunk-size`     | int     | `200,000`                      | Positions per chunk when streaming from disk              |

---

## 2. Model / Weights

| Argument          | Type   | Default                      | Description                                                  |
|-------------------|--------|------------------------------|--------------------------------------------------------------|
| `--load-weights`  | string | *(none)*                     | Load weights from file; resets optimizer to avoid shape mismatch |
| `--fresh`         | flag   | off                          | Start with random weights; ignore any existing weights file  |
| `--output`        | string | `assets/nnue_weights.bin`    | Destination path for saved weights                           |

---

## 3. Training Loop

| Argument        | Type | Default | Description                                            |
|-----------------|------|---------|--------------------------------------------------------|
| `--epochs`      | int  | `500`   | Total number of training epochs                        |
| `--batch-size`  | int  | `8192`  | Positions per mini-batch                               |
| `--early-stop`  | int  | `15`    | Stop after N consecutive epochs without validation improvement |
| `--save-every`  | int  | `10`    | Save a checkpoint every N epochs                       |
| `--grad-accum`  | int  | `1` (off) | Gradient accumulation steps before optimizer step    |
| `--warmup-steps`| int  | `0` (off) | Linear LR warmup over N optimizer steps              |

---

## 4. Loss Function

| Argument             | Type  | Default    | Description                                                               |
|----------------------|-------|------------|---------------------------------------------------------------------------|
| `--lam`              | float | `0.5`      | Loss blend: `lam × eval_loss + (1 − lam) × result_loss`                  |
| `--eval-scale`       | float | `400.0`    | Centipawn scaling factor applied before sigmoid                           |
| `--label-smoothing`  | float | `0.0` (off)| Label smoothing epsilon for result loss                                   |
| `--filter-eval-max`  | float | `0.0` (off)| Hard-drop positions where \|eval\| exceeds this value                     |
| `--eval-soft-cap`    | float | `0.0` (off)| Downweight positions with \|eval\| above this value instead of dropping   |
| `--draw-weight`      | float | *(none)*   | Loss multiplier for drawn positions (1.0 = neutral; enhanced default: 3.0)|
| `--mate-boost`       | float | `3.0`      | Weight multiplier for decisive positions with \|eval\| > 2000 cp          |
| `--phase-balanced`   | flag  | on         | Balance loss contribution by game phase (opening/middlegame/endgame)      |

---

## 5. Optimizer & Regularisation

| Argument         | Type  | Default | Description                               |
|------------------|-------|---------|-------------------------------------------|
| `--lr`           | float | `0.001` | Initial learning rate                     |
| `--lr-min`       | float | `1e-6`  | Minimum LR floor for cosine schedule      |
| `--grad-clip`    | float | `1.0`   | Gradient clipping max norm                |
| `--weight-decay` | float | `0.01`  | AdamW weight decay (L2 regularisation)    |
| `--dropout`      | float | `0.1`   | Dropout rate applied during training      |

---

## 6. Learning Rate Schedule

| Argument                                   | Type  | Default | Description                                                         |
|--------------------------------------------|-------|---------|---------------------------------------------------------------------|
| `--cosine-lr` / `--no-cosine-lr`           | flag  | on      | Use cosine annealing LR schedule; `--no-cosine-lr` = constant LR   |
| `--cosine-restarts` / `--no-cosine-restarts` | flag | on    | Warm restarts; disable for single cosine decay (better for fine-tuning) |
| `--cosine-t0`                              | int   | `50`    | Epochs before first restart                                         |
| `--cosine-t-mult`                          | int   | `2`     | Multiplier applied to restart interval after each cycle             |

---

## 7. Stochastic Weight Averaging (SWA)

| Argument      | Type | Default | Description                                  |
|---------------|------|---------|----------------------------------------------|
| `--swa`       | flag | off     | Enable Stochastic Weight Averaging           |
| `--swa-start` | int  | `3`     | Epoch after which SWA averaging begins       |

---

## 8. Plotting & Logging

| Argument       | Type | Default | Description                                                    |
|----------------|------|---------|----------------------------------------------------------------|
| `--plot`       | flag | off     | Generate a final summary plot after training completes         |
| `--show-plot`  | flag | off     | Automatically open the plot image (requires `--plot`)          |
| `--plot-every` | int  | `10`    | Save an intermediate plot every N epochs during training       |
| `--clear-log`  | flag | off     | Delete `training_log.csv` before training for a clean history  |

---

## 9. Enhanced Mode

| Argument     | Type | Default | Description                                                                                         |
|--------------|------|---------|-----------------------------------------------------------------------------------------------------|
| `--enhanced` | flag | off     | Master switch that enables all enhancements with sensible defaults in one flag:                     |
|              |      |         | • `--warmup-steps 1000`                                                                             |
|              |      |         | • `--grad-accum 4`                                                                                  |
|              |      |         | • `--label-smoothing 0.02`                                                                          |
|              |      |         | • `--filter-eval-max 10.0`                                                                          |
|              |      |         | • `--draw-weight 3.0`                                                                               |

---

## 10. Hardcoded Top-level Constants

These values are set directly in source code and cannot be overridden via command line.

| Constant                | Value       | Description                                                   |
|-------------------------|-------------|---------------------------------------------------------------|
| `NUM_FEATURES`          | `768`       | Input feature count: 6 piece types × 2 colours × 64 squares  |
| `MAX_PRELOAD_POSITIONS` | `8,000,000` | Maximum positions loaded into RAM before switching to streaming|
| `STREAM_CHUNK_SIZE`     | `200,000`   | Default chunk size when streaming positions from disk         |

---

*End of document*
