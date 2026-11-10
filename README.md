# pytorch-training-template

A compact, reproducible PyTorch training script that I use as a template for training runs. It trains a small CNN on MNIST and covers the basics that real training code needs.

## Features

- **Reproducibility:** seeds Python and PyTorch, and uses a seeded train/validation split
- **Device selection:** uses the GPU automatically when available, otherwise the CPU
- **Validation every epoch:** reports loss and accuracy on held-out data
- **Best-checkpoint saving:** keeps the weights with the best validation accuracy
- **Metrics log:** writes per-epoch results to a JSON file
- **Command-line options:** epochs, batch size, learning rate, seed and output folder

## Usage

```bash
pip install torch torchvision
python train.py --epochs 5 --batch-size 128
```

MNIST is downloaded automatically into a `data/` folder on the first run.

| Option | Default | Meaning |
| --- | --- | --- |
| `--epochs` | 5 | Number of passes over the training data |
| `--batch-size` | 128 | Examples per training step |
| `--lr` | 0.001 | Learning rate for Adam |
| `--seed` | 42 | Random seed |
| `--out` | `runs/latest` | Where results are saved |

## Output

Each run writes two files to the output folder:

- `best.pt`: weights of the best model by validation accuracy
- `metrics.json`: train loss, validation loss, validation accuracy and time for each epoch

## Using it for your own model

Replace the `Net` class with another model and swap the dataset in `main()`. The seeding, validation, checkpointing and logging stay the same.
