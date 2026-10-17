"""Resume the interrupted baseline run. Training starts only when this file is executed."""

import os
from pathlib import Path

from ultralytics import RTDETR


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_DIR / 'runs' / 'rtdetr-resnet18' / 'weights' / 'last.pt'


def main():
    # The saved data YAML uses paths relative to this repository.
    os.chdir(PROJECT_DIR)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Baseline checkpoint not found: {CHECKPOINT}')

    # Load the checkpoint itself: constructing from YAML leaves ckpt_path empty.
    model = RTDETR(str(CHECKPOINT))
    checkpoint = model.ckpt
    completed_epochs = int(checkpoint.get('epoch', -1)) + 1
    saved_args = checkpoint.get('train_args', {})
    total_epochs = int(saved_args.get('epochs', 0))
    if checkpoint.get('optimizer') is None:
        raise RuntimeError('The checkpoint has no optimizer state; it cannot resume this interrupted run.')
    if not 0 < completed_epochs < total_epochs:
        raise RuntimeError(f'No unfinished training to resume: {completed_epochs}/{total_epochs} epochs.')

    print(f'Resuming baseline from epoch {completed_epochs + 1} to {total_epochs}; '
          f'batch={saved_args.get("batch")}, imgsz={saved_args.get("imgsz")}, '
          f'save_dir={saved_args.get("save_dir")}')
    # Recover epoch, optimizer, EMA, output directory and training arguments.
    # Do not override batch/epochs here: continue with the original experiment settings.
    model.train(resume=True)


if __name__ == '__main__':
    main()
