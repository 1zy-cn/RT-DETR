# SVRDD split repair (2026-10-21)

The earlier directory `/autodl-fs/data/RT-DETR/datasets/SVRDD` contains 11,498
image/label file instances but only 8,000 unique image filenames. All matching
filenames across splits have byte-identical images and labels. In particular,
1,586 validation images and 1,600 test images are also in training; another 312
images occur in both validation and test. These sets have no three-way overlap.
Metrics from the earlier split cannot be used as unbiased generalization results.
The earlier baseline was stopped, and its run directory is retained for audit.

The [official SVRDD archive](https://zenodo.org/records/10100129) includes
`train.txt`, `val.txt`, and `test.txt` with 6,000/1,000/1,000 mutually disjoint
filenames. Their union exactly matches the 8,000 unique filenames in the local
copy of the earlier dataset. The three lists and their SHA-256 hashes are in
`scripts/svrdd_official_manifest/`.

Rebuild without changing the old dataset:

```bash
python scripts/prepare_svrdd_official_split.py \
  --source /autodl-fs/data/RT-DETR/datasets/SVRDD \
  --destination /autodl-fs/data/RT-DETR/datasets/SVRDD-official-split
```

`SVRDD.yaml` points to the new directory. The three fixed-IoU experiment names
include `officialsplit` so they cannot write into the earlier run directories.
All three experiments must be restarted from random initialization with the
same official split, batch 8, 640-pixel images, 300 epochs, and seed 0. Do not
compare their mAP directly with runs from the earlier overlapping split.
