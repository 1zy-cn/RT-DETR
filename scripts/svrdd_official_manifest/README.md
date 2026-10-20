# SVRDD official split lists

`train.txt`, `val.txt`, and `test.txt` are copied byte-for-byte from
`SVRDD_YOLO.zip` in the [SVRDD Zenodo record](https://zenodo.org/records/10100129)
(DOI: 10.5281/zenodo.10100129, CC BY 4.0). They list 6,000, 1,000, and 1,000
different image filenames, respectively. The original archive organizes images
by district and uses Windows-style path separators in these lists.

SHA-256:

| File | SHA-256 |
| --- | --- |
| `train.txt` | `677f9284a12d4763affe8111b0a187fe5d99d7bd4771bfa1f8cb186ccaf533eb` |
| `val.txt` | `3a1dc0a613849072342ac630cc612e4d9317a22e8d5f3f302e676a7398ee1f7c` |
| `test.txt` | `ea7ab251adbb1d4f0463e99dc8a0da4cc1b57a72d1001a131956a0fba38168d9` |

Use `scripts/prepare_svrdd_official_split.py` to rebuild a disjoint YOLO
directory from the previously downloaded SVRDD data. The script preserves the
source directory and records the manifest hashes with the generated dataset.
