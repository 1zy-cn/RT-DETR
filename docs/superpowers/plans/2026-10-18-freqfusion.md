# FreqFusion experiments implementation plan

> Execute inline using executing-plans; review the complete change before delivery.

**Goal:** Prepare two local, selectable experiments: baseline + P3 FreqFusion and the same model with eight enabled SCSA blocks.

**Approved design:** User approved the preceding in-chat P3-only design and requested both configurations and their training entrypoint. Preserve the baseline bidirectional neck, decoder, dataset and batch-8/300-epoch protocol. Work in the existing local checkout as requested; server checkouts/branches are for the user's later run.

**Architecture:** A two-input FreqFusionConcat module enhances projected C3 and upsamples fused P4, concatenating in baseline order before RepC3. Pure PyTorch reflection-padding reassembly is fixed across environments; all ALPF/AHPF/offset branches enabled. Reference upstream commit 3fb0c70637a3c194fb74294d3ce4681958b26241. Add explicit experiment selection and preserve source snapshots, module guards and legacy HSFPN+SCSA mode.

**Tech stack:** Existing PyTorch 2.0.1, YAML parser, pytest; no new runtime dependencies.

## Tasks

- [x] Add failing operator and detector tests; implement reference-compatible FreqFusion, parser registration and two P3 YAMLs. Check numerical reassembly, real detection loss gradients, enabled SCSA and 640 inference. Compare independently against pinned upstream implementation outside the repository.
- [x] Add failing experiment-selection/provenance tests; update train.py with required --experiment, distinct shared-disk output names, common hyperparameters, --check, --data/--project/--device overrides. Snapshot the actual selected YAML and module implementation.
- [x] Document independent main/temporary-checkout commands and fixed operator provenance; run existing and new checks, both CLI checks, and obtain a whole-change review. Prepare the verified changes for publication to main as authorized by the follow-up request to finish repository management; do not start training or change server checkouts.

## Review focus

- HR/LR source order, channel count and renumbered decoder/PAN connections.
- All frequency/offset and SCSA paths receive gradients; no environment-dependent backend.
- Explicit selection rejects mismatched models and snapshots the actual selected files.
- Identical common training settings; separate persistent run directories even for /autodl-tmp code.
- CPU checks do not imply real-data accuracy or GPU throughput validation.

## Execution ledger

- Inspected clean local main at 6015c64, current parser/launcher and upstream source. Upstream standalone source has no declared repository license in GitHub metadata; retain attribution and describe the local reference-based implementation without inventing an upstream license.

- Model RED: four missing-implementation assertions failed; GREEN: four model/operator tests passed. Entrypoint RED: three missing-choice tests failed while legacy tests passed; GREEN: all six entry tests passed.
- Complete CPU suite: 18 passed; both explicit CLI checks passed; missing experiment exits 2. Upstream same-weight output/gradient comparison passed at double precision.
- Independent whole-change review completed with no blocking P1/P2 findings. Reviewed operator implementation, parser registration, model routing, training initialization, snapshots, validation evidence and branch instructions.
- Ruling: user follow-up requests repository management, so commit/push the verified common implementation to main; leave the later temporary experiment branch to the user choice documented in the guide.
