# CDRB + Conv3XC candidate implementation

Goal: configure two independently testable RT-DETR-R18 modifications for SVRDD chapter 3, with an SCSA combination and reproducible ablations.

1. Add a BasicBlock-compatible backbone block: input projection where needed, a 9x9 dilated reparameterization block before window self-attention, iRMB local depthwise convolution, optional SCSA, and the original outer residual path. Reuse the existing DRB and iRMB implementations. Keep the names clear that this is a paper-inspired adaptation, not a checkpoint-compatible reproduction.
2. Add Conv3XC with a train graph and explicit deploy fusion. Replace the three RepConv internals of the P3 top-down RepC3 (layer 19), keeping the fusion block's inputs, outputs and decoder routes unchanged.
3. Generate six static YAMLs for the 2 x 2 x 2 matrix with SCSA, CDRB and Conv3XC. Integrate them into `train.py` structure and formal-protocol checks.
4. Test shape, gradients, deployment equivalence, every model configuration and real detection loss. Report remaining risks including memory, speed, initial weights, split integrity and literature attribution.
