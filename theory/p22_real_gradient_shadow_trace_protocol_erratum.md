# P22 shadow-protocol arithmetic erratum

This erratum corrects one sentence in the frozen P22 acquisition contract
without changing the machine-readable protocol or its acquisition hash. The
`autocast` field in
`experiments/training/p22_real_gradient_shadow_trace_protocol.json` says that
"only the pinned five Newton--Schulz stages use BF16." That wording is too
narrow.

At pinned Keller--Jordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, the candidate path first casts the
stored FP32 Nesterov signal to BF16. The one-time orientation, Frobenius norm,
addition of the Python-float `1e-7`, normalization division, and five Jordan
stages therefore operate on a BF16-stored candidate state before orientation
is restored. The upstream aspect multiplier is then applied before the P22
post-aspect capture. P22 verifies that the returned candidate is stored as
BF16 on the selected accelerator, but does not claim native BF16 accumulation
or tensor-core semantics.

The model parameters, forward and backward path, gradients, EMA, and Nesterov
state remain FP32 without autocast. The executed source and adapter were
correct; this is a description error only. It does not change either
trace-off execution, the exact 303-mismatch repeatability verdict, or the rule
that trace-on remains barred. Any replay must retain the original
machine-readable protocol bytes and cite this erratum alongside them.

The frozen prose protocol's scaffold section also lists the trace-on command
before the repeatability command. That ordering is editorially wrong: execute
fresh trace-off A and B, run `verify-repeatability`, and create trace-on only
if that verifier passes. This is the same prerequisite stated by the frozen
acceptance rule; the erratum prevents the example command order from defeating
the fail-closed gate.
