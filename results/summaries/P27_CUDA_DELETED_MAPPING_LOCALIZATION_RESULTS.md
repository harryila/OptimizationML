# P27 CUDA deleted-mapping localization outcome

## Verdict

Attempt `20260906-01` is **terminal before container creation**. The sole
`prepare-runtime` invocation stopped at P27's independent contract
reconstruction. It did not inspect the image, access the GPU, create a
container, initialize CUDA, run the deleted-mapping localizer, construct a
model or optimizer, execute training code, or observe a candidate.

The remote checkout had the exact source-freeze commit
`1071afa1b20b3e12be3dafdebf9dae63f39b0ed9`, tree
`e32a8d89a4470c7e4ee2631c8abb16c2f0a4d923`, and canonical P27 contract
SHA-256
`02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a`.
The reconstruction passed 22 of 23 checks. Its only false check was
`terminal_p26_commit_tree_tag_and_bytes_exact`.

## Exact cause

The operator-transfer bundle, SHA-256
`1fb136fb027b739f94c08a0f20cfeb7d5d417742cb3bd5578bbb278b86c641cc`,
advertised only:

- `refs/heads/p27-cuda-deleted-mapping-localization` at the exact source
  freeze; and
- `refs/tags/p26-attempt02-acquisition-source`.

It omitted the required
`refs/tags/p26-permission-safe-acquisition-checkpoint` ref and annotated tag
object `bacad707d3779bfa10957e18cb4c69b1a7f0cbce`, which peels to terminal P26
commit `5429da23ff18888daa2312c530a4587780484d8b`. P27 correctly failed closed.
This is an operator-transfer provenance defect, not a contract, localizer,
CUDA, backend, or mapping-classification result.

## Retained evidence and gates

The invocation began at `2026-09-06T14:11:41Z`. The terminal attempt contains
only its evidence directory and two mode-`0600`, uid/gid-`1000` reconstruction
logs:

| stream | bytes | SHA-256 |
| --- | ---: | --- |
| stdout | 1,718 | `911cf5e086227c95004ab9bf87438463f4c6d49ff3b77457ea89a98abca845b4` |
| stderr | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The exact path-free stdout is committed as
[`p27_prepare_contract_reconstruction.remote.json`](p27_prepare_contract_reconstruction.remote.json).
No P27 runtime lock, host attestation, native localization artifact, or
sanitized localization artifact exists. CUDA/localization observations,
optimizer steps, candidate observations, and training steps are all zero.

## Terminal policy and route

Attempt `20260906-01` must not be reused, cleaned up, or retried. The
localization invocation itself was never consumed, but this attempt identity
is nevertheless terminal because its retained evidence already records a
failed prepare phase.

The next attempt requires a new preregistered identity and a reviewed,
hash-bound transfer that carries and verifies the P26 checkpoint ref, its
annotated object, and its peeled commit **before creating the attempt root**.
This outcome creates no P28 contract and authorizes no scientific acquisition.

P18--P21 and the terminal P26 result are unaffected. P27 has not yet
established the deleted mapping's identity, CUDA repeatability,
observer noninterference, real-gradient fidelity, native shield safety, or
neural-training behavior.

## Machine-readable record

The compact
[`p27_cuda_deleted_mapping_localization_outcome.json`](p27_cuda_deleted_mapping_localization_outcome.json)
has canonical SHA-256
`d87df6cc792dfa0a0d6a2a982395953ba2ef87aa2086266d9d3c59abb3427d0e`.
Reconstruct it with:

```bash
python3 scripts/reconstruct_p27_cuda_deleted_mapping_localization_outcome.py
```
