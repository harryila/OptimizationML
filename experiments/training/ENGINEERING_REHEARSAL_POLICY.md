# Engineering rehearsals and scientific acquisitions

This policy separates repeatable launch-path engineering from frozen scientific
acquisition. It applies to P30 and to later accelerator protocols.

## Engineering rehearsal

An engineering rehearsal may exercise provisioning, identities, permissions,
source transfer, receipt creation, sealing, container launch plumbing, and other
non-scientific handoffs. It must:

- use a disposable namespace and an identity that is not a frozen acquisition
  attempt identity;
- use no scientific gradient, candidate, optimizer-update, or training
  observation;
- avoid CUDA initialization unless the rehearsal contract explicitly needs GPU
  plumbing and still forbids scientific observations;
- retain exact client-side stdout, stderr, and exit status for the earliest
  provisioning, verifier, and seal commands outside any directory whose exact
  contents are frozen by an acquisition contract; and
- be repeatable while engineering defects are corrected.

A failed rehearsal is an engineering result, not a new research phase. It does
not authorize deletion or rewriting of a historical acquisition attempt.

## Scientific acquisition

A scientific acquisition begins only after the complete launch path has passed
its required rehearsal and the acquisition source, gates, attempt identity, and
evidence policy have been frozen. Once acquisition state is created, its
one-shot and no-repair rules apply exactly. All prior terminal attempts, tags,
native records, and sanitized records remain immutable.

Rehearsal evidence cannot be promoted into scientific evidence. Scientific
thresholds cannot be tuned from confirmatory observations unless a later
held-out protocol is preregistered first.

## Client stream retention

For each client-initiated state boundary, retain three no-overwrite files:

- `<label>.stdout.log`;
- `<label>.stderr.log`; and
- `<label>.exit-status.txt`.

The status file contains one canonical decimal exit status and a newline. Raw
client streams remain external to exact-content acquisition directories.
Curated outcomes may bind sanitized copies or hashes after review. Command-line
arguments are not copied into the stream bundle because they may contain
credentials or other operator secrets.

P30's disposable Linux UID-1000-to-root handoff rehearsal is governed by this
policy. It authenticates engineering behavior only; it does not initialize
CUDA, inspect deleted mappings, or authorize training.
