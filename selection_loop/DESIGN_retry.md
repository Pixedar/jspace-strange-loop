# Research v7b — does the state know whether its own read worked? (design, written before results, 14 Oct 2026)

## Why
Today's main experiment (`RESULTS.md`) shows a state that represents *which animal matters* and steers the model's
read through that representation. That is a representation of what matters for the task, not yet of the system's own
processing. The next step: let the system's own reads sometimes fail and ask whether the state represents what happened to its own read, and uses that to read again.

## Set-up (`focus_retry.py`, built on `focus_sel.py`)
Same streams, records, frozen model (Qwen2.5-0.5B-Instruct) and loop as the main experiment. Changes:
- every read can fail to deliver (probability 0.3): the colour does not arrive; what arrives is noise plus the read's
  own "nothing new" weight. The colour passes a noisy channel (noise 0.1 per colour number), so a faint read delivers
  mostly noise;
- each chunk allows a second, optional read. Both reads cost a little (0.1 × the attention placed on records);
- the second read's query comes from the state *after* the first read has passed through the frozen model (`retry`),
  or from the state *before* it (`blind`, which cannot know what happened); `single` has one read per chunk;
- `retry` without the "nothing new" weight in what arrives (`--no_null 1`): then only the presence or absence of a
  colour signal tells the state what happened.
- the read cost is off for the first 300 steps and ramps in over the next 300 (with it on from the start, both two-read
  arms stopped reading within 100 steps, before reading could be useful).
Training: the right colour at every question plus the read cost. Nothing rewards reading again or not. 1,200 steps,
batch 16, one seed per arm (more if the first result is clear).

**Change made during the run (11:05 UTC), after seeing only learning speed, none of the measures below.** From scratch
the arms learned far too slowly under failures and channel noise (accuracy 0.39 / 0.54 / 0.62 for retry / blind /
single after 600-1,000 steps, against 0.88 for the main experiment's state arm after 600 steps without failures): the
noise drowns the early, half-aimed reads, so the focus is barely learned. The arms were stopped and restarted from the
main experiment's trained state loop (seed 0; the replicate from seed 1): every weight copied, the read flag's weights
zero, every arm from the same checkpoint. Selection is then already there; what to do about failed reads and about the
second read must still be learned. Read cost from step 100, full by step 200; 600 steps. Arms: retry, blind, retry
without the "nothing new" weight (seed-0 checkpoint), retry and blind (seed-1 checkpoint), single (seed-0 checkpoint).

## What would count as the state representing the outcome of its own read and using it (fixed before the results)
1. **Selective second read.** In `retry`, attention of the second read on the tracked record is high after a failed
   first read and low after a delivered one; `blind` cannot make this distinction.
2. **Same accuracy for fewer reads** than `blind`, and higher accuracy than `single`.
3. **Paired counterfactual**: same stream, history, first attention and channel noise,
   branching only on whether the first read is delivered. Exchanging the two intermediate states for the second query
   alone should move the second read with the exchanged state (and the answer with it: the delivered branch still
   answers right; the failed branch answers wrong if the exchanged state suppresses the second read).
4. **Edits** of the intermediate state along failed-minus-delivered (class means from separate calibration streams):
   toward "failed" raises the second read after delivered first reads, toward "delivered" lowers it after failed ones,
   through the query alone; random directions of the same size do much less. (Edits applied only to the second update
   cannot change the second read by construction: a wiring check, not evidence.)
5. **Follows what happened, not a habit:** at test-time failure rates of 0 and 0.6 (trained at 0.3) the second read
   after failed vs delivered first reads stays separated, and the number of reads follows the failure rate.
6. `retry --no_null 1`: if it also passes 1-4, the state reads the outcome from the colour signal itself, without the
   explicit trace of its own attention.

What a positive result would and would not show: outcome-sensitive control — a
learned representation of whether its own read delivered, used to regulate further selection. Not a general self-model,
and not knowledge of its own reliability; a learned "missing signal" detector would also pass. A refutation: the second
read as frequent after delivered as after failed first reads, or a decodable outcome that edits and exchanges do not
act through.
