# Research v8 — open access: does selective reading emerge when nothing forces it? (design, 14 Oct 2026)

## Why
In the main v7 experiment (`RESULTS.md`) access was restricted: every record was encoded on its own and the loop had to
pick one through its own attention read. That made selection necessary by design. Here nothing is restricted: the frozen
model reads each chunk as ordinary text — the instruction (when there is one) and all four records together, in context,
its own attention free. The loop's only influence is the compressed state it writes into the model's processing
before the chunk; the only thing carried to the next chunk is what the loop reads back afterwards. Any selection now has
to happen inside the frozen model's own attention, steered by the written state. Whether that happens — or the system
solves the task some other way, or not at all — is the question.

## Set-up (`focus_open.py`)
Same task and streams as v7 (ten animals, ten colours, six chunks of four update records, instruction in chunk 0, a
switch in 30 % of streams, a question after every chunk that never names the animal). Per chunk:
1. the frozen model reads `Notes: [6 placeholders] <chunk text>`; at block L_w the loop writes into the placeholders its
   state (4 slots) and its self-model's forecast of the next state (2 slots);
2. the next state is read from the model's own representation of the chunk's **last token** at block L_r (a fixed
   standardisation and one linear map) — no attention of our own, no access to individual records;
3. the question is asked with the new state written into the question prompt's placeholders; the answer is the model's
   choice among the ten colour words.
Training: the right colour at every question (and the self-model's own forecasting loss, which does not train the
state). Nothing rewards any attention pattern or any content of the state. State size: 32 numbers (main) and 8 numbers
(a tighter compression, where keeping every animal's colour is much harder).

## What would show that the state regulates the model's own selection (fixed before the results)
1. **It works across chunks**: accuracy well above the same trained system with no memory (the initial state written
   in every chunk), on chunks without an instruction in view.
2. **Selective content**: the tracked animal's current colour is decodable from the state, while the new colours of
   the *other* animals updated in the same chunk are not (decoders cross-validated with whole streams held out).
   If the other animals' colours are decodable too, the system stores rather than selects.
3. **The written state steers intake**: on twin streams (same records; tracked animal A or B, both updated at chunk t
   with different colours) adding the focus difference B − A (class means from separate calibration streams) to the
   state *before* chunk t makes the answer at chunk t B's new colour — which, by 2, the unedited state never took in.
   Controls: the same edit applied *after* the chunk was read (to the state the question sees) — if B's new colour was
   never taken in, this cannot produce it; random directions of the same size; half strength; toward an animal absent
   from the chunk.
4. **The model's own attention follows the state**: attention from the chunk's last token to the tracked animal's
   record (layers between L_w and L_r) is higher than to the other records, falls without memory, and moves from A's
   record to B's record under the edit in 3.
Larger models may behave differently: the same design runs on Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct
(chosen over 1.5B after a speed test, before any result: 3B could be trained for about 2,400 steps).
