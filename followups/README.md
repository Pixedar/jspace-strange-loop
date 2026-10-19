# Bounded follow-ups: what changes, and what the changes mean

These investigations were completed on 19 October 2026 after the final spotlight batch. They are exploratory, use small samples and answer narrower questions than stable identity or subjective experience. All conditions, including unsuccessful controls, are retained. New paid compute cost: **USD 0**. The saved-life probe used a laptop RTX 3050 Ti; the native multimodal pilot used a free Colab Tesla T4.

## 1. Saved interfaces change wording; this does not establish personality

Three saved 0.6B interfaces—two self-describing runs and the recorded-other control—were compared with a zero-spotlight baseline. Every condition used the same explicit float32 attention implementation, including the baseline, avoiding a zero-bias versus native-kernel comparison. The base model was BF16 Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`; TF32 was disabled.

On three matched prose inputs, after the first 30 tokens, the best self-describing interface changed 23.2%, 19.0% and 20.0% of top predictions. The second changed 14.5%, 15.9% and 9.2%; the recorded-other control changed 13.0%, 12.7% and 15.4%. These are input-specific top-prediction differences, not accuracy gains.

Four fixed prompts then received 64-token greedy continuations under each condition. Their wording changes, including under the control. The prompts, full truncated continuations, token counts, mean distribution changes and all recorded state/attention trajectories are preserved. A few divergent continuations are not a personality scale, a replication across readers, or evidence of a self-description-specific persona. No model self-report is used to explain a hidden state.

See [behavior results](results/behaviour/results.json), [`code/behaviour.py`](code/behaviour.py), and the numeric recordings in the data release. The [recorded viewer](../interactive/) uses the earlier controlled frozen passes, with a common fitted projection and variance retained stated explicitly. Proximity in that projection is not treated as semantic similarity. The old J-space word lens and the new learned picture are different representations; this follow-up does not invent a word interpretation for the new picture.

## 2. Native text, image and audio: a loop can develop without strong fusion

**Built with Qwen.** The native model retains the [Qwen Research license](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE), which permits research/evaluation use and is not an open-source license. This release publishes the independent experimental code and measured observations, without redistributing the base model or the multimodal reader checkpoints. Their checkpoint hashes are recorded; the complete private artifact backup is retained separately. The main Qwen3 interface checkpoints belong to a different model family and remain in the public data release.

The model was the unquantized **Qwen2.5-Omni-3B Thinker**, revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`, with the native image/audio encoders and joint text decoder. The loaded Thinker contains about 4.70 billion parameters. We used native eager attention, FP16 inference, float32 learned readers and hidden-state tuple index 25. Native inference peaked around 10.7 GB of allocated GPU memory. The original laptop's 4 GB GPU cannot hold this unquantized configuration; a free T4 provided the required memory.

An image contains left and right colored panels. An audio direction specifies which panel's color to name. The four image/direction combinations of red/blue train the reader; green/yellow provides four held-out-color combinations. Each image is also reversed. This is a deliberately small factorial task, not a benchmark or a test of general multimodal competence.

The first pilot used one or two beeps. Image controls passed, but the model called both audio controls “two.” Its joint score was 4/8, the same as silence. That pilot cannot test integration because its audio positive control failed. A documented exploratory revision used equal-duration offline-synthesized “left” and “right” speech. Both transcription controls and all four image controls then passed. Joint color choice was 5/8; silence and gray-image interventions each gave 4/8. Giving the direction as text gave 8/8. Eight trials do not establish a reliable native binding advantage. Historical JSON keys `beeps: 1/2` refer to left/right in the spoken revision.

### Matched reader comparison

The centered-field reader objective was transferred to cached native states. A 2,000-step position-discrimination warm-up with zero divergence preceded 3,000 centered-field steps. Two seeds used identical initialization, probes and deterministic per-step permutations of four distinct contexts. Each self-description run had a recorded-other-field control. There was **no prediction-loss training, native-model response feedback, temporal native cache or full live spotlight training**. These omissions make this a mechanism screen, not a replication of the complete v24 architecture.

Two intake windows were compared from the same native forward passes:

- **Recent text:** the last 16 contextual decoder positions.
- **Sampled modalities:** four image positions, four audio positions and eight recent text positions, with 16 positions total.

| Intake | Seed | Description target | Held-out normalized dependence | Pattern correlation |
|---|---:|---|---:|---:|
| Recent text | 0 | Itself | 0.000281 | +0.001 |
| Recent text | 1 | Itself | <0.000001 | +0.002 |
| Recent text | 0 | Recorded other | 0.000147 | −0.144 |
| Recent text | 1 | Recorded other | <0.000001 | +0.016 |
| Sampled modalities | 0 | Itself | 0.512 | +0.795 |
| Sampled modalities | 1 | Itself | 0.505 | +0.866 |
| Sampled modalities | 0 | Recorded other | 0.000031 | −0.064 |
| Sampled modalities | 1 | Recorded other | <0.000001 | +0.064 |

These are the same normalized-ratio and pattern-agreement concepts as the main report, not probabilities. Raw explained variance for the two successful screens remains about −584 and −604: the field head is again uncalibrated. Clamping the state-dependent reader query makes destinations essentially constant over probes. The earlier preliminary suffix screen is also retained; it used a divergence term and sampling with replacement. Comparisons to that preliminary screen cannot isolate the intake change. The final table above uses the matched corrected schedule in both arms.

### Input interventions prevent a fusion overclaim

The rich sampled-modality loops are **dominated by spoken direction**. Across training and held-out colors, reversing image panels changes next destinations by RMS 0.0028–0.0087; swapping left/right speech changes them by 0.676–0.695. The image effect is about 0.4–1.3% of the audio effect. A small nonadditive factorial interaction is measurable, but does not establish a unified shared representation. Gray-image and silence interventions use full native encoder/decoder forwards, rather than zeroing selected latent features. Reader attention mass is descriptive and is not a causal decomposition of modality contributions; contextual audio/text positions can already carry image information.

The recorded-other control is also strongly direction-sensitive. State dependence, input sensitivity, task success, and integration across modalities are separate properties. A visually rich trajectory should not be mistaken for a merged inner world.

One final frozen one-shot test applied the learned reading as the same bounded log-attention bias to native decoder layers 18–24. It affected the final text queries; it was not live recurrent learning. Zero bias exactly reproduced unmodified logits. For seed 0, the self reader changed no top predictions on eight trials, while moving distributions by median KL 0.073. A reversed-position-bias control changed one; the recorded-other reader also changed one. Return to the self picture was at most 0.000247 relative to its baseline norm. Some sampled modality positions precede the affected text queries, so this tiny return partly reflects the one-shot geometry and must not be treated as an estimate for a fully recurrent multimodal loop.

![Matched native reader screen and input interventions](../figures/multimodal_followup.png)

The limited positive result is a transferred centered-field mechanism with suitable access to native states. Reliable cross-modal use, richer stimuli, live prediction-loss learning, feedback through the native model, and longer-lived identities remain untested. This does not establish qualia, subjective unity or self-awareness.

## Reproduction

Fetch the [selected artifacts](../self_reference/README.md), then run `python followups/code/behaviour.py` for the local text-model probe. Set `JSPACE_BEHAVIOUR_OUTPUT` to choose a fresh output directory. Allow roughly 1.3 GB of GPU memory for the qualified 0.6B probe; do not silently substitute a new precision/kernel for a replay.

For the multimodal pilot, use a fresh CUDA environment with `transformers==5.8.1`, PyTorch, Accelerate, NumPy, Pillow and SoundFile, and at least the measured memory plus headroom. Review the upstream model's license. From the repository root, execute these files **in the same Python/notebook namespace**:

```python
from pathlib import Path
for name in ["omni_setup", "beep_perception", "spoken_perception", "preliminary_screen",
             "paired_intakes", "intake_screen", "native_spotlight", "modality_check"]:
    exec(Path(f"followups/code/{name}.py").read_text(), globals())
```

The small speech waveforms and exact perception stimuli construction are included. No base-model weights or multimodal reader checkpoints are redistributed. `source_hashes.json` identifies the executed source snapshots before portable-path and transport-output edits. JSONs preserve the complete reported conditions; selected native features and response fields are in the immutable data release. The experiments used one GPU execution each; no backend-equivalence or broad seed-generalization claim is made.
