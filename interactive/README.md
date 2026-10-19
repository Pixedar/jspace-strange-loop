# Starting-state interventions

The [interactive experiment](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space) compares the responses of two trained feedback interfaces on matching external text. One was trained to describe its own possible updates; the control describes another run's recordings.

At a fixed recorded context, the interface is evaluated from 12 candidate control states. The bars show the distance between each resulting next state and the next state from a selected reference candidate. Distances use all eight control coordinates. Because the states have length one, their distance ranges from 0 to 2. A near-zero distance means the candidates lead to nearly the same next state. It does not mean their starting states were close.

The default view selects token 600, copy 1 of a strong developmental run and its control. All 60 sampled contexts and four copies are included. Each copy reads a separate text stream; each interface maintains its own model cache. At a given context, the cache is held fixed while candidate states are tested. This is a saved one-step intervention, not a simulation or a new model run.

The path-cut view displays separately recorded interventions holding either the reader input or the route into model attention fixed. It measures the proportion of steps with distance greater than 0.5. The [report](../REPORT.md) gives the training comparisons, sampling limits and complete results.

## Run locally

From the repository root:

```bash
python -m http.server 8000 --directory interactive
```

Open `http://localhost:8000`. No model weights or inference are needed to use the page. To regenerate its data after downloading the selected artifacts:

```bash
python figures/make_probe_view.py
```

`probe_responses.json` contains the original response coordinates, source hashes, sampling positions, path-cut results and six final-interface summaries. `trajectories.json` retains supplementary recordings projected onto two pooled principal-component axes, with retained variance recorded. That projection is not used for the displayed distance comparison. Neither representation assigns word meanings to the learned state coordinates.
