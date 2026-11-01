# Small authored-input demo and symmetry check

These are native layer-24 observations from pinned Qwen3-8B over twelve original relational scenes. Each scene was processed with its own fresh model context; the reader can then read their concatenated observations. Scene boundaries are included. This is not a continuous LLM life or a reproduction of the main population experiment.

The file contains one observation stream. The runner repeats it for four same-start copies. With identical histories they must remain identical. This negative control demonstrates that the implementation does not manufacture individuality. Distinct histories are required for the reported population experiments.

From the release folder:

```powershell
python reproduce.py --states demo/states.npz --out results/demo --steps 468 --warmup 0
```

Use `--device cpu` if needed. No 8B model download is required for this recorded-input demo. The checkpoint still saves a complete reader controller; it does not contain an LLM cache. The scenes are authored for this study, and their native observations are separate from the older multi-source corpus.
