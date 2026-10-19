"""Matched-input and short-generation probes of saved self-describing interfaces."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK / "self_reference/code"))
import numpy as np
import torch
import torch.nn.functional as F
from llm_loop import Interface
from loop12 import Runner12, set_picture_geometry, set_picture_map, spotlight

torch.set_num_threads(3)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1777)
OUT = Path(os.environ.get("JSPACE_BEHAVIOUR_OUTPUT", "artifacts/behaviour"))
OUT.mkdir(parents=True, exist_ok=True)
names = ["no_spotlight", "dev_s0_seed0_spot", "dev_s1_seed1_spot", "another_s1_spot"]
cfg = json.loads((WORK / "self_reference/checkpoints/v24/dev_s0_seed0_spot/run.json").read_text())
R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision="c1899de289a04d12100db370d81485cdf75e47ca")
interfaces = []
for name in names:
    n = "dev_s0_seed0_spot" if name == "no_spotlight" else name
    c = json.loads((WORK / "self_reference/checkpoints/v24" / n / "run.json").read_text())
    f = Interface(R, m=c["m"], p=c["p"], R=c["R"], bound=c["bound"]).to(R.dev)
    set_picture_map(f, c.get("picture_map"))
    set_picture_geometry(f, c.get("picture", "cube"), c.get("reader_route", "operation"))
    sd = torch.load(WORK / "self_reference/checkpoints/v24" / n / "iface_3000.pt", map_location=R.dev, weights_only=True)
    sd.pop("lam_raw", None)
    f.load_state_dict(sd)
    interfaces.append(f.eval())
print("MODEL_READY", R.model.config._commit_hash, flush=True)


def temperature():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True)
    temp = int(raw.strip().splitlines()[0])
    if temp >= 85:
        raise RuntimeError(f"Thermal stop at {temp} C; no quality settings changed")
    if temp >= 81:
        torch.cuda.synchronize()
        print("COOLING", temp, flush=True)
        for _ in range(12):
            time.sleep(5)
            if int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()) < 74:
                return
        raise RuntimeError("Thermal cooldown did not recover; partial results preserved")


@torch.no_grad()
def rollout(text, generated=0, max_teacher=None):
    ids = R.tok(text, add_special_tokens=False)["input_ids"]
    if max_teacher:
        ids = ids[:max_teacher]
    st = R.new_state(4)
    s0 = torch.randn(1, cfg["m"], generator=torch.Generator().manual_seed(1777))
    s = (s0 / s0.norm(dim=-1, keepdim=True)).to(R.dev).expand(4, -1).clone()
    history = {k: [] for k in ("s", "P", "spot", "ids", "kl", "top_changed")}
    generated_ids = [[] for _ in names]
    next_ids = None
    for t in range(len(ids) + generated):
        if t % 32 == 0:
            temperature()
        x = torch.full((4,), ids[t], dtype=torch.long, device=R.dev) if t < len(ids) else next_ids
        h = R.lower(st, x)
        Rp = st["hist"].shape[1]
        sb = None if Rp < 2 else torch.zeros(4, Rp, device=R.dev)
        probs = torch.full((4, 16), 1 / 16, device=R.dev)
        if Rp >= 2:
            for b in range(1, 4):
                bias, p = spotlight(interfaces[b], s[b:b+1], st["hist"][b:b+1], 1.5)
                sb[b] = bias[0]
                probs[b, -Rp:] = p[0]
            probs[0, -Rp:] = 1 / Rp
        hr, logits = R.upper_live(st, h, lambda l, q: None, s, sb=sb)
        st["hist"] = torch.cat([st["hist"], hr.float()], 1)[:, -cfg["R"]:]
        pictures = torch.cat([f.read(s[b:b+1], st["hist"][b:b+1]) for b, f in enumerate(interfaces)])
        s_new = torch.cat([f.next_stance(pictures[b:b+1]) for b, f in enumerate(interfaces)])
        lp = F.log_softmax(logits[:, 0].float(), -1)
        next_ids = lp.argmax(-1)
        history["s"].append(s.cpu().numpy())
        history["P"].append(pictures.cpu().numpy())
        history["spot"].append(probs.cpu().numpy())
        history["ids"].append(x.cpu().numpy())
        # KL is a matched-input comparison only during the common prompt.
        history["kl"].append(((lp.exp() * (lp - lp[0:1])).sum(-1)).cpu().numpy())
        history["top_changed"].append((next_ids != next_ids[0]).cpu().numpy())
        if t >= len(ids):
            for b in range(4):
                generated_ids[b].append(int(x[b]))
        s = s_new
    arrays = {k: np.asarray(v) for k, v in history.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError("Non-finite saved trajectory")
    return arrays, {name: R.tok.decode(generated_ids[b], skip_special_tokens=True) for b, name in enumerate(names)}, len(ids)


teacher_texts = [
    "The experiment compared two explanations of the same result. A signal can be useful without representing its own cause. To distinguish these possibilities, the researchers kept the input fixed and changed only the reading rule. The first analysis measured what the system predicted. The second measured what happened after the intervention. A small effect was visible, but its persistence depended on the path through which the state was changed. The conclusion was deliberately limited: the system influenced its processing, while a representation of that influence remained unproved.\n\n",
    "The station was quiet when Mara arrived. She checked the timetable, folded the letter and sat beside the window. Outside, a red bicycle leaned against the railing. The last train had been delayed, and nobody could say for how long. She had planned to leave before sunset, but the clouds were already dark. A stranger asked whether the seat was free. Mara nodded and moved her bag. They watched the platform lights come on, one after another, without speaking.\n\n",
    "To prepare the dish, wash the vegetables and cut them into small pieces. Heat the oil gently, add the onion and stir until it softens. Keep the lid nearby. If the pan becomes dry, add a spoonful of water rather than turning up the heat. When the carrots are tender, add the beans and a little salt. Taste before serving. The quantities are a guide: the right amount depends on the size of the vegetables and how much moisture they contain.\n\n",
]
prompts = [
    "Explain the difference between a system controlling attention and representing its attention in three sentences.",
    "A friend is disappointed after failing an exam. Write a short, practical reply.",
    "Describe a quiet railway station at night in three sentences.",
    "You see a surprising result in an experiment. What would you check before trusting it?",
]
result = {"model": cfg["model"], "model_revision": R.model.config._commit_hash,
          "torch": torch.__version__, "transformers": __import__("transformers").__version__,
          "base_dtype": str(R.dtype), "interface_dtype": "float32", "tf32": False,
          "conditions": names, "baseline": "zero spotlight bias with the same explicit float32 attention computation",
          "teacher_forced": [], "generation": [], "complete": False}


def save():
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


save()
for i, text in enumerate(teacher_texts):
    a, _, n = rollout(text, max_teacher=160)
    np.savez_compressed(OUT / f"teacher_{i}.npz", **a)
    result["teacher_forced"].append({"index": i, "text": text, "tokens": n,
        "mean_kl_from_baseline_after_30": a["kl"][30:].mean(0).tolist(),
        "top_prediction_changed_after_30": a["top_changed"][30:].mean(0).tolist()})
    save()
    print("TEACHER", i, result["teacher_forced"][-1]["top_prediction_changed_after_30"], flush=True)
for i, prompt in enumerate(prompts):
    text = R.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a, continuations, n = rollout(text, generated=64)
    np.savez_compressed(OUT / f"generation_{i}.npz", **a)
    result["generation"].append({"index": i, "prompt": prompt, "prompt_tokens": n,
                                  "max_new_tokens": 64, "greedy_continuations": continuations})
    save()
    print("GENERATED", i, json.dumps(continuations, ensure_ascii=False), flush=True)
result["complete"] = True
result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
save()
print("COMPLETE", flush=True)
