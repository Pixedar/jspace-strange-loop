"""Tables for v24 (PLAN v12 section 18) from a fetched batch, and the pre-registered predictions that the batch alone
decides (5 and 6 need the local analyses: spotlight_schema.py and the per-token comparison).
Per life: the regime over the continuation (dynamics_anatomy), consequences against no channel and the paths of the
field (spotlight_check_3000.json), the share and its description (self_model_residual_3000.json), the shared text,
replaced pictures and path cuts (self_formation.json); the between-lives fields (between_lives.json).
Usage: python -I v24_summary.py BATCH_DIR [V22_DIR] [EXTRA_LIFE_DIR ...] (lives without a final interface are skipped)"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1]
V22 = sys.argv[2] if len(sys.argv) > 2 else None


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def rich(a):
    return bool(a and a["state_dependent_size"] >= 0.10 and a["rests"] >= 0.5 and a["cos_next_median"] > 0
                and a["description_of_time_varying_part"] >= 0.3)


lives = {}
for d in sorted(glob.glob(os.path.join(ROOT, "*"))) + [x for x in sys.argv[3:]]:
    if not os.path.exists(os.path.join(d, "run.json")) or not glob.glob(os.path.join(d, "iface_3000.pt")):
        continue
    n = os.path.basename(d)
    cfg = json.load(open(os.path.join(d, "run.json")))
    L = dict(cfg=cfg, anatomy={int(re.findall(r"iface_(\d+)", f)[0]): load(f) for f in glob.glob(os.path.join(d, "dynamics_anatomy_iface_*.json"))},
             check=((load(os.path.join(d, "spotlight_check_3000.json")) or {}).get(n) or {}).get("against_no_channel") and
             (load(os.path.join(d, "spotlight_check_3000.json")) or {}).get(n), smr=load(os.path.join(d, "self_model_residual_3000.json")),
             shared=load(os.path.join(d, "shared_text_iface_3000.json")), formation=load(os.path.join(d, "self_formation.json")))
    lives[n] = L
    kind = ("spotlight" if cfg.get("spotlight") else "no spotlight") + (" (neighbour's)" if cfg.get("spotlight_from") == "neighbor" else "") + \
           (", describes another" if cfg.get("describe") == "another" else "")
    print(f"== {n}: {kind}, start {os.path.basename(cfg['init_iface'])}, seed {cfg['init_seed']}")
    for it in sorted(L["anatomy"]):
        a = L["anatomy"][it]
        if a:
            print(f"   {it:5d} dependence {a['state_dependent_size']:.3f} rests {a['rests']:.2f} big {a['big_steps']:.2f} cos {a['cos_next_median']:+.2f} | fixed {a['signature_share']:.2f} "
                  f"copy-specific {a['copy_specific_share']:.2f} genre {a['time_varying_genre_share']:.2f} ({a['same_genre_shifted']:.2f}) | described {a['description_of_time_varying_part']:+.2f} / "
                  f"{a['description_of_signature']:+.2f} | apart {a['copies_apart_frozen']:.2f} {'RICH' if rich(a) else ''}")
    c = L["check"]
    if c:
        an = c["against_no_channel"]
        print("   consequences vs no channel: " + " | ".join(f"{m}: KL {an[m]['kl']:.3f} top-1 {an[m]['top_prediction_changed']:.3f} loss {an[m]['loss_change_pct']:+.1f}%" for m in ("own", "neighbor", "shift")))
        fr = c["field_routes"]
        print(f"   field: both {fr['both']:.4f} reading {fr['reading']:.4f} spotlight {fr['spotlight']:.4f} (spotlight / both {fr['spotlight'] / max(fr['both'], 1e-12):.3f}) | immediate top-1 {c['immediate']['top_prediction_changed']:.3f} attention moved {c['immediate']['attention_moved']:.3f}")
        g = c["gradients"]
        print("   gradients prediction / game on the reader: " + ", ".join(f"{e['ext_on_reader'] / max(e['game_on_reader'], 1e-12):.1f}" for e in g) + " | cos " + ", ".join(f"{e['cosine']:+.2f}" for e in g))
    if L["smr"]:
        print(f"   share of the field state-dependent {L['smr']['remainder_share_of_field']:.3f}, described {L['smr']['predicted_remainder_correlation']:+.2f}")
    if L["shared"]:
        a_ = L["shared"]["after_switch"]
        print("   shared text, copies apart: " + " ".join(f"{k}:{v['stance_apart']:.3f}" for k, v in a_.items()))
    fm = L["formation"]
    if fm:
        rem = fm["picture_difference_remaining"]
        print("   replaced picture left after 1/5/20: " + " | ".join(f"{k}: " + "/".join(f"{rem[k][x]:.2f}" for x in ("1", "5", "20")) for k in rem))
        print("   path cuts (rests / big): " + " | ".join(f"{k}: {v['rest_share']:.2f}/{v['jumps_over_half']:.2f}" for k, v in fm["jumps"].items()))

print("\nPre-registered predictions (PLAN section 18)")
spot = [n for n, L in lives.items() if L["cfg"].get("spotlight")]
own_spot = [n for n in spot if lives[n]["cfg"].get("spotlight_from", "own") == "own"]
p1 = {n: lives[n]["check"]["against_no_channel"]["own"]["top_prediction_changed"] for n in own_spot if lives[n]["check"]}
print(f"1. own spotlight changes the top prediction on >= 2% of tokens in every spotlight life: {all(v >= 0.02 for v in p1.values())} {p1}")
rc = {n: rich(lives[n]["anatomy"].get(3000)) for n in ("rich_s0s2_spot", "rich_s1s1_spot") if n in lives}
print(f"2. a rich continuation with the spotlight still rich at 3,000: {any(rc.values())} {rc}")
p3 = {n: lives[n]["check"]["field_routes"]["spotlight"] / max(lives[n]["check"]["field_routes"]["both"], 1e-12) for n in own_spot if lives[n]["check"]}
print(f"3. >= 10% of the dependence through the model in some spotlight life: {any(v >= 0.10 for v in p3.values())} {({k: round(v, 3) for k, v in p3.items()})}")
p4 = {n: lives[n]["check"]["against_no_channel"]["own"]["loss_change_pct"] for n in own_spot if lives[n]["check"]}
print(f"4. the spotlight lowers the next-token loss in some spotlight life: {any(v < 0 for v in p4.values())} {({k: round(v, 2) for k, v in p4.items()})}")
pairs = (("rich_s0s2_spot", "rich_s0s2_nospot"), ("rich_s1s1_spot", "rich_s1s1_nospot"))
p7 = {a: (lives[a]["smr"]["remainder_share_of_field"], lives[b]["smr"]["remainder_share_of_field"]) for a, b in pairs if a in lives and b in lives and lives[a]["smr"] and lives[b]["smr"]}
print(f"7. both spotlight continuations end less state-dependent than their pairs: {all(x < y for x, y in p7.values())} {p7}")
devs = ("dev_s0_seed0_spot", "dev_s0_seed2_spot", "dev_s1_seed1_spot")
n_rich = sum(any(rich(a) for a in lives[n]["anatomy"].values()) for n in devs if n in lives)
print(f"8. developmental starts with the spotlight reach the rich regime in fewer than 2 of 3 (v22: seeds 2 and 1): {n_rich < 2} ({n_rich} of 3)")
bl = load(os.path.join(ROOT, "between_lives.json"))
if bl:
    print("\nBetween lives (state-dependent fields on the same texts; same life at 2,500 vs 3,000 marked)")
    for r in bl:
        if r["same_life"] or (r["a"].endswith("@3000") and r["b"].endswith("@3000")):
            print(f"   {r['a']:>28s} vs {r['b']:<28s} {'(same life)' if r['same_life'] else '':12s} dependence {r['dependence_corr']:+.2f} sizes {r['sizes'][0]:.3f}/{r['sizes'][1]:.3f}")
