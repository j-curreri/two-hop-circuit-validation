import json, statistics as st
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "results", "results.jsonl"))]
corr = [r for r in R if r["correct"]]
T = [r for r in corr if r.get("n_state", 0) > 0 and "ablate_frozen" in r]
print(f"prompts {len(R)}  model correct {len(corr)}  with >=1 labelled state feature {len(T)}")
def summ(tag):
    base = [r["p_capital"] for r in T]; ab = [r[f"ablate_{tag}"]["p_capital"] for r in T]; ct = [r[f"control_{tag}"] for r in T]
    print(f"[{tag}] median P(capital): base {st.median(base):.2f}  state-ablated {st.median(ab):.2f}  random-ablated {st.median(ct):.2f}")
    print(f"[{tag}] P<0.5 after ablation: state {sum(a<0.5 for a in ab)}/{len(T)}  random {sum(c<0.5 for c in ct)}/{len(T)}")
    flipped_cap = [r for r in T if r[f"ablate_{tag}"]["p_capital"] < 0.5 and r[f"ablate_{tag}"]["top1_is_capital"]]
    print(f"[{tag}] of those, new top-1 is another state's capital: {len(flipped_cap)}  e.g. {[ (r['city'], r[f'ablate_{tag}']['top1']) for r in flipped_cap][:6]}")
summ("frozen"); summ("free")
for k in ("swap_x1", "swap_x2"):
    print(f"{k}: answer becomes donor's capital in {sum(r[k]['success'] for r in T)}/{len(T)}")
# what predicts failure?
res = [r for r in T if r["ablate_frozen"]["p_capital"] >= 0.5]
print("\nablation-resistant:", [(r["city"], r["n_state"]) for r in res])
print("median n_state: resistant", st.median([r["n_state"] for r in res]), " ablation-sensitive", st.median([r["n_state"] for r in T if r not in res]))
print("ablation-sensitive:", [(r["city"], r["n_state"]) for r in T if r not in res][:40])
print("\nswap x1 success donors:", sorted({r["swap_x1"]["donor_state"] for r in T if r["swap_x1"]["success"]}))
print("swap x2 fail donors:", sorted({r["swap_x2"]["donor_state"] for r in T if not r["swap_x2"]["success"]}))
print("failed baseline:", [(r["city"], r["top1"]) for r in R if not r["correct"]])
print("no state features:", [r["city"] for r in corr if r.get("n_state", 0) == 0])
