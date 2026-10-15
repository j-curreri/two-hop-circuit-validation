"""Diagnose ablation-resistant prompts: is it labeller recall, or a route outside the features?
For every testable prompt, zero ALL pruned-graph features at the city token (labelled or not),
then separately ALL graph features at the city token in layers < 20 only."""
import json, os, torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TEMPLATE = "Fact: The capital of the state containing {city} is"
R = [json.loads(l) for l in open(os.path.join(OUT, "results.jsonl"))]
T = [r for r in R if r["correct"] and r.get("n_state", 0) > 0]
m = ReplacementModel.from_pretrained("Qwen/Qwen3-8B", "mwhanna/qwen3-8b-transcoders", dtype=torch.bfloat16)
rows = []
for r in T:
    prompt = TEMPLATE.format(city=r["city"]); toks = m.ensure_tokenized(prompt)
    cap = m.tokenizer.encode(" " + r["capital"])[0]
    g = attribute(prompt=prompt, model=m, max_n_logits=10, desired_logit_prob=0.95, batch_size=256, max_feature_nodes=8192, verbose=False)
    pr = prune_graph(g, 0.8, 0.98)
    city = []
    for i in range(len(g.selected_features)):
        if pr.node_mask[i]:
            l, p, f = (int(x) for x in g.active_features[g.selected_features[i]])
            if p == r["city_pos"]: city.append((l, p, f))
    out = dict(city=r["city"], p=r["p_capital"], state_ablated=r["ablate_frozen"]["p_capital"], n_city_feats=len(city))
    with torch.no_grad():
        for name, sel in (("all_city", city), ("city_lt20", [c for c in city if c[0] < 20])):
            logits, _ = m.feature_intervention(toks, [(l, p, f, 0.0) for l, p, f in sel], return_activations=False)
            pp = logits[0, -1].float().softmax(-1)
            out[name] = round(pp[cap].item(), 4); out[name + "_top1"] = m.tokenizer.decode(pp.argmax().item())
    print(json.dumps(out), flush=True); rows.append(out)
json.dump(rows, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
print("WROTE diag.json")
