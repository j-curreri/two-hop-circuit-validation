"""Two-hop circuit validation suite (city -> state -> capital) on Qwen3-8B + PLTs.

For every prompt the model answers correctly:
  1. build an attribution graph, keep the pruned nodes at the city token and the final token
  2. label each node as a 'state feature' if its top output logits (from the public feature data
     on HF, independent of the graph) name the state
  3. ablate: zero the state features -> does P(capital) drop? what replaces it?
     control: zero the same number of random non-state graph nodes at the same positions
  4. swap: zero the state features and inject another state's features (from a donor prompt)
     -> does the answer become the donor's capital?
"""
import gzip, json, os, random, sys, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph

REPO = "mwhanna/qwen3-8b-transcoders"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TEMPLATE = "Fact: The capital of the state containing {city} is"
SEED = 0

# (city, state, capital, label keys). Skipped: state pairs whose names collide (Carolinas,
# Dakotas, Virginias) only keep one member; cities/capitals that contain a state name.
TRIPLES = [
    ("Dallas", "Texas", "Austin"), ("Houston", "Texas", "Austin"), ("San Antonio", "Texas", "Austin"),
    ("Los Angeles", "California", "Sacramento"), ("San Francisco", "California", "Sacramento"), ("San Diego", "California", "Sacramento"),
    ("Miami", "Florida", "Tallahassee"), ("Orlando", "Florida", "Tallahassee"), ("Tampa", "Florida", "Tallahassee"),
    ("Buffalo", "New York", "Albany"), ("Rochester", "New York", "Albany"),
    ("Chicago", "Illinois", "Springfield"),
    ("Philadelphia", "Pennsylvania", "Harrisburg"), ("Pittsburgh", "Pennsylvania", "Harrisburg"),
    ("Seattle", "Washington", "Olympia"), ("Spokane", "Washington", "Olympia"),
    ("Eugene", "Oregon", "Salem"),
    ("Detroit", "Michigan", "Lansing"),
    ("Cleveland", "Ohio", "Columbus"), ("Cincinnati", "Ohio", "Columbus"),
    ("Savannah", "Georgia", "Atlanta"),
    ("St. Louis", "Missouri", "Jefferson City"),
    ("New Orleans", "Louisiana", "Baton Rouge"),
    ("Memphis", "Tennessee", "Nashville"), ("Knoxville", "Tennessee", "Nashville"),
    ("Louisville", "Kentucky", "Frankfort"),
    ("Milwaukee", "Wisconsin", "Madison"),
    ("Las Vegas", "Nevada", "Carson City"), ("Reno", "Nevada", "Carson City"),
    ("Tucson", "Arizona", "Phoenix"), ("Flagstaff", "Arizona", "Phoenix"),
    ("Boulder", "Colorado", "Denver"),
    ("Baltimore", "Maryland", "Annapolis"),
    ("Huntsville", "Alabama", "Montgomery"), ("Mobile", "Alabama", "Montgomery"),
    ("Omaha", "Nebraska", "Lincoln"),
    ("Wichita", "Kansas", "Topeka"),
    ("Cedar Rapids", "Iowa", "Des Moines"),
    ("Newark", "New Jersey", "Trenton"),
    ("Worcester", "Massachusetts", "Boston"),
    ("New Haven", "Connecticut", "Hartford"), ("Bridgeport", "Connecticut", "Hartford"),
    ("Norfolk", "Virginia", "Richmond"),
    ("Charlotte", "North Carolina", "Raleigh"),
    ("Provo", "Utah", "Salt Lake City"),
    ("Billings", "Montana", "Helena"),
    ("Anchorage", "Alaska", "Juneau"),
    ("Hilo", "Hawaii", "Honolulu"),
    ("Albuquerque", "New Mexico", "Santa Fe"),
    ("Fayetteville", "Arkansas", "Little Rock"),
    ("Biloxi", "Mississippi", "Jackson"),
    ("Wilmington", "Delaware", "Dover"),
    ("Burlington", "Vermont", "Montpelier"),
    ("Manchester", "New Hampshire", "Concord"),
    ("Bangor", "Maine", "Augusta"),
    ("Sioux Falls", "South Dakota", "Pierre"),
    ("Casper", "Wyoming", "Cheyenne"),
]
GENERIC = {"new", "north", "south", "west"}

def state_keys(state):
    return [w for w in state.lower().split() if w not in GENERIC]

def names_state(tok, state):
    t = tok.strip().lower()
    for k in state_keys(state):
        if t == k or (len(t) >= 4 and k.startswith(t)) or (len(k) >= 5 and t.startswith(k[:-1])) or (len(k) == 4 and t.startswith(k)):
            return True
    return False

# ---- feature metadata from HF (same source the browser panel uses), cached on disk
CACHE = os.path.expanduser("~/.cache/circuit-tracer-features/" + REPO.replace("/", "__"))
os.makedirs(CACHE, exist_ok=True)
BASE = f"https://huggingface.co/{REPO}/resolve/main/features/"
idx_path = os.path.join(CACHE, "index.json.gz")
if not os.path.exists(idx_path):
    urllib.request.urlretrieve(BASE + "index.json.gz", idx_path)
_tok = os.path.expanduser("~/.cache/huggingface/token")
AUTH = {"Authorization": "Bearer " + open(_tok).read().strip()} if os.path.exists(_tok) else {}
INDEX = json.loads(gzip.open(idx_path).read())

def feature_meta(layer, feat):
    p = os.path.join(CACHE, f"{layer}_{feat}.json")
    if os.path.exists(p):
        return json.load(open(p))
    ix = INDEX[str(layer)]
    s, e = ix["offsets"][feat], ix["offsets"][feat + 1]
    raw, err = None, None
    for attempt in range(7):
        try:
            req = urllib.request.Request(BASE + ix["filename"], headers={"Range": f"bytes={s}-{e}", **AUTH})
            raw = urllib.request.urlopen(req, timeout=60).read()
            break
        except Exception as ex:
            err = ex
            time.sleep(min(60, 2 ** attempt) + random.random())
    if raw is None:
        print(f"  [warn] fetch failed L{layer}/{feat}: {err!r} -> treated as unlabelled", flush=True)
        return {"top_logits": [], "freq": None, "fetch_failed": True}
    d = json.loads(zlib.decompress(raw[4:4 + int.from_bytes(raw[:4], "little")], 47))
    keep = {"top_logits": d.get("top_logits", [])[:10], "freq": d.get("activation_frequency")}
    json.dump(keep, open(p, "w"))
    return keep

# ---- model
m = ReplacementModel.from_pretrained("Qwen/Qwen3-8B", REPO, dtype=torch.bfloat16)
tok = m.tokenizer

def first_id(word):
    return tok.encode(" " + word)[0]

def probs(logits):
    return logits[0, -1].float().softmax(-1)

def top1(p):
    return tok.decode(p.argmax().item())

def run_iv(toks, iv, freeze):
    logits, _ = m.feature_intervention(toks, iv, freeze_attention=freeze, return_activations=False)
    return probs(logits)

capital_ids = {first_id(c) for _, _, c in TRIPLES}
records, graphs = [], {}

# ---- pass 1: baseline + graph + labelled state features
for city, state, capital in TRIPLES:
    prompt = TEMPLATE.format(city=city)
    toks = m.ensure_tokenized(prompt)
    prefix = m.ensure_tokenized(TEMPLATE.split("{city}")[0] + city)
    city_pos, last_pos = len(prefix) - 1, len(toks) - 1
    with torch.no_grad():
        p = probs(m(toks.unsqueeze(0)))
    cap = first_id(capital)
    rec = dict(city=city, state=state, capital=capital, p_capital=round(p[cap].item(), 4), top1=top1(p),
               correct=p.argmax().item() == cap)
    print(f"[base] {city:14s} {state:14s} P({capital})={rec['p_capital']:.3f} top1={rec['top1']!r}", flush=True)
    if not rec["correct"]:
        records.append(rec); continue

    g = attribute(prompt=prompt, model=m, max_n_logits=10, desired_logit_prob=0.95,
                  batch_size=256, max_feature_nodes=8192, verbose=False)
    pr = prune_graph(g, node_threshold=0.8, edge_threshold=0.98)
    nf = len(g.selected_features)
    nodes = []
    for i in range(nf):
        if not pr.node_mask[i]:
            continue
        af = g.selected_features[i]
        layer, pos, feat = (int(x) for x in g.active_features[af])
        if pos in (city_pos, last_pos):
            nodes.append(dict(layer=layer, pos=pos, feat=feat, act=float(g.activation_values[af]),
                              role="city" if pos == city_pos else "last"))
    with ThreadPoolExecutor(32) as ex:
        metas = list(ex.map(lambda n: feature_meta(n["layer"], n["feat"]), nodes))
    for n, meta in zip(nodes, metas):
        n["top_logits"] = meta["top_logits"]; n["freq"] = meta["freq"]
        n["is_state"] = any(names_state(t, state) for t in meta["top_logits"])
    st = [n for n in nodes if n["is_state"]]
    rec.update(city_pos=city_pos, last_pos=last_pos, n_graph_nodes=len(nodes), n_state=len(st),
               state_feats=[{k: n[k] for k in ("layer", "feat", "role", "act", "top_logits")} for n in st])
    graphs[city] = dict(toks=toks, nodes=nodes, city_pos=city_pos, last_pos=last_pos)
    print(f"        graph nodes at city/last={len(nodes)}  state-labelled={len(st)} "
          f"{sorted({(n['layer'], n['feat']) for n in st})[:6]}", flush=True)
    records.append(rec)

# ---- pass 2: ablation, random control, swap
rng = random.Random(SEED)
ok = [r for r in records if r["correct"] and r["n_state"] > 0]
for r in ok:
    G = graphs[r["city"]]; toks = G["toks"]
    cap = first_id(r["capital"])
    st = [n for n in G["nodes"] if n["is_state"]]
    other = [n for n in G["nodes"] if not n["is_state"]]
    zero = [(n["layer"], n["pos"], n["feat"], 0.0) for n in st]
    with torch.no_grad():
        for freeze in (True, False):
            tag = "frozen" if freeze else "free"
            p = run_iv(toks, zero, freeze)
            r[f"ablate_{tag}"] = dict(p_capital=round(p[cap].item(), 4), top1=top1(p),
                                      top1_is_capital=p.argmax().item() in capital_ids)
            ctrl = []
            for _ in range(3):
                pick = rng.sample(other, min(len(st), len(other)))
                pc = run_iv(toks, [(n["layer"], n["pos"], n["feat"], 0.0) for n in pick], freeze)
                ctrl.append(pc[cap].item())
            r[f"control_{tag}"] = round(sum(ctrl) / len(ctrl), 4)

        # donor: next prompt (cyclic) with a different state
        donors = [d for d in ok if d["state"] != r["state"]]
        d = donors[ok.index(r) % len(donors)]
        D = graphs[d["city"]]
        dcap = first_id(d["capital"])
        for scale in (1.0, 2.0):
            inject = [(n["layer"], G["city_pos"] if n["role"] == "city" else G["last_pos"], n["feat"], n["act"] * scale)
                      for n in D["nodes"] if n["is_state"]]
            p = run_iv(toks, zero + inject, True)
            r[f"swap_x{scale:g}"] = dict(donor=d["city"], donor_state=d["state"], donor_capital=d["capital"],
                                        p_donor_capital=round(p[dcap].item(), 4), p_capital=round(p[cap].item(), 4),
                                        top1=top1(p), success=p.argmax().item() == dcap)
    a, s = r["ablate_frozen"], r["swap_x1"]
    print(f"[iv] {r['city']:14s} P={r['p_capital']:.2f} -> ablate {a['p_capital']:.2f} ({a['top1']!r}) "
          f"ctrl {r['control_frozen']:.2f} | swap<-{s['donor_state']}: P({s['donor_capital']})={s['p_donor_capital']:.2f} "
          f"top1={s['top1']!r} x2:{r['swap_x2']['success']}", flush=True)

with open(os.path.join(OUT, "results.jsonl"), "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print("WROTE", os.path.join(OUT, "results.jsonl"))
