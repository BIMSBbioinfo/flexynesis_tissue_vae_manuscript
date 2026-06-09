import pandas as pd, numpy as np, torch, joblib
from pathlib import Path

MODEL_DIR = Path("model")
model = torch.load(MODEL_DIR / "vae_tissue.final_model.pth", map_location="cpu", weights_only=False)
model.eval()
art = joblib.load(MODEL_DIR / "vae_tissue.artifacts.joblib")
gene_list = list(art["feature_lists"]["gex"])
scaler = art["transforms"]["gex"]

df = pd.read_csv("test_5samples.csv", index_col=0)
gic = len(set(df.columns) & set(gene_list)); gir = len(set(df.index) & set(gene_list))
if gir > gic: df = df.T
overlap = len(set(df.columns) & set(gene_list))
print(f"Gene overlap: {overlap}/{len(gene_list)} ({100*overlap/len(gene_list):.1f}%)")

aligned = pd.DataFrame(0.0, index=df.index, columns=gene_list)
common = [g for g in gene_list if g in df.columns]
aligned[common] = df[common].values
aligned = aligned.fillna(0)

X = torch.tensor(scaler.transform(aligned.values), dtype=torch.float32)
with torch.no_grad():
    h = model.encoders[0](X)
    mu = model.FC_mean(h[0])
emb = mu.numpy()
print("Embeddings shape:", emb.shape)

with torch.no_grad():
    logits = model.MLPs["uberon_tissue"](mu)
    lm = model.dataset.label_mappings["uberon_tissue"]
    n2i = {v: k for k, v in lm.items()}
    if "nan" in n2i: logits[:, n2i["nan"]] = -1e9
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
labels = [lm[int(i)] for i in pred]
conf = probs.max(dim=1).values.numpy()
print(pd.DataFrame({"Sample": df.index, "Tissue": labels, "Conf": [f"{c:.1%}" for c in conf]}))
