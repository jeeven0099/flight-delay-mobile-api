"""
STEP 7 — DEPARTURE DELAY EVALUATION
=============================
Loads a trained checkpoint and evaluates departure-delay predictions.
Reports per-band MAE, RMSE, per-airline breakdown, per-route breakdown,
and delay severity buckets.

Usage:
  python 07_dashboard.py --checkpoint checkpoints/best_model.pt
  python 07_dashboard.py --checkpoint checkpoints/best_model_6k_ep16.pt
  python 07_dashboard.py --checkpoint checkpoints/best_model.pt --split val
"""

import os, argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

# ── PATHS ────────────────────────────────────────────────────────────────────
DRIVE_BASE     = "C:/Users/user/Desktop/Airline_Graphs_Project"
GRAPH_DATA_DIR = f"{DRIVE_BASE}/graph_data"
CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"
OUTPUT_DIR     = f"{DRIVE_BASE}/evaluation"

# ── MODEL CONFIG (must match training) ───────────────────────────────────────
HIDDEN_DIM      = 512
NUM_HEADS       = 8
NUM_GNN_LAYERS  = 2
GRU_HIDDEN_DIM  = 512
GRU_NUM_LAYERS  = 2
MLP_HIDDEN_DIM  = 256
TAIL_HIDDEN_DIM = 128
DROPOUT         = 0.0   # disabled at eval

NODE_TYPES = ["airport", "flight"]
EDGE_TYPES = [
    ("airport", "rotation",    "airport"),
    ("airport", "congestion",  "airport"),
    ("airport", "network",     "airport"),
    ("airport", "departs_to",  "flight"),
    ("airport", "arrives_from","flight"),
    ("flight",  "rotation",    "flight"),
    ("flight",  "departs_from","airport"),
    ("flight",  "arrives_at",  "airport"),
]

LABEL_HORIZONS_FL = [0, 1, 3, 6]
GATE_FEATURE_INDICES  = [0, 3, 4, 7, 11]
PARTIAL_GATE_FEATURES = [0, 7]
MASK_FULL_THRESHOLD    = 2.0 / 24
MASK_PARTIAL_THRESHOLD = 1.0 / 24
ORDINAL_THRESHOLDS     = [0.0, 15.0, 60.0, 120.0, 240.0, 720.0]
SEVERE_DELAY_THRESHOLD = 120.0
SEVERE_PROB_THRESHOLD  = 0.50
REGRESSION_TARGET_TRANSFORM = None
TAIL_UPLIFT_THRESHOLDS = [240.0, 720.0]


def threshold_col_name(thr):
    return f"p_ge_{int(thr)}"


def bucket_col_name(lo=None, hi=None):
    if lo is None:
        return f"p_bucket_lt_{int(hi)}"
    if hi is None:
        return f"p_bucket_ge_{int(lo)}"
    return f"p_bucket_{int(lo)}_{int(hi)}"


def ordinal_bucket_probs(raw_probs):
    probs = np.clip(np.asarray(raw_probs, dtype=np.float32), 0.0, 1.0)
    # Ordinal threshold probabilities should be non-increasing as the threshold
    # rises. Enforce that before deriving disjoint bucket masses.
    mono = np.minimum.accumulate(probs)

    out = {bucket_col_name(None, ORDINAL_THRESHOLDS[0]): float(1.0 - mono[0])}
    for i in range(len(ORDINAL_THRESHOLDS) - 1):
        lo = ORDINAL_THRESHOLDS[i]
        hi = ORDINAL_THRESHOLDS[i + 1]
        out[bucket_col_name(lo, hi)] = float(mono[i] - mono[i + 1])
    out[bucket_col_name(ORDINAL_THRESHOLDS[-1], None)] = float(mono[-1])
    return out


def model_target_to_delay(model_target):
    if REGRESSION_TARGET_TRANSFORM == "signed_log1p":
        return torch.sign(model_target) * torch.expm1(torch.abs(model_target))
    return model_target




# ════════════════════════════════════════════════════════════════════════════
# MODEL (identical to 06_train_gnn.py)
# ════════════════════════════════════════════════════════════════════════════

class FlightDelayGNN(nn.Module):
    def __init__(self, ap_in, fl_in, hidden, heads, layers,
                 gru_h, gru_layers, mlp_h, tail_h,
                 n_airports, n_tails, dropout=0.0, cls_out_dim=1,
                 use_tail_uplift=False,
                 tail_uplift_thresholds=TAIL_UPLIFT_THRESHOLDS,
                 tail_uplift_detach_gates=True):
        super().__init__()
        self.hidden_dim     = hidden
        self.gru_hidden     = gru_h
        self.gru_num_layers = gru_layers
        self.num_airports   = n_airports
        self.num_tails      = n_tails
        self.tail_hidden    = tail_h
        self.use_tail_uplift = use_tail_uplift
        self.tail_uplift_thresholds = [float(x) for x in tail_uplift_thresholds]
        self.tail_uplift_detach_gates = tail_uplift_detach_gates
        self.tail_uplift_indices = [
            ORDINAL_THRESHOLDS.index(float(thr))
            for thr in self.tail_uplift_thresholds
            if float(thr) in ORDINAL_THRESHOLDS
        ]

        self.ap_proj = nn.Sequential(Linear(ap_in, hidden), nn.LayerNorm(hidden))
        self.fl_proj = nn.Sequential(Linear(fl_in, hidden), nn.LayerNorm(hidden))
        self.rotation_gate = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        meta = (NODE_TYPES, EDGE_TYPES)
        self.convs    = nn.ModuleList([HGTConv(hidden, hidden, meta, heads=heads)
                                       for _ in range(layers)])
        self.ap_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.fl_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drops    = nn.ModuleList([nn.Dropout(dropout) for _ in range(layers)])

        self.ap_gru = nn.GRU(hidden, gru_h, num_layers=gru_layers, batch_first=False)
        self.ap_context_proj = nn.Linear(gru_h, hidden)
        self.tail_gru  = nn.GRUCell(hidden, tail_h)
        self.tail_proj = nn.Linear(tail_h, hidden)
        self.fl_fuse = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.fl_gate = nn.Sequential(nn.Linear(hidden*3, hidden), nn.Sigmoid())
        self.ap_head = nn.Sequential(
            nn.Linear(gru_h, mlp_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_h, mlp_h//2), nn.ReLU(), nn.Linear(mlp_h//2, 1))
        self.fl_head = nn.Sequential(
            nn.Linear(hidden, mlp_h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_h, mlp_h//2), nn.ReLU(), nn.Linear(mlp_h//2, 1))
        self.fl_cls = nn.Sequential(
            nn.Linear(hidden, mlp_h//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_h//2, cls_out_dim))
        if self.use_tail_uplift and self.tail_uplift_indices:
            self.fl_tail_uplift = nn.Sequential(
                nn.Linear(hidden, mlp_h//2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(mlp_h//2, len(self.tail_uplift_indices)))
        else:
            self.fl_tail_uplift = None

    def forward(self, data, ap_h, tail_h=None):
        n_fl = data["flight"].num_nodes
        dev  = ap_h.device
        if tail_h is None:
            tail_h = {}

        x_ap = self.ap_proj(data["airport"].x.float())
        x_fl = (self.fl_proj(data["flight"].x.float()) if n_fl > 0
                else torch.zeros(0, self.hidden_dim, device=dev))
        x = {"airport": x_ap, "flight": x_fl}

        rot_et = ("flight","rotation","flight")
        if (n_fl > 0 and hasattr(data[rot_et], "edge_index")
                and data[rot_et].edge_index.shape[1] > 0):
            rei = data[rot_et].edge_index
            rea = data[rot_et].edge_attr.float()
            if rea.shape[1] >= 2:
                g = self.rotation_gate(rea[:,:2])
                m = g * x["flight"][rei[0]]
                x["flight"] = x["flight"].clone()
                x["flight"].scatter_add_(
                    0, rei[1].unsqueeze(-1).expand_as(m), m)

        eid = {et: data[et].edge_index
               for et in EDGE_TYPES if et != rot_et
               and hasattr(data[et], "edge_index")
               and data[et].edge_index.shape[1] > 0}

        for i, conv in enumerate(self.convs):
            xn = conv(x, eid)
            if "airport" in xn:
                x["airport"] = self.ap_norms[i](
                    self.drops[i](xn["airport"]) + x["airport"])
            if "flight" in xn and n_fl > 0:
                x["flight"] = self.fl_norms[i](
                    self.drops[i](xn["flight"]) + x["flight"])

        _, ap_h_new = self.ap_gru(x["airport"].unsqueeze(0), ap_h)

        new_tail = dict(tail_h)
        if n_fl > 0:
            emb = x["flight"]
            ap_ctx_all = self.ap_context_proj(ap_h_new[-1])
            ap_context  = torch.zeros(n_fl, self.hidden_dim, device=dev)
            dep_et = ("flight","departs_from","airport")
            if hasattr(data[dep_et], "edge_index") and data[dep_et].edge_index.shape[1] > 0:
                fl_idx = data[dep_et].edge_index[0]
                ap_idx = data[dep_et].edge_index[1]
                ap_context[fl_idx] = ap_ctx_all[ap_idx]

            tail_context = torch.zeros(n_fl, self.hidden_dim, device=dev)
            if hasattr(data["flight"], "tail_id") and data["flight"].tail_id is not None:
                tids = data["flight"].tail_id.tolist()
                prev_h = torch.stack([
                    tail_h.get(int(tid), torch.zeros(self.tail_hidden, device=dev))
                    for tid in tids])
                new_h = self.tail_gru(emb, prev_h)
                tail_context = self.tail_proj(new_h)
                with torch.no_grad():
                    nh_det = new_h.detach()
                    for j, tid in enumerate(tids):
                        new_tail[int(tid)] = nh_det[j]

            cat    = torch.cat([emb, ap_context, tail_context], dim=-1)
            fl_out = self.fl_gate(cat) * self.fl_fuse(cat) + emb
            fl_pred_z = self.fl_head(fl_out).squeeze(-1)
            fl_logits = self.fl_cls(fl_out)
            if self.fl_tail_uplift is not None and fl_logits.ndim > 1:
                uplift_mag = F.softplus(self.fl_tail_uplift(fl_out))
                tail_probs = torch.sigmoid(fl_logits[:, self.tail_uplift_indices])
                fl_pred_z = fl_pred_z + (tail_probs * uplift_mag).sum(dim=-1)
            fl_pred = model_target_to_delay(fl_pred_z)
            if fl_logits.shape[-1] == 1:
                fl_logits = fl_logits.squeeze(-1)
        else:
            fl_pred = torch.zeros(0, device=dev)
            fl_logits = torch.zeros(0, self.fl_cls[-1].out_features, device=dev)

        return fl_pred, fl_logits, ap_h_new, new_tail

    def init_hidden(self, device):
        return (torch.zeros(self.gru_num_layers, self.num_airports,
                            self.gru_hidden, device=device), {})


# ════════════════════════════════════════════════════════════════════════════
# MASKING
# ════════════════════════════════════════════════════════════════════════════

def apply_masking(x_fl):
    if x_fl.shape[0] == 0:
        return x_fl
    x   = x_fl.float().clone()
    t2d = x[:, 14]
    fm  = t2d > MASK_FULL_THRESHOLD
    pm  = (t2d > MASK_PARTIAL_THRESHOLD) & ~fm
    for i in GATE_FEATURE_INDICES:
        x[fm, i] = 0.0
    for i in PARTIAL_GATE_FEATURES:
        x[pm, i] = 0.0
    return x.half()


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, snapshots, device, static_edges=None,
             include_ordinal_detail=False):
    """
    Run full sequential evaluation over all snapshots.
    Collects per-flight predictions with metadata for detailed analysis.
    Returns a DataFrame with one row per (flight, horizon_band).
    """
    model.eval()
    nw_ei = static_edges["network_ei"].to(device) if static_edges else None
    nw_ea = static_edges["network_ea"].to(device) if static_edges else None

    ap_h, tail_h = model.init_hidden(device)
    records = []

    with torch.no_grad():
        for si, snap in enumerate(snapshots):
            snap = snap.clone().to(device)
            if static_edges is not None:
                snap["airport","network","airport"].edge_index = nw_ei
                snap["airport","network","airport"].edge_attr  = nw_ea
            if snap["flight"].num_nodes > 0:
                snap["flight"].x = apply_masking(snap["flight"].x)

            fl_pred, fl_logits, ap_h, tail_h = model(snap, ap_h, tail_h)
            ap_h   = ap_h.detach()
            tail_h = {k: v.detach() for k, v in tail_h.items()}

            if snap["flight"].num_nodes == 0:
                continue

            fl_y   = snap["flight"].y.cpu().numpy()
            preds  = fl_pred.cpu().numpy()
            ordinal_prob_matrix = None
            if fl_logits.ndim == 1:
                severe_probs = torch.sigmoid(fl_logits).cpu().numpy()
            else:
                ordinal_prob_matrix = torch.sigmoid(fl_logits).cpu().numpy()
                severe_idx = (ORDINAL_THRESHOLDS.index(SEVERE_DELAY_THRESHOLD)
                              if SEVERE_DELAY_THRESHOLD in ORDINAL_THRESHOLDS
                              else fl_logits.shape[1] - 1)
                severe_probs = ordinal_prob_matrix[:, severe_idx]
            fids   = (snap["flight"].flight_id.cpu().numpy()
                      if hasattr(snap["flight"], "flight_id") else
                      np.arange(snap["flight"].num_nodes))

            for h in LABEL_HORIZONS_FL:
                m = getattr(snap["flight"], f"y_mask_{h}h", None)
                if m is None or m.sum() == 0:
                    continue
                m_np = m.cpu().numpy()
                for j in np.where(m_np)[0]:
                    rec = {
                        "flight_id":  int(fids[j]),
                        "snap_idx":   si,
                        "horizon_h":  h,
                        "pred":       float(preds[j]),
                        "severe_prob": float(severe_probs[j]),
                        "actual":     float(fl_y[j]),
                        "abs_err":    abs(float(preds[j]) - float(fl_y[j])),
                        "sq_err":     (float(preds[j]) - float(fl_y[j]))**2,
                    }
                    if include_ordinal_detail and ordinal_prob_matrix is not None:
                        raw_probs = ordinal_prob_matrix[j]
                        for idx, thr in enumerate(ORDINAL_THRESHOLDS[:raw_probs.shape[0]]):
                            rec[threshold_col_name(thr)] = float(raw_probs[idx])
                        rec.update(ordinal_bucket_probs(raw_probs[:len(ORDINAL_THRESHOLDS)]))
                    records.append(rec)

            if (si + 1) % 1000 == 0:
                print(f"  {si+1:,}/{len(snapshots):,} snapshots "
                      f"({len(records):,} predictions)")

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════

def binary_pr(actual_mask, pred_mask):
    tp = int((actual_mask & pred_mask).sum())
    fp = int((~actual_mask & pred_mask).sum())
    fn = int((actual_mask & ~pred_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, int(pred_mask.sum()), int(actual_mask.sum())


def print_summary(df, label="TEST", class_target_threshold=None):
    print(f"\n{'='*60}")
    print(f"  {label} SET RESULTS")
    print(f"{'='*60}")

    # Overall
    print(f"\n  Overall (all bands, all flights):")
    print(f"    MAE  : {df['abs_err'].mean():.3f} min")
    print(f"    RMSE : {np.sqrt(df['sq_err'].mean()):.3f} min")
    print(f"    N    : {len(df):,} flight-band predictions")

    # Per horizon band
    print(f"\n  Per horizon band:")
    band_names = {0: "0h  [<1h before dep]",
                  1: "1h  [1-3h before dep]",
                  3: "3h  [3-6h before dep]",
                  6: "6h  [>6h before dep] ← novel"}
    for h in LABEL_HORIZONS_FL:
        sub = df[df["horizon_h"] == h]
        if len(sub) == 0:
            continue
        mae  = sub["abs_err"].mean()
        rmse = np.sqrt(sub["sq_err"].mean())
        n    = len(sub)
        print(f"    {band_names[h]:35s}  MAE={mae:.2f}  RMSE={rmse:.2f}  N={n:,}")

    # Delay severity buckets
    print(f"\n  By actual departure-delay severity (6h band only):")
    df6 = df[df["horizon_h"] == 6].copy()
    buckets = [
        ("On time  (<0 min)",   df6["actual"] < 0),
        ("Minor   (0-15 min)",  (df6["actual"] >= 0)  & (df6["actual"] < 15)),
        ("Moderate(15-60 min)", (df6["actual"] >= 15) & (df6["actual"] < 60)),
        ("Heavy   (60-120 min)", (df6["actual"] >= 60) & (df6["actual"] < 120)),
        ("Severe  (120-240 min)", (df6["actual"] >= 120) & (df6["actual"] < 240)),
        ("Extreme (240-720 min)", (df6["actual"] >= 240) & (df6["actual"] < 720)),
        ("Ultra   (>=720 min)",   df6["actual"] >= 720),
    ]
    for name, mask in buckets:
        sub = df6[mask]
        if len(sub) == 0: continue
        print(f"    {name:25s}  MAE={sub['abs_err'].mean():.2f}  N={len(sub):,}")

    severe_actual_all = df["actual"] >= SEVERE_DELAY_THRESHOLD
    severe_pred_reg_all = df["pred"] >= SEVERE_DELAY_THRESHOLD
    reg_p, reg_r, reg_pos, act_pos = binary_pr(severe_actual_all.to_numpy(),
                                               severe_pred_reg_all.to_numpy())
    print(f"\n  Severe-delay detection (>={SEVERE_DELAY_THRESHOLD:.0f} min, all bands):")
    print(f"    Regression [pred>=120]      Precision={reg_p:.3f}  "
          f"Recall={reg_r:.3f}  Pred+={reg_pos:,}  Actual+={act_pos:,}")

    if class_target_threshold == SEVERE_DELAY_THRESHOLD and "severe_prob" in df.columns:
        severe_pred_cls_all = df["severe_prob"] >= SEVERE_PROB_THRESHOLD
        cls_p, cls_r, cls_pos, _ = binary_pr(severe_actual_all.to_numpy(),
                                             severe_pred_cls_all.to_numpy())
        print(f"    Classifier [p>=0.50]       Precision={cls_p:.3f}  "
              f"Recall={cls_r:.3f}  Pred+={cls_pos:,}  Actual+={act_pos:,}")

    if len(df6) > 0:
        severe_actual_6h = df6["actual"] >= SEVERE_DELAY_THRESHOLD
        severe_pred_reg_6h = df6["pred"] >= SEVERE_DELAY_THRESHOLD
        reg6_p, reg6_r, reg6_pos, act6_pos = binary_pr(severe_actual_6h.to_numpy(),
                                                       severe_pred_reg_6h.to_numpy())
        print(f"\n  Severe-delay detection (>={SEVERE_DELAY_THRESHOLD:.0f} min, 6h band):")
        print(f"    Regression [pred>=120]      Precision={reg6_p:.3f}  "
              f"Recall={reg6_r:.3f}  Pred+={reg6_pos:,}  Actual+={act6_pos:,}")

        if class_target_threshold == SEVERE_DELAY_THRESHOLD and "severe_prob" in df6.columns:
            severe_pred_cls_6h = df6["severe_prob"] >= SEVERE_PROB_THRESHOLD
            cls6_p, cls6_r, cls6_pos, _ = binary_pr(severe_actual_6h.to_numpy(),
                                                    severe_pred_cls_6h.to_numpy())
            print(f"    Classifier [p>=0.50]       Precision={cls6_p:.3f}  "
                  f"Recall={cls6_r:.3f}  Pred+={cls6_pos:,}  Actual+={act6_pos:,}")

    return df


def print_breakdowns(df, lookup):
    """Per-airline and per-route breakdowns using flight_lookup.parquet."""
    keep_cols = ["flight_id", "ORIGIN", "DEST", "Operating_Airline"]
    keep_cols = [c for c in keep_cols if c in lookup.columns]
    if "flight_id" not in keep_cols:
        print("\n  Flight lookup missing flight_id; skipping airline/route breakdowns.")
        return

    merged = df.merge(lookup[keep_cols], on="flight_id", how="left")

    # Per-airline MAE at 6h
    df6 = merged[merged["horizon_h"] == 6]
    if "Operating_Airline" in df6.columns and df6["Operating_Airline"].notna().any():
        print(f"\n  Per-airline MAE (6h band, top 10 by flight count):")
        al = (df6.groupby("Operating_Airline")
                 .agg(mae=("abs_err","mean"), n=("abs_err","count"))
                 .sort_values("n", ascending=False).head(10))
        for airline, row in al.iterrows():
            print(f"    {airline:6s}  MAE={row['mae']:.2f}  N={int(row['n']):,}")

    # Top 10 worst routes at 6h
    if "ORIGIN" in df6.columns and df6["ORIGIN"].notna().any():
        print(f"\n  Top 10 hardest routes (6h band, min 100 flights):")
        rt = (df6.groupby(["ORIGIN","DEST"])
                 .agg(mae=("abs_err","mean"), n=("abs_err","count"))
                 .reset_index())
        rt = rt[rt["n"] >= 100].sort_values("mae", ascending=False).head(10)
        for _, row in rt.iterrows():
            print(f"    {row['ORIGIN']}→{row['DEST']:6s}  "
                  f"MAE={row['mae']:.2f}  N={int(row['n']):,}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--split", default="test",
                        choices=["test","val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--drive_base", default=None,
                        help="Override DRIVE_BASE path")
    parser.add_argument("--save_ordinal_detail", action="store_true",
                        help="Save full ordinal threshold and bucket probabilities "
                             "to the output CSV. Leave off for memory-safe full-val runs.")
    args = parser.parse_args()

    global DRIVE_BASE, GRAPH_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR
    global ORDINAL_THRESHOLDS, REGRESSION_TARGET_TRANSFORM, TAIL_UPLIFT_THRESHOLDS
    if args.drive_base:
        DRIVE_BASE     = args.drive_base
        GRAPH_DATA_DIR = f"{DRIVE_BASE}/graph_data"
        CHECKPOINT_DIR = f"{DRIVE_BASE}/checkpoints"
        OUTPUT_DIR     = f"{DRIVE_BASE}/evaluation"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_path)
    print(f"\nLoading checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  Trained epoch : {ck.get('epoch', '?')}")
    m = ck.get("metrics", {})
    class_target_threshold = ck.get("class_target_threshold")
    if ck.get("ordinal_thresholds") is not None:
        ORDINAL_THRESHOLDS = [float(x) for x in ck["ordinal_thresholds"]]
    REGRESSION_TARGET_TRANSFORM = ck.get("regression_target_transform")
    use_tail_uplift = ck.get("use_tail_uplift")
    if use_tail_uplift is None:
        use_tail_uplift = any(k.startswith("fl_tail_uplift.") for k in ck["model_state"].keys())
    TAIL_UPLIFT_THRESHOLDS = [
        float(x) for x in ck.get("tail_uplift_thresholds", TAIL_UPLIFT_THRESHOLDS)
    ]
    tail_uplift_detach_gates = ck.get("tail_uplift_detach_gates", True)
    cls_out_dim = ck.get("cls_out_dim")
    if cls_out_dim is None:
        w = ck["model_state"].get("fl_cls.3.weight")
        cls_out_dim = int(w.shape[0]) if w is not None else 1
    print(f"  Val ckpt      : {m.get('val_ckpt', float('nan')):.3f}")
    print(f"  Val 6h MAE    : {m.get('val_6h',   float('nan')):.3f}")
    print(f"  Val 3h MAE    : {m.get('val_3h',   float('nan')):.3f}")
    print(f"  Val 1h MAE    : {m.get('val_1h',   float('nan')):.3f}")
    if class_target_threshold is not None:
        print(f"  Cls target    : >= {class_target_threshold:.0f} min")
    if cls_out_dim > 1:
        print(f"  Ordinal bins  : >= {ORDINAL_THRESHOLDS}")
    if REGRESSION_TARGET_TRANSFORM:
        print(f"  Reg target    : {REGRESSION_TARGET_TRANSFORM}")
    if use_tail_uplift:
        print(f"  Tail uplift   : >= {TAIL_UPLIFT_THRESHOLDS}")

    # Load snapshots
    split_name = args.split
    print(f"\nLoading {split_name} snapshots ...")
    snaps = torch.load(
        os.path.join(GRAPH_DATA_DIR, f"snapshots_{split_name}.pt"),
        map_location="cpu", weights_only=False)
    print(f"  {len(snaps):,} snapshots")

    print("\nLoading static edges ...")
    static = torch.load(
        os.path.join(GRAPH_DATA_DIR, "static_edges.pt"),
        map_location="cpu", weights_only=False)

    # Build model from checkpoint metadata
    n_ap  = ck.get("num_airports", snaps[0]["airport"].num_nodes)
    ap_in = snaps[0]["airport"].x.shape[1]
    fl_in = (snaps[0]["flight"].x.shape[1]
             if snaps[0]["flight"].num_nodes > 0 else 19)
    n_tails = ck.get("num_tails", 12000)

    model = FlightDelayGNN(
        ap_in=ap_in, fl_in=fl_in,
        hidden=ck.get("hidden_dim", HIDDEN_DIM),
        heads=NUM_HEADS, layers=NUM_GNN_LAYERS,
        gru_h=ck.get("gru_hidden", GRU_HIDDEN_DIM),
        gru_layers=ck.get("gru_num_layers", GRU_NUM_LAYERS),
        mlp_h=MLP_HIDDEN_DIM,
        tail_h=ck.get("tail_hidden", TAIL_HIDDEN_DIM),
        n_airports=n_ap, n_tails=n_tails, dropout=0.0,
        cls_out_dim=cls_out_dim,
        use_tail_uplift=use_tail_uplift,
        tail_uplift_thresholds=TAIL_UPLIFT_THRESHOLDS,
        tail_uplift_detach_gates=tail_uplift_detach_gates,
    ).to(device)
    model.load_state_dict(ck["model_state"])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load flight lookup for breakdowns
    lookup_path = os.path.join(GRAPH_DATA_DIR, "flight_lookup.parquet")
    lookup = None
    if os.path.exists(lookup_path):
        lookup = pd.read_parquet(lookup_path)
        print(f"  Flight lookup: {len(lookup):,} rows")

    # Run evaluation
    print(f"\nEvaluating on {split_name} set ...")
    t0 = time.time()
    df = evaluate(
        model, snaps, device, static,
        include_ordinal_detail=args.save_ordinal_detail,
    )
    print(f"  Done in {time.time()-t0:.1f}s | {len(df):,} predictions")

    # Print results
    print_summary(df, label=split_name.upper(),
                  class_target_threshold=class_target_threshold)

    # Save results
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path  = os.path.join(OUTPUT_DIR, f"eval_{ckpt_name}_{split_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Predictions saved → {out_path}")

    if lookup is not None:
        print_breakdowns(df, lookup)

    # Summary CSV for easy comparison across models
    summary = {
        "checkpoint": ckpt_name, "split": split_name,
        "epoch": ck.get("epoch", "?"),
        "overall_mae":  round(df["abs_err"].mean(), 3),
        "overall_rmse": round(np.sqrt(df["sq_err"].mean()), 3),
    }
    for h in LABEL_HORIZONS_FL:
        sub = df[df["horizon_h"] == h]
        summary[f"mae_{h}h"] = round(sub["abs_err"].mean(), 3) if len(sub) else None
        summary[f"rmse_{h}h"]= round(np.sqrt(sub["sq_err"].mean()), 3) if len(sub) else None

    severe_all = df["actual"] >= SEVERE_DELAY_THRESHOLD
    severe_pred_reg = df["pred"] >= SEVERE_DELAY_THRESHOLD
    reg_p, reg_r, _, _ = binary_pr(severe_all.to_numpy(), severe_pred_reg.to_numpy())
    summary["severe_threshold_min"] = SEVERE_DELAY_THRESHOLD
    summary["severe_reg_precision"] = round(reg_p, 3)
    summary["severe_reg_recall"] = round(reg_r, 3)

    df6 = df[df["horizon_h"] == 6]
    if len(df6):
        severe_6h = df6["actual"] >= SEVERE_DELAY_THRESHOLD
        summary["severe_mae_6h"] = round(df6[severe_6h]["abs_err"].mean(), 3) if severe_6h.any() else None
    else:
        summary["severe_mae_6h"] = None

    if class_target_threshold == SEVERE_DELAY_THRESHOLD and "severe_prob" in df.columns:
        severe_pred_cls = df["severe_prob"] >= SEVERE_PROB_THRESHOLD
        cls_p, cls_r, _, _ = binary_pr(severe_all.to_numpy(), severe_pred_cls.to_numpy())
        summary["severe_cls_precision"] = round(cls_p, 3)
        summary["severe_cls_recall"] = round(cls_r, 3)
    else:
        summary["severe_cls_precision"] = None
        summary["severe_cls_recall"] = None

    summary_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    summary_df   = pd.DataFrame([summary])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Summary appended → {summary_path}")


if __name__ == "__main__":
    main()
