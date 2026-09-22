"""
Bee Wing Origin Classifier - Simple UI
Upload a bee wing image to predict which region it is from.
"""

import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, JumpingKnowledge, global_mean_pool, global_max_pool

# Import gradio
try:
    import gradio as gr
except ImportError:
    gr = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model & Extraction Logic
# ---------------------------------------------------------------------------

def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    )

class WingGNN(nn.Module):
    def __init__(self, num_classes=5, node_dim=8, edge_dim=5, hidden=128, layers=4, dropout=0.3, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        
        self.convs = nn.ModuleList([
            GINEConv(make_mlp(hidden, hidden, hidden), train_eps=True, edge_dim=hidden)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.jk = JumpingKnowledge("cat")
        
        self.head = nn.Sequential(
            nn.Linear(hidden * layers * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        h = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        
        hs = []
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = F.dropout(F.relu(norm(conv(h, edge_index, e))), p=self.dropout, training=self.training) + h_in
            hs.append(h)
            
        h_all = self.jk(hs)
        pool = torch.cat([global_mean_pool(h_all, batch), global_max_pool(h_all, batch)], dim=1)
        return self.head(pool)


def load_model():
    """Loads model weights from available checkpoint."""
    ck_paths = ["wing_gnn_model.pt", "wing_gnn_superpixel.pt"]
    for ck_path in ck_paths:
        if os.path.exists(ck_path):
            try:
                ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
                classes = ck.get("classes", ['Austria', 'Greece', 'Hungary', 'Moldova', 'Morocco'])
                hparams = ck.get("hparams", {"num_classes": len(classes), "node_dim": 8, "edge_dim": 5, "hidden": 128, "layers": 4, "dropout": 0.3})
                # Filter out unsupported kwargs if any
                model = WingGNN(**{k: v for k, v in hparams.items() if k in ['num_classes', 'node_dim', 'edge_dim', 'hidden', 'layers', 'dropout']}).to(DEVICE)
                model.load_state_dict(ck["state_dict"])
                model.eval()
                print(f"Loaded trained model from: {ck_path}")
                return model, classes
            except Exception as e:
                print(f"Error loading {ck_path}: {e}")
    
    # Fallback to initialized model if no checkpoint
    classes = ['Austria', 'Greece', 'Hungary', 'Moldova', 'Morocco']
    model = WingGNN(num_classes=len(classes)).to(DEVICE)
    model.eval()
    print("Using freshly initialized WingGNN model (no checkpoint found).")
    return model, classes

MODEL, CLASSES = load_model()


def extract_graph_and_vis(img_input, max_side=800, n_segments=120, compactness=12.0):
    """Processes image array or path into graph data and visualization."""
    if isinstance(img_input, (str, Path)):
        gray = cv2.imread(str(img_input), cv2.IMREAD_GRAYSCALE)
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 3:
            gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_input
    else:
        raise ValueError("Unsupported image input")

    if gray is None:
        raise ValueError("Cannot read image")

    # Resize preserving aspect ratio
    h, w = gray.shape
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)

    # SLIC Superpixels directly on gray image (no silhouette mask needed)
    labels = slic(gray.astype(np.float64), n_segments=n_segments,
                  compactness=compactness, sigma=1.0, channel_axis=None,
                  start_label=1, enforce_connectivity=True)

    props = regionprops(labels, intensity_image=gray)
    if not props:
        raise RuntimeError("SLIC produced no superpixels")

    ids = [p.label for p in props]
    cy = np.array([p.centroid[0] for p in props], dtype=np.float32)
    cx = np.array([p.centroid[1] for p in props], dtype=np.float32)
    area = np.array([p.area for p in props], dtype=np.float32)
    total_area = float(gray.shape[0] * gray.shape[1])
    area_frac = (area / total_area) * 50.0

    mean_i = np.array([float(p.intensity_mean if hasattr(p, 'intensity_mean') else p.mean_intensity) for p in props], dtype=np.float32) / 255.0
    ecc = np.array([float(p.eccentricity) for p in props], dtype=np.float32)
    sol = np.array([float(p.solidity) for p in props], dtype=np.float32)
    std_i = ndi.standard_deviation(gray, labels, index=ids).astype(np.float32) / 255.0
    std_i = np.nan_to_num(std_i, nan=0.0)

    # Center & scale coordinates
    xy = np.stack([cx, cy], 1)
    xy = xy - xy.mean(0, keepdims=True)
    scale_norm = np.sqrt((xy ** 2).sum(1).mean())
    if scale_norm > 1e-8:
        xy = xy / scale_norm
    if len(xy) >= 3:
        cov = np.cov(xy.T)
        w_val, v = np.linalg.eigh(cov)
        xy = xy @ v[:, np.argsort(w_val)[::-1]]
        for k in range(2):
            if (xy[:, k] ** 3).sum() < 0:
                xy[:, k] *= -1.0
    r = np.linalg.norm(xy, axis=1, keepdims=True)

    # 8 node features
    x = np.concatenate([xy, r, area_frac[:, None], mean_i[:, None], std_i[:, None], ecc[:, None], sol[:, None]], axis=1)

    # Adjacency
    h_a, h_b = labels[:, :-1].ravel(), labels[:, 1:].ravel()
    hm = (h_a != h_b) & (h_a > 0) & (h_b > 0)
    v_a, v_b = labels[:-1, :].ravel(), labels[1:, :].ravel()
    vm = (v_a != v_b) & (v_a > 0) & (v_b > 0)
    pa = np.concatenate([h_a[hm], v_a[vm]])
    pb = np.concatenate([h_b[hm], v_b[vm]])
    lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
    n = int(labels.max()) + 1
    keys = lo.astype(np.int64) * n + hi.astype(np.int64)
    uniq, counts = np.unique(keys, return_counts=True)
    edges = np.stack([uniq // n, uniq % n], axis=1)
    edges0 = edges - 1

    d = xy[edges0[:, 1]] - xy[edges0[:, 0]]
    cdist = np.linalg.norm(d, axis=1)
    theta = np.arctan2(d[:, 1], d[:, 0])
    bstrength = np.clip(counts / max(counts.mean(), 1e-8), 0, 10)
    idiff = np.abs(mean_i[edges0[:, 1]] - mean_i[edges0[:, 0]])

    ef = np.stack([cdist, bstrength, idiff, np.cos(2 * theta), np.sin(2 * theta)], 1).astype(np.float32)
    edge_index = np.concatenate([edges0, edges0[:, ::-1]], axis=0).T.astype(np.int64)
    edge_attr = np.concatenate([ef, ef], axis=0).astype(np.float32)

    # Visualization
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    vis = mark_boundaries(rgb, labels, color=(1, 1, 0), mode="thick")
    vis = (vis * 255).astype(np.uint8)
    for a, b in edges0:
        cv2.line(vis, (int(cx[a]), int(cy[a])), (int(cx[b]), int(cy[b])), (0, 220, 255), 1)
    for i in range(len(cx)):
        cv2.circle(vis, (int(cx[i]), int(cy[i])), 3, (255, 50, 50), -1)

    data = Data(x=torch.from_numpy(x).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_attr).float())
    return data, vis


def predict(image):
    """Gradio prediction function."""
    if image is None:
        return "Please upload an image.", None, None

    try:
        data, vis_img = extract_graph_and_vis(image)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(DEVICE)

        with torch.no_grad():
            logits = MODEL(data)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Strictly focus on the 5 geographic classes (exclude 'Unrecognized' if present)
        valid_indices = [i for i, c in enumerate(CLASSES) if c.lower() != "unrecognized"]
        filtered_classes = [CLASSES[i] for i in valid_indices]
        filtered_probs = probs[valid_indices]
        if filtered_probs.sum() > 0:
            filtered_probs = filtered_probs / filtered_probs.sum()

        best_idx = int(np.argmax(filtered_probs))
        best_class = filtered_classes[best_idx]
        confidence = float(filtered_probs[best_idx])

        # Formatted result
        conf_dict = {filtered_classes[i]: float(filtered_probs[i]) for i in range(len(filtered_classes))}
        summary_text = f"Predicted Region: **{best_class}** ({confidence * 100:.1f}% confidence)"

        return summary_text, conf_dict, vis_img

    except Exception as e:
        return f"Error analyzing image: {str(e)}", None, None


# ---------------------------------------------------------------------------
# Launch UI
# ---------------------------------------------------------------------------

def create_ui():
    if gr is None:
        print("Gradio not installed.")
        return

    # Find sample images for quick testing (excluding Unrecognized)
    sample_images = []
    bees_dir = Path("bees")
    if bees_dir.exists():
        for c in sorted(bees_dir.iterdir()):
            if c.is_dir() and c.name.lower() != "unrecognized":
                imgs = list(c.glob("*.png")) + list(c.glob("*.jpg"))
                if imgs:
                    sample_images.append(str(imgs[0]))

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Upload Bee Forewing Image", type="filepath"),
        outputs=[
            gr.Markdown(label="Prediction Result"),
            gr.Label(num_top_classes=5, label="Region Probabilities"),
            gr.Image(label="Extracted Superpixel Graph (RAG)", type="numpy")
        ],
        title="🐝 Bee Wing Geographic Origin Classifier",
        description=(
            "Upload a bee forewing image to classify its geographic origin using a "
            "Superpixel Graph Neural Network (GNN). The model extracts a Region Adjacency "
            "Graph directly from the wing and classifies it into one of the 5 regions: "
            "**Austria**, **Greece**, **Hungary**, **Moldova**, or **Morocco**."
        ),
        examples=sample_images if sample_images else None,
        flagging_mode="never"
    )
    return demo

if __name__ == "__main__":
    demo = create_ui()
    if demo:
        print("Starting Gradio Web Server at http://127.0.0.1:7860 ...")
        theme = gr.themes.Soft() if hasattr(gr, "themes") else None
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=theme)

