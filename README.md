# GCN · Cora Node Classifier — Streamlit Frontend

Beautiful, dark-themed frontend for your 2-layer Graph Convolutional Network trained on the Cora dataset.

## File setup

```
your-project/
├── app.py
├── requirements.txt
├── gcn.onnx
└── gcn.onnx.data          ← must be named exactly gcn.onnx.data (not gcn_onnx.data)
```

> ⚠️ Rename `gcn_onnx.data` → `gcn.onnx.data` — ONNX expects the external data file
>    to be named `<model>.data`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push your repo with `app.py`, `requirements.txt`, `gcn.onnx`, `gcn.onnx.data`
2. Go to share.streamlit.io → New app → point to `app.py`
3. Done — Streamlit Cloud installs requirements automatically

## What the app does

| Feature | Detail |
|---|---|
| Demo graph | Generates a Cora-like graph with configurable nodes & edge density |
| Custom graph | Upload your own graph as JSON `{"features": [[...]], "edges": [[...]]}` |
| Graph view | Interactive Plotly graph coloured by predicted class |
| Node inspector | Per-node probability breakdown & neighbour list |
| Analytics | Class distribution, mean confidence, full node table |

## Model architecture

- **Input**: Node feature matrix (N × 1433 float32) + Edge index (2 × E int64)
- **Layer 1**: GCNConv (1433 → 32) + ReLU
- **Layer 2**: GCNConv (32 → 7)
- **Output**: Logits (N × 7) → softmax → predicted class
# gcn_cora
