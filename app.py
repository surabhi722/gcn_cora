import streamlit as st
import numpy as np
import onnxruntime as ort
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import os
import shutil

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GCN · Cora Node Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CORA CLASS LABELS ───────────────────────────────────────────────────────
CORA_CLASSES = {
    0: "Case Based",
    1: "Genetic Algorithms",
    2: "Neural Networks",
    3: "Probabilistic Methods",
    4: "Reinforcement Learning",
    5: "Rule Learning",
    6: "Theory",
}

CLASS_COLORS = [
    "#6366F1",  # indigo   - Case Based
    "#22D3EE",  # cyan     - Genetic Algorithms
    "#F59E0B",  # amber    - Neural Networks
    "#10B981",  # emerald  - Probabilistic Methods
    "#EF4444",  # red      - Reinforcement Learning
    "#A855F7",  # purple   - Rule Learning
    "#F97316",  # orange   - Theory
]

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark page background */
[data-testid="stAppViewContainer"] {
    background: #0D1117;
}
[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid #30363D;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: #8B949E;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #818CF8;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    flex: 1;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.75rem;
    color: #8B949E;
    margin-top: 0.3rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Section heading */
.section-head {
    font-size: 0.8rem;
    font-weight: 600;
    color: #8B949E;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 0.75rem 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Result card */
.result-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.result-class {
    font-size: 1.5rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0.4rem 0 0.2rem 0;
}
.result-conf {
    font-size: 0.85rem;
    color: #8B949E;
}
.conf-bar-bg {
    background: #21262D;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #6366F1, #818CF8);
}

/* Node pill */
.node-pill {
    display: inline-block;
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #8B949E;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* Info box */
.info-box {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    color: #8B949E;
    line-height: 1.6;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #8B949E !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    color: #E6EDF3 !important;
}
[data-testid="stSidebar"] h2 {
    color: #E6EDF3 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Class legend */
.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 0.8rem;
    color: #8B949E;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Streamlit button override */
.stButton > button {
    background: #6366F1 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    border: none !important;
}

/* Number input */
.stNumberInput input {
    background: #21262D !important;
    color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] {
    background: #30363D;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-bottom: 1px solid #30363D;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8B949E;
    border-bottom: 2px solid transparent;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: #818CF8 !important;
    border-bottom: 2px solid #6366F1 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load GCN ONNX model — copies both files to a temp dir with correct naming."""
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    shutil.copy("/tmp/gcn.onnx",      os.path.join(tmp, "gcn.onnx"))
    shutil.copy("/tmp/gcn.onnx.data", os.path.join(tmp, "gcn.onnx.data"))
    sess = ort.InferenceSession(os.path.join(tmp, "gcn.onnx"))
    return sess

@st.cache_data
def load_cora_sample():
    """Generate a realistic Cora-like sample graph (or load real data if present)."""
    np.random.seed(42)
    n = 80   # nodes for visualisation
    # Sparse BoW features (Cora: 1433 dims, binary)
    features = (np.random.rand(n, 1433) < 0.012).astype(np.float32)
    # Community edges — 7 clusters
    labels_true = np.random.randint(0, 7, n)
    edges_src, edges_dst = [], []
    for i in range(n):
        # Intra-cluster edges (higher prob)
        same = np.where(labels_true == labels_true[i])[0]
        diff = np.where(labels_true != labels_true[i])[0]
        for j in same:
            if j != i and np.random.rand() < 0.35:
                edges_src.append(i); edges_dst.append(j)
        for j in diff:
            if j != i and np.random.rand() < 0.03:
                edges_src.append(i); edges_dst.append(j)
    edges = np.array([edges_src, edges_dst], dtype=np.int64)
    return features, edges, n


# ─── INFERENCE ───────────────────────────────────────────────────────────────
def run_inference(sess, features, edges):
    logits = sess.run(None, {
        "mode_features": features,
        "edge_indices": edges
    })[0]
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    return probs, preds

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─── GRAPH PLOT ──────────────────────────────────────────────────────────────
def build_graph_figure(features, edges, preds, probs, selected_node=None):
    n = features.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_pairs = list(zip(edges[0].tolist(), edges[1].tolist()))
    G.add_edges_from(edge_pairs)

    pos = nx.spring_layout(G, seed=7, k=1.8/np.sqrt(n))

    # Edges
    xe, ye = [], []
    for u, v in G.edges():
        xe += [pos[u][0], pos[v][0], None]
        ye += [pos[u][1], pos[v][1], None]

    fig = go.Figure()

    # Edge trace
    fig.add_trace(go.Scatter(
        x=xe, y=ye, mode='lines',
        line=dict(color='rgba(255,255,255,0.06)', width=0.8),
        hoverinfo='none', showlegend=False
    ))

    # Node traces per class
    for cls in range(7):
        idx = np.where(preds == cls)[0]
        if len(idx) == 0:
            continue
        xs = [pos[i][0] for i in idx]
        ys = [pos[i][1] for i in idx]
        confs = [f"{probs[i, cls]*100:.1f}%" for i in idx]
        sizes = [12 + probs[i, cls] * 16 for i in idx]

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            name=CORA_CLASSES[cls],
            marker=dict(
                color=CLASS_COLORS[cls],
                size=sizes,
                opacity=0.85,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            customdata=[[i, confs[j]] for j, i in enumerate(idx)],
            hovertemplate=(
                "<b>Node %{customdata[0]}</b><br>"
                f"Class: {CORA_CLASSES[cls]}<br>"
                "Confidence: %{customdata[1]}<extra></extra>"
            )
        ))

    # Highlight selected node
    if selected_node is not None and selected_node < n:
        sx, sy = pos[selected_node]
        fig.add_trace(go.Scatter(
            x=[sx], y=[sy], mode='markers',
            marker=dict(
                color='white', size=22, opacity=1,
                line=dict(color=CLASS_COLORS[preds[selected_node]], width=3)
            ),
            hoverinfo='none', showlegend=False
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(22,27,34,0.8)',
            bordercolor='#30363D',
            borderwidth=1,
            font=dict(color='#8B949E', size=11),
            itemsizing='constant'
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=480,
    )
    return fig


def build_confidence_chart(probs_row):
    classes = [CORA_CLASSES[i] for i in range(7)]
    values  = (probs_row * 100).tolist()
    fig = go.Figure(go.Bar(
        x=values, y=classes,
        orientation='h',
        marker=dict(color=CLASS_COLORS, opacity=0.85),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#8B949E', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='#21262D', color='#8B949E',
                   ticksuffix='%', range=[0, max(values)*1.22]),
        yaxis=dict(color='#8B949E', tickfont=dict(size=11)),
        height=260,
    )
    return fig


def build_dist_chart(preds):
    counts = {CORA_CLASSES[c]: int(np.sum(preds == c)) for c in range(7)}
    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker=dict(color=CLASS_COLORS, opacity=0.85),
        text=list(counts.values()),
        textposition='outside',
        textfont=dict(color='#8B949E', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=80),
        xaxis=dict(color='#8B949E', tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(color='#8B949E', gridcolor='#21262D'),
        height=260,
    )
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    mode = st.radio(
        "Input mode",
        ["Demo graph (sample Cora)", "Custom graph (JSON upload)"],
        label_visibility="visible"
    )

    st.markdown("---")

    if mode == "Demo graph (sample Cora)":
        n_nodes = st.slider("Number of nodes", 20, 120, 80, 10)
        edge_density = st.slider("Edge density", 0.05, 0.60, 0.30, 0.05,
                                  help="Fraction of intra-cluster edges added")
        seed = st.slider("Graph seed", 0, 99, 42)
        run_btn = st.button("Run Inference →")
    else:
        st.markdown(
            '<div class="info-box">Upload a JSON with keys <code>features</code> '
            '(N×1433 list) and <code>edges</code> (2×E list).</div>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader("Graph JSON", type="json")
        run_btn = st.button("Run Inference →")

    st.markdown("---")
    st.markdown("**Class legend**", unsafe_allow_html=False)
    for i, (cls, col) in enumerate(zip(CORA_CLASSES.values(), CLASS_COLORS)):
        st.markdown(
            f'<div class="legend-item"><div class="legend-dot" style="background:{col}"></div>{cls}</div>',
            unsafe_allow_html=True
        )


# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">GCN · CORA DATASET</div>
  <h1>Graph Convolutional Network</h1>
  <p>Two-layer GCN trained on the Cora citation network · 2708 papers · 1433 BoW features · 7 research classes<br>
     Runs entirely in-browser via ONNX Runtime — no GPU required.</p>
</div>
""", unsafe_allow_html=True)

# ─── STATS ROW ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-value">2 708</div><div class="stat-label">Cora nodes (papers)</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-value">5 429</div><div class="stat-label">Citation edges</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-value">1 433</div><div class="stat-label">BoW features</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-value">7</div><div class="stat-label">Research classes</div></div>', unsafe_allow_html=True)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
try:
    sess = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}\n\nMake sure gcn.onnx and gcn.onnx.data are in /tmp/")
    st.stop()


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "features" not in st.session_state:
    st.session_state.features = None
    st.session_state.edges    = None
    st.session_state.probs    = None
    st.session_state.preds    = None


# ─── RUN INFERENCE ────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running GCN inference…"):
        if mode == "Demo graph (sample Cora)":
            np.random.seed(seed)
            n = n_nodes
            features = (np.random.rand(n, 1433) < 0.012).astype(np.float32)
            labels_true = np.random.randint(0, 7, n)
            esrc, edst = [], []
            for i in range(n):
                same = np.where(labels_true == labels_true[i])[0]
                diff = np.where(labels_true != labels_true[i])[0]
                for j in same:
                    if j != i and np.random.rand() < edge_density:
                        esrc.append(i); edst.append(j)
                for j in diff:
                    if j != i and np.random.rand() < max(0.01, edge_density * 0.08):
                        esrc.append(i); edst.append(j)
            edges = np.array([esrc, edst], dtype=np.int64)
        else:
            if uploaded is None:
                st.warning("Please upload a JSON file first.")
                st.stop()
            data = json.load(uploaded)
            features = np.array(data["features"], dtype=np.float32)
            edges    = np.array(data["edges"],    dtype=np.int64)

        probs, preds = run_inference(sess, features, edges)
        st.session_state.features = features
        st.session_state.edges    = edges
        st.session_state.probs    = probs
        st.session_state.preds    = preds

    st.success(f"Classified {features.shape[0]} nodes across 7 research categories.", icon="✅")


# ─── RESULTS ─────────────────────────────────────────────────────────────────
if st.session_state.probs is not None:
    probs   = st.session_state.probs
    preds   = st.session_state.preds
    features= st.session_state.features
    edges   = st.session_state.edges
    n       = features.shape[0]

    tab1, tab2, tab3 = st.tabs(["  Graph View  ", "  Node Inspector  ", "  Analytics  "])

    # TAB 1: Graph
    with tab1:
        sel_node = st.number_input("Highlight node (ID)", 0, n-1, 0, label_visibility="visible")
        fig_graph = build_graph_figure(features, edges, preds, probs, selected_node=sel_node)
        st.plotly_chart(fig_graph, use_container_width=True)

        # Highlighted node detail
        cls  = int(preds[sel_node])
        conf = float(probs[sel_node, cls]) * 100
        col1, col2 = st.columns([1, 2])
        with col1:
            dot_col = CLASS_COLORS[cls]
            st.markdown(f"""
            <div class="result-card">
              <div style="font-size:0.75rem;color:#8B949E;font-family:'JetBrains Mono',monospace;">NODE {sel_node}</div>
              <div class="result-class" style="color:{dot_col}">{CORA_CLASSES[cls]}</div>
              <div class="result-conf">Confidence: <strong>{conf:.1f}%</strong></div>
              <div class="conf-bar-bg"><div class="conf-bar" style="width:{conf}%;background:{dot_col};"></div></div>
            </div>
            """, unsafe_allow_html=True)

            # Neighbours
            nbrs = set()
            for s, d in zip(edges[0], edges[1]):
                if s == sel_node: nbrs.add(d)
                if d == sel_node: nbrs.add(s)
            st.markdown(f'<div class="section-head">Neighbours ({len(nbrs)})</div>', unsafe_allow_html=True)
            pills = "".join(f'<span class="node-pill">{nb}</span>' for nb in sorted(list(nbrs))[:16])
            if len(nbrs) > 16:
                pills += f'<span class="node-pill">+{len(nbrs)-16} more</span>'
            st.markdown(pills, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-head">Class probabilities</div>', unsafe_allow_html=True)
            st.plotly_chart(build_confidence_chart(probs[sel_node]), use_container_width=True)

    # TAB 2: Node inspector
    with tab2:
        st.markdown('<div class="section-head">Inspect any node</div>', unsafe_allow_html=True)
        node_id = st.number_input("Node ID", 0, n-1, 0, key="inspector_node")
        cls2 = int(preds[node_id])
        conf2 = float(probs[node_id, cls2]) * 100

        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.metric("Predicted Class", CORA_CLASSES[cls2])
        with ic2:
            st.metric("Confidence", f"{conf2:.1f}%")
        with ic3:
            nnz = int(np.sum(features[node_id] > 0))
            st.metric("Active BoW Features", f"{nnz} / 1433")

        st.markdown("---")
        st.markdown('<div class="section-head">Full probability distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(build_confidence_chart(probs[node_id]), use_container_width=True)

        st.markdown('<div class="section-head">Top-3 Predictions</div>', unsafe_allow_html=True)
        top3 = np.argsort(probs[node_id])[::-1][:3]
        for rank, c in enumerate(top3):
            pct = probs[node_id, c] * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
              <div style="width:22px;height:22px;border-radius:50%;background:{CLASS_COLORS[c]};
                          display:flex;align-items:center;justify-content:center;
                          font-size:0.7rem;font-weight:700;color:white;">#{rank+1}</div>
              <div style="flex:1;">
                <div style="color:#E6EDF3;font-size:0.88rem;font-weight:500;">{CORA_CLASSES[c]}</div>
                <div class="conf-bar-bg"><div class="conf-bar" style="width:{pct}%;background:{CLASS_COLORS[c]};"></div></div>
              </div>
              <div style="color:#8B949E;font-size:0.85rem;font-family:'JetBrains Mono',monospace;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # TAB 3: Analytics
    with tab3:
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown('<div class="section-head">Class distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(build_dist_chart(preds), use_container_width=True)
        with ac2:
            st.markdown('<div class="section-head">Mean confidence per class</div>', unsafe_allow_html=True)
            mean_conf = [float(probs[preds == c, c].mean() * 100) if (preds == c).sum() > 0 else 0 for c in range(7)]
            fig_conf = go.Figure(go.Bar(
                x=[CORA_CLASSES[c] for c in range(7)],
                y=mean_conf,
                marker=dict(color=CLASS_COLORS, opacity=0.85),
                text=[f"{v:.1f}%" for v in mean_conf],
                textposition='outside',
                textfont=dict(color='#8B949E', size=11)
            ))
            fig_conf.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=80),
                xaxis=dict(color='#8B949E', tickangle=-30, tickfont=dict(size=10)),
                yaxis=dict(color='#8B949E', gridcolor='#21262D', ticksuffix='%'),
                height=260
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-head">Node summary table</div>', unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame({
            "Node ID": list(range(n)),
            "Predicted Class": [CORA_CLASSES[p] for p in preds],
            "Confidence (%)": [f"{probs[i, preds[i]]*100:.1f}" for i in range(n)],
            "Active Features": [int(np.sum(features[i] > 0)) for i in range(n)]
        })
        st.dataframe(
            df,
            use_container_width=True,
            height=300,
            hide_index=True,
        )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
      <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
      <div style="font-size:1.1rem;font-weight:600;color:#E6EDF3;margin-bottom:0.5rem;">
        No graph classified yet
      </div>
      <div style="color:#8B949E;font-size:0.9rem;max-width:400px;margin:auto;line-height:1.7;">
        Use the sidebar to configure a demo graph or upload your own, then press
        <strong style="color:#818CF8;">Run Inference →</strong> to classify nodes.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #21262D;margin:2rem 0 1rem 0;">
<div style="text-align:center;color:#484F58;font-size:0.75rem;font-family:'JetBrains Mono',monospace;">
  GCN · 2-layer Graph Convolutional Network · Cora dataset · ONNX Runtime
</div>
""", unsafe_allow_html=True)
