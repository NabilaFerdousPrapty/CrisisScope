"""
Crisis-CLIP Streamlit demo.

Pipeline:
  1. User uploads an image + enters social-media-style text.
  2. Informativeness model gates the rest of the pipeline (matches your
     CrisisMMD task design: non-informative posts aren't analyzed further,
     but you can override this in the sidebar for demo purposes).
  3. If informative: humanitarian-category (5-class) and damage-severity
     (3-class) models run, each with live Grad-CAM + text-token XAI.

Run with:
    streamlit run app.py
"""

import time
import re
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor

def clean_social_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter/X mentions
    text = re.sub(r"@\w+", " ", text)

    # Convert hashtags: #earthquake -> earthquake
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove common social media prefixes
    text = re.sub(r"\brt\b", " ", text)
    

    # Keep letters, numbers, decimal points, and useful punctuation
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
from models import (
    CLIP_MODEL_ID,
    MAX_TEXT_LEN,
    INFORMATIVE_LABELS,
    HUMANITARIAN_LABELS,
    DAMAGE_LABELS,
    load_informativeness_model,
    load_humanitarian_model,
    load_damage_model,
)
from xai import compute_explanations, overlay_saliency
import style

st.set_page_config(page_title="Crisis-CLIP", layout="wide", page_icon="🛰️")
st.markdown(style.CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading CLIP processor...")
def get_processor():
    return CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


@st.cache_resource(show_spinner="Loading informativeness model...")
def get_informativeness_model(ckpt_path, _device):
    return load_informativeness_model(ckpt_path, _device)


@st.cache_resource(show_spinner="Loading humanitarian-category model...")
def get_humanitarian_model(ckpt_path, _device):
    return load_humanitarian_model(ckpt_path, _device)


@st.cache_resource(show_spinner="Loading damage-severity model...")
def get_damage_model(ckpt_path, _device):
    return load_damage_model(ckpt_path, _device)


# ---------------------------------------------------------------------------
# Sidebar — checkpoint paths & options
# ---------------------------------------------------------------------------

st.sidebar.markdown("## Model checkpoints")
info_ckpt = st.sidebar.text_input("Informativeness .pth", "checkpoints/best_informativeness_lora.pth")
human_ckpt = st.sidebar.text_input("Humanitarian (LoRA) .pth", "checkpoints/best_humanitarian_enhanced_lora.pth")
damage_ckpt = st.sidebar.text_input("Damage severity (LoRA) .pth", "checkpoints/best_damage_model.pth")

st.sidebar.markdown("## Options")
force_continue = st.sidebar.checkbox(
    "Always run humanitarian/damage models (ignore informativeness gate)", value=False
)
show_xai = st.sidebar.checkbox("Show Grad-CAM + token importance", value=True)

st.sidebar.caption(
    "Place your three trained .pth files anywhere on disk and point these "
    "fields at them. Models load once and are cached."
)

device = get_device()
st.sidebar.markdown(f'<span class="cc-latency">Device: {device}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="cc-eyebrow" style="color:#8A93A6;">CRISIS-CLIP · MULTIMODAL DISASTER ANALYSIS</div>
    <h1 style="margin-top:0;">Read the signal in a single post</h1>
    <p style="color:#8A93A6; max-width:640px; margin-top:-0.5rem;">
        Upload an image and its caption. Three CLIP-based classifiers run in
        sequence — informativeness, humanitarian category, damage severity —
        each with a live Grad-CAM and text-token explanation of its call.
    </p>
    """,
    unsafe_allow_html=True,
)

col_input1, col_input2 = st.columns([1, 1])
with col_input1:
    uploaded_image = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
with col_input2:
    text_input = st.text_area("Caption / tweet text", height=150, placeholder="e.g. Streets flooded near downtown, several homes damaged.")

run_button = st.button("Analyze", type="primary", use_container_width=True)


def predict(model, ids, mask, pix):
    with torch.no_grad():
        logits = model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), pix.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
    pred = int(probs.argmax())
    return pred, probs


def format_token_label(tok, width=7):
    """
    Cleans up a raw CLIP BPE token for display: strips the `</w>`
    end-of-word marker (it's a tokenizer artifact, not meaningful content)
    and wraps anything longer than `width` characters onto two lines so
    labels stay readable at 0-45 degree rotation instead of needing a steep
    75 degree rotation to avoid overlapping.
    """
    clean = tok.replace("</w>", "")
    if not clean:
        return tok  # fall back to raw token if cleaning empties it out
    wrapped = textwrap.wrap(clean, width=width, break_long_words=True)
    return "\n".join(wrapped[:2]) if wrapped else clean


def show_probs(labels_map, probs):
    """Renders custom Beacon-gradient bars (sorted high to low) instead of st.progress."""
    order = np.argsort(-probs)
    rows = [(labels_map[idx], float(probs[idx])) for idx in order]
    colors = []
    for _, p in rows:
        # interpolate along the Beacon gradient by rank confidence
        rgba = style.BEACON_CMAP(0.15 + 0.7 * p)
        colors.append("#%02X%02X%02X" % tuple(int(c * 255) for c in rgba[:3]))
    st.markdown(style.render_prob_bars(rows, colors), unsafe_allow_html=True)


def plot_token_beeswarm(tokens, contrib_per_dim, activation_per_dim, max_tokens=10):
    """
    SHAP-beeswarm-style plot for a single (image, text) prediction.
    Each row = one token. Each dot = one embedding dimension's signed
    Gradient x Input contribution (x position), colored by that dimension's
    activation strength — the closest honest analogue to a SHAP summary plot
    for a single instance, since a true beeswarm needs many samples per
    feature and here we only have one.

    The x-axis is clipped to the 2nd-98th percentile of the plotted values
    (with padding) rather than auto-scaling to the full min/max. A single
    outlier dimension in one token can otherwise stretch the axis so far
    that every other token's spread collapses to a visually flat line, even
    though the underlying values are meaningful. Clipping only affects the
    axis limits — points beyond the range are still drawn, just off-canvas
    at the edges, so no data is discarded or altered.
    """
    total_abs = np.abs(contrib_per_dim).sum(axis=1)
    order = np.argsort(-total_abs)[:max_tokens]
    order = order[::-1]  # most important token ends up at the top

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(order))))
    norm = plt.Normalize(vmin=np.percentile(activation_per_dim, 5), vmax=np.percentile(activation_per_dim, 95))
    cmap = style.BEACON_CMAP

    rng = np.random.default_rng(0)
    plotted_x = []
    for row, tok_idx in enumerate(order):
        x = contrib_per_dim[tok_idx]
        c = activation_per_dim[tok_idx]
        jitter = rng.uniform(-0.35, 0.35, size=len(x))
        ax.scatter(x, np.full_like(x, row) + jitter, c=c, cmap=cmap, norm=norm, s=12, alpha=0.75, linewidths=0)
        plotted_x.append(x)

    # Percentile-based x-limits so one outlier dimension doesn't flatten the
    # rest of the swarm visually.
    all_plotted_x = np.concatenate(plotted_x) if plotted_x else np.array([0.0])
    lo, hi = np.percentile(all_plotted_x, [2, 98])
    if hi <= lo:
        lo, hi = all_plotted_x.min(), all_plotted_x.max()
    if hi <= lo:
        lo, hi = -1e-6, 1e-6
    pad = (hi - lo) * 0.2
    ax.set_xlim(lo - pad, hi + pad)

    ax.axvline(0, color=style.PAPER_BORDER, linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([format_token_label(tokens[i], width=10) for i in order], fontfamily="monospace")
    ax.set_xlabel("Gradient × Input contribution")
    ax.set_title("Text token attribution", loc="left", fontsize=11, fontweight="bold")
    style.apply_chart_style(fig, ax)
    ax.grid(axis="x", color=style.PAPER_BORDER, linewidth=0.7, alpha=0.7)
    ax.grid(axis="y", visible=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Embedding activation", color=style.PAPER_TEXT_MUTED, fontsize=8.5)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(colors=style.PAPER_TEXT_MUTED, labelsize=8)

    fig.tight_layout()
    return fig


def show_xai_panel(model, tokenizer, ids, mask, pix, pil_image, pred_class, container):
    """
    Renders the Grad-CAM + token-attribution panel for one stage, and
    returns the raw `compute_explanations` result dict so the caller can
    stash it for cross-stage comparison (see the divergence check after
    stage 3 below).
    """
    result = compute_explanations(model, tokenizer, ids, mask, pix, device, target_class=pred_class)
    grid = result["grid"]
    tokens = result["tokens"]
    importance = result["importance"]
    contrib_per_dim = result["contrib_per_dim"]
    activation_per_dim = result["activation_per_dim"]

    overlay = overlay_saliency(pil_image, grid)  # jet default -> real heatmap colors, not the brand palette
    # container.code(
    #     f"grid shape        : {grid.shape}\n"
    #     f"grid min/max/std  : {grid.min():.6f} / {grid.max():.6f} / {grid.std():.6f}\n"
    #     f"pred_class        : {pred_class}\n"
    #     f"probs             : {result['probs']}\n"
    #     f"import path       : {compute_explanations.__module__} -> {compute_explanations.__code__.co_filename}\n"
    # )
    container.markdown(
        '<div class="cc-eyebrow" style="margin-top:0.5rem;">WHY THIS PREDICTION</div>',
        unsafe_allow_html=True,
    )

    container.image(overlay, caption="Grad-CAM — image regions driving the prediction", use_container_width=True)

    # Cap to the top-N most important tokens rather than plotting the whole
    # sequence (which can be 30-40+ tokens for a full caption). Uncapped, the
    # figure width (scaled per-token) balloons far past the fixed height,
    # producing an extreme aspect ratio that gets squashed when scaled to
    # fit the container width.
    MAX_BAR_TOKENS = 15
    if len(tokens) > MAX_BAR_TOKENS:
        keep_idx = np.sort(np.argsort(-importance)[:MAX_BAR_TOKENS])
    else:
        keep_idx = np.arange(len(tokens))
    display_tokens = [tokens[i] for i in keep_idx]
    display_importance = importance[keep_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(display_tokens) * 0.6), 3.8))
    bar_colors = [style.BEACON_CMAP(0.15 + 0.75 * v) for v in display_importance]
    ax.bar(range(len(display_tokens)), display_importance, color=bar_colors)
    ax.set_xticks(range(len(display_tokens)))
    ax.set_xticklabels([])  # hide default single-row labels; drawn manually below, staggered

    # Draw labels in two staggered rows (even index near the axis, odd index
    # further below) instead of one crowded row. This is what actually
    # prevents overlap for short adjacent tokens — word-wrapping alone only
    # helps long tokens, and rotation alone still collides at this density.
    for i, tok in enumerate(display_tokens):
        label = format_token_label(tok, width=10).replace("\n", "")
        y_offset = -0.05 if i % 2 == 0 else -0.16
        ax.text(
            i, y_offset, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=8, fontfamily="monospace",
            color=style.PAPER_TEXT_MUTED,
        )

    ax.set_ylabel("Saliency")
    title = "Text token importance"
    if len(tokens) > MAX_BAR_TOKENS:
        title += f" (top {MAX_BAR_TOKENS} of {len(tokens)} shown)"
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    style.apply_chart_style(fig, ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)  # must come AFTER tight_layout, which would otherwise reset it
    container.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # SHAP-beeswarm-style view of the same text attribution, full width
    beeswarm_fig = plot_token_beeswarm(tokens, contrib_per_dim, activation_per_dim)
    container.pyplot(beeswarm_fig, use_container_width=True)
    plt.close(beeswarm_fig)

    # Plain-language summary so the "why" isn't only visual
    if len(tokens) > 0:
        top_k = min(3, len(tokens))
        top_idx = np.argsort(-importance)[:top_k]
        top_words = [format_token_label(tokens[i], width=99).replace("\n", "") for i in top_idx if importance[i] > 0]
        if top_words:
            container.markdown(
                f'<span class="cc-latency">Most influential words: '
                f'<b style="color:{style.INK_TEXT} !important;">{", ".join(top_words)}</b></span>',
                unsafe_allow_html=True,
            )

    return result


def show_divergence_check(stage_results, container):
    """
    Small diagnostic panel: compares the token-importance vectors across
    whichever stages ran (informativeness / humanitarian / damage) using
    cosine similarity, so you can directly verify the three stages produce
    genuinely different attributions rather than eyeballing separate plots.

    stage_results: dict like {"Informativeness": result, "Humanitarian": result, ...}
    where each result is a dict returned by compute_explanations (must share
    the same `tokens`, since all stages run on the same input encoding).

    Similarity near 1.0 for a pair means their importance vectors are nearly
    identical — worth double-checking that those two stages are really
    loading different checkpoints. Lower values indicate the stages are
    genuinely attending to different words for their predictions.
    """
    names = list(stage_results.keys())
    if len(names) < 2:
        return

    def cosine_sim(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    container.markdown(
        '<div class="cc-eyebrow" style="margin-top:0.5rem;">CROSS-MODEL DIVERGENCE CHECK</div>',
        unsafe_allow_html=True,
    )

    rows_html = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            imp_a = stage_results[name_a]["importance"]
            imp_b = stage_results[name_b]["importance"]
            if len(imp_a) != len(imp_b):
                rows_html.append(
                    f'<div class="cc-latency">{name_a} vs {name_b}: token counts differ '
                    f'({len(imp_a)} vs {len(imp_b)}), skipping comparison.</div>'
                )
                continue
            sim = cosine_sim(imp_a, imp_b)
            flag = " — nearly identical, worth checking these load different checkpoints" if sim > 0.98 else ""
            rows_html.append(
                f'<div class="cc-latency">{name_a} vs {name_b}: cosine similarity = {sim:.3f}{flag}</div>'
            )

    container.markdown("".join(rows_html), unsafe_allow_html=True)
    container.caption(
        "Similarity close to 1.0 means two stages are weighting the same words almost identically. "
        "Lower values confirm each model is genuinely attending to different parts of the text."
    )


if run_button:
    if uploaded_image is None or not text_input.strip():
        st.warning("Please provide both an image and some text.")
        st.stop()

    t0 = time.time()
    pil_image = Image.open(uploaded_image).convert("RGB")
    processor = get_processor()

    cleaned_text = clean_social_text(text_input)

    st.caption(f"Text used by model: {cleaned_text}")

    encoding = processor(
        text=[cleaned_text],
        images=pil_image,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
    )
    ids = encoding["input_ids"].squeeze(0)
    mask = encoding["attention_mask"].squeeze(0)
    pix = encoding["pixel_values"].squeeze(0)

    st.image(pil_image, caption="Input image", width=320)

    stage_results = {}

    # --- Stage 1: Informativeness -------------------------------------------------
    try:
        info_model = get_informativeness_model(info_ckpt, device)
    except Exception as e:
        st.error(f"Could not load informativeness checkpoint at `{info_ckpt}`: {e}")
        st.stop()

    info_pred, info_probs = predict(info_model, ids, mask, pix)
    is_informative = info_pred == 1
    strip_color = style.result_color(0.0 if is_informative else 1.0)

    st.markdown(style.card_open("STAGE 1 / 3", "Informativeness", strip_color), unsafe_allow_html=True)
    show_probs(INFORMATIVE_LABELS, info_probs)
    if show_xai:
        stage_results["Informativeness"] = show_xai_panel(info_model, processor.tokenizer, ids, mask, pix, pil_image, info_pred, st)
    st.markdown(style.card_close(), unsafe_allow_html=True)

    if not is_informative and not force_continue:
        st.info(
            "Predicted **Not Informative** — humanitarian category and damage "
            "severity are skipped (this mirrors the CrisisMMD task design). "
            "Tick the sidebar override to run them anyway."
        )
        st.markdown(f'<span class="cc-latency">Total latency: {(time.time() - t0)*1000:.0f} ms</span>', unsafe_allow_html=True)
        st.stop()

    # --- Stage 2: Humanitarian category ---------------------------------------
    try:
        human_model = get_humanitarian_model(human_ckpt, device)
    except Exception as e:
        st.error(f"Could not load humanitarian checkpoint at `{human_ckpt}`: {e}")
        st.stop()

    human_pred, human_probs = predict(human_model, ids, mask, pix)
    not_humanitarian_idx = 2  # "Not Humanitarian" class index
    human_severity = 0.15 if human_pred == not_humanitarian_idx else 0.55
    st.markdown(style.card_open("STAGE 2 / 3", "Humanitarian Category", style.result_color(human_severity)), unsafe_allow_html=True)
    show_probs(HUMANITARIAN_LABELS, human_probs)
    if show_xai:
        stage_results["Humanitarian"] = show_xai_panel(human_model, processor.tokenizer, ids, mask, pix, pil_image, human_pred, st)
    st.markdown(style.card_close(), unsafe_allow_html=True)

    # --- Stage 3: Damage severity ----------------------------------------------
    try:
        damage_model = get_damage_model(damage_ckpt, device)
    except Exception as e:
        st.error(f"Could not load damage-severity checkpoint at `{damage_ckpt}`: {e}")
        st.stop()

    damage_pred, damage_probs = predict(damage_model, ids, mask, pix)
    damage_severity_map = {0: 1.0, 1: 0.5, 2: 0.05}  # Severe / Mild / None
    st.markdown(style.card_open("STAGE 3 / 3", "Damage Severity", style.result_color(damage_severity_map.get(damage_pred, 0.5))), unsafe_allow_html=True)
    show_probs(DAMAGE_LABELS, damage_probs)
    if show_xai:
        stage_results["Damage Severity"] = show_xai_panel(damage_model, processor.tokenizer, ids, mask, pix, pil_image, damage_pred, st)
    st.markdown(style.card_close(), unsafe_allow_html=True)

    if show_xai and len(stage_results) >= 2:
        st.markdown(style.card_open("DIAGNOSTIC", "Model Attribution Comparison", style.CURRENT), unsafe_allow_html=True)
        show_divergence_check(stage_results, st)
        st.markdown(style.card_close(), unsafe_allow_html=True)

    st.markdown(
        f'<span class="cc-latency">✓ Done — total latency: {(time.time() - t0)*1000:.0f} ms</span>',
        unsafe_allow_html=True,
    )