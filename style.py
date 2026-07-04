"""
Design tokens for the Crisis-CLIP app: a "control room before a storm"
aesthetic — dark instrument-panel shell, light paper-white cards holding the
actual data, and a consistent blue->magenta "Beacon" gradient tying together
every explainability visual (Grad-CAM, beeswarm, bars).
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
INK = "#0B1220"       # app background
PANEL = "#131C2E"     # sidebar / dark card background
PAPER = "#F7F8FB"     # light "printout" card background
PAPER_BORDER = "#E4E7EF"
INK_TEXT = "#0B1220"  # text on light cards
FOG = "#8A93A6"        # secondary text on dark background
MIST = "#C7CCDA"       # secondary text on light background
PAPER_TEXT_MUTED = "#5B6272"

SIGNAL = "#FF6A3D"    # amber — alert / severe / not-informative
CURRENT = "#2DD4BF"   # teal — trust / informative / low severity

BEACON_BLUE = "#3B82F6"
BEACON_MAGENTA = "#E9408A"

BEACON_CMAP = LinearSegmentedColormap.from_list(
    "beacon", [BEACON_BLUE, "#8B5CF6", BEACON_MAGENTA]
)

# ---------------------------------------------------------------------------
# Fonts + global CSS
# ---------------------------------------------------------------------------
FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

CUSTOM_CSS = f"""
<link href="{FONT_IMPORT}" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background-color: {INK};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] * {{
        color: {FOG} !important;
    }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        font-family: 'Space Grotesk', sans-serif;
        color: {MIST} !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }}
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: #EDEFF4 !important;
    }}
    h1 {{
        letter-spacing: -0.01em;
    }}
    p, li, span, label, div {{
        color: #EDEFF4;
    }}
    .cc-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {FOG};
        margin-bottom: 0.15rem;
    }}
    .cc-card {{
        background-color: {PAPER};
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1.25rem 0;
        border-left: 5px solid {CURRENT};
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }}
    .cc-card * {{
        color: {INK_TEXT} !important;
    }}
    .cc-card .cc-eyebrow {{
        color: {PAPER_TEXT_MUTED} !important;
    }}
    .cc-card-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}
    .cc-bar-row {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.35rem 0;
    }}
    .cc-bar-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        width: 46%;
        flex-shrink: 0;
        color: {INK_TEXT};
    }}
    .cc-bar-track {{
        flex-grow: 1;
        background: #E9EBF2;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }}
    .cc-bar-fill {{
        height: 100%;
        border-radius: 6px;
    }}
    .cc-bar-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        width: 3.2rem;
        text-align: right;
        color: {PAPER_TEXT_MUTED};
        flex-shrink: 0;
    }}
    .cc-latency {{
        font-family: 'JetBrains Mono', monospace;
        color: {FOG};
        font-size: 0.85rem;
    }}
    div[data-testid="stFileUploader"] section {{
        background-color: {PANEL};
        border: 1px dashed rgba(255,255,255,0.15);
    }}
    .stButton > button {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: linear-gradient(90deg, {BEACON_BLUE}, {BEACON_MAGENTA});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }}
    .stButton > button:hover {{
        opacity: 0.9;
        color: white;
    }}
</style>
"""


def result_color(severity: float) -> str:
    """
    Maps a 0..1 severity/alert level to a hex color on the Signal<->Current
    axis (teal = calm/low, amber = alert/high) — used for the card's
    signature left-edge strip.
    """
    severity = max(0.0, min(1.0, severity))
    c1 = tuple(int(CURRENT[i : i + 2], 16) for i in (1, 3, 5))
    c2 = tuple(int(SIGNAL[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(c1[i] + (c2[i] - c1[i]) * severity) for i in range(3))
    return "#%02X%02X%02X" % mixed


def card_open(eyebrow: str, title: str, strip_color: str) -> str:
    return f"""
    <div class="cc-card" style="border-left-color:{strip_color};">
        <div class="cc-eyebrow">{eyebrow}</div>
        <div class="cc-card-title">{title}</div>
    """


def card_close() -> str:
    return "</div>"


def render_prob_bars(labels_and_probs, colors=None) -> str:
    """
    Builds the custom gradient-bar HTML used instead of st.progress, so bars
    match the app's palette instead of Streamlit's default red.
    labels_and_probs: list of (label, prob) tuples, already sorted.
    colors: optional list of hex colors, same length/order.
    """
    rows = []
    for i, (label, prob) in enumerate(labels_and_probs):
        color = colors[i] if colors else BEACON_BLUE
        pct = max(0.0, min(1.0, prob)) * 100
        rows.append(
            f"""
            <div class="cc-bar-row">
                <div class="cc-bar-label">{label}</div>
                <div class="cc-bar-track">
                    <div class="cc-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <div class="cc-bar-value">{pct:.1f}%</div>
            </div>
            """
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------

def apply_chart_style(fig, ax_list):
    """Applies the paper-card chart look: light bg, minimal spines, muted grid."""
    if not isinstance(ax_list, (list, tuple)):
        ax_list = [ax_list]
    fig.patch.set_facecolor(PAPER)
    for ax in ax_list:
        ax.set_facecolor(PAPER)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(PAPER_BORDER)
        ax.tick_params(colors=PAPER_TEXT_MUTED, labelsize=8.5)
        ax.xaxis.label.set_color(PAPER_TEXT_MUTED)
        ax.yaxis.label.set_color(PAPER_TEXT_MUTED)
        ax.title.set_color(INK_TEXT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("monospace")
        ax.grid(axis="y", color=PAPER_BORDER, linewidth=0.7, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)


plt.rcParams["font.family"] = "sans-serif"