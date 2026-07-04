"""
Design tokens for the Crisis-CLIP app: a clean, white academic-journal
aesthetic — think a formal research paper or conference poster rather than a
dark control room. Off-white page, serif headings, restrained navy/burgundy
accent, and a muted "Beacon" gradient (navy -> burgundy) tying together every
explainability visual (Grad-CAM, beeswarm, bars).

All function names/signatures are unchanged from the previous version, so
app.py does not need to change.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
INK = "#FFFFFF"        # app background (was dark ink, now paper white)
PANEL = "#F6F7F9"       # sidebar / secondary panel background
PAPER = "#FFFFFF"       # card background
PAPER_BORDER = "#DDE1E8"
INK_TEXT = "#1B2230"    # primary text color
FOG = "#5B6272"         # secondary text on light background (sidebar)
MIST = "#3C4454"        # secondary heading text on light background
PAPER_TEXT_MUTED = "#6B7280"

SIGNAL = "#8B2635"      # deep burgundy — alert / severe / not-informative
CURRENT = "#1B2A4A"     # deep navy — calm / informative / low severity

BEACON_BLUE = "#1B2A4A"      # deep navy
BEACON_MAGENTA = "#8B2635"   # deep burgundy

BEACON_CMAP = LinearSegmentedColormap.from_list(
    "beacon", [BEACON_BLUE, "#5B4B6E", BEACON_MAGENTA]
)

ACCENT_GOLD = "#A9812E"  # small formal accent (rules, active states)

# ---------------------------------------------------------------------------
# Fonts + global CSS
# ---------------------------------------------------------------------------
FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Source+Serif+4:wght@500;600;700&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500&display=swap"
)

CUSTOM_CSS = f"""
<link href="{FONT_IMPORT}" rel="stylesheet">
<style>
    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .stApp {{
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F8FC 35%, #FFFFFF 65%, #FBF8F3 100%);
        background-size: 300% 300%;
        animation: cc-bg-shift 30s ease infinite;
    }}
    @keyframes cc-bg-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    section[data-testid="stSidebar"] {{
        background-color: {PANEL};
        border-right: 1px solid {PAPER_BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {FOG} !important;
    }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        font-family: 'Source Serif 4', serif;
        color: {MIST} !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 0.78rem !important;
        border-bottom: 2px solid {ACCENT_GOLD};
        padding-bottom: 0.35rem;
        margin-bottom: 0.6rem !important;
    }}
    h1, h2, h3 {{
        font-family: 'Source Serif 4', serif !important;
        color: {INK_TEXT} !important;
        font-weight: 600 !important;
    }}
    h1 {{
        letter-spacing: -0.01em;
        padding-bottom: 0.6rem;
        margin-bottom: 0.9rem !important;
        display: inline-block;
        position: relative;
    }}
    h1::after {{
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, {BEACON_BLUE}, {ACCENT_GOLD} 40%, {BEACON_MAGENTA}, {BEACON_BLUE});
        background-size: 220% 100%;
        border-radius: 2px;
        animation: cc-underline-shift 7s linear infinite;
    }}
    @keyframes cc-underline-shift {{
        from {{ background-position: 0% 0; }}
        to {{ background-position: 220% 0; }}
    }}
    p, li, span, label, div {{
        color: {INK_TEXT};
    }}
    .cc-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {FOG};
        margin-bottom: 0.15rem;
    }}
    .cc-card {{
        background-color: {PAPER};
        border-radius: 6px;
        padding: 1.35rem 1.6rem;
        margin: 0.85rem 0 1.35rem 0;
        border: 1px solid {PAPER_BORDER};
        border-left: 4px solid {CURRENT};
        box-shadow: 0 1px 3px rgba(20,24,36,0.05);
        transition: box-shadow 0.15s ease, transform 0.15s ease;
        animation: cc-fade-up 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    .cc-card:hover {{
        box-shadow: 0 6px 16px rgba(20,24,36,0.09);
        transform: translateY(-1px);
    }}
    @keyframes cc-fade-up {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .cc-card * {{
        color: {INK_TEXT} !important;
    }}
    .cc-card .cc-eyebrow {{
        color: {PAPER_TEXT_MUTED} !important;
    }}
    .cc-card-title {{
        font-family: 'Source Serif 4', serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }}
    .cc-bar-row {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 0.4rem 0;
    }}
    .cc-bar-label {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        width: 46%;
        flex-shrink: 0;
        color: {INK_TEXT};
    }}
    .cc-bar-track {{
        flex-grow: 1;
        background: #EDEFF3;
        border-radius: 3px;
        height: 9px;
        overflow: hidden;
        border: 1px solid {PAPER_BORDER};
    }}
    .cc-bar-fill {{
        height: 100%;
        border-radius: 2px;
        width: var(--bar-w, 0%);
        animation: cc-grow-bar 0.9s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    @keyframes cc-grow-bar {{
        from {{ width: 0%; }}
        to {{ width: var(--bar-w, 0%); }}
    }}
    .cc-bar-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        width: 3.2rem;
        text-align: right;
        color: {PAPER_TEXT_MUTED};
        flex-shrink: 0;
    }}
    .cc-latency {{
        font-family: 'JetBrains Mono', monospace;
        color: {PAPER_TEXT_MUTED};
        font-size: 0.82rem;
    }}
    div[data-testid="stFileUploader"] section {{
        background-color: {PANEL};
        border: 1px dashed {PAPER_BORDER};
    }}
    .stButton > button {{
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        letter-spacing: 0.01em;
        background: {CURRENT} !important;
        border: none;
        border-radius: 5px;
        padding: 0.65rem 1.2rem;
        box-shadow: 0 2px 6px rgba(27,42,74,0.25);
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
        animation: cc-btn-glow 3.2s ease-in-out infinite;
    }}
    @keyframes cc-btn-glow {{
        0%, 100% {{ box-shadow: 0 2px 6px rgba(27,42,74,0.25); }}
        50% {{ box-shadow: 0 2px 14px rgba(27,42,74,0.5); }}
    }}
    .stButton > button:active {{
        transform: scale(0.98);
    }}
    /* Streamlit wraps the label in nested p/div/span tags; the global
       text-color rule above would otherwise make it invisible on the
       dark button background, so we force white here with higher
       specificity and !important. */
    .stButton > button,
    .stButton > button p,
    .stButton > button div,
    .stButton > button span {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover {{
        background: {SIGNAL} !important;
        box-shadow: 0 3px 10px rgba(139,38,53,0.3);
        animation-play-state: paused;
    }}
    .stButton > button:hover,
    .stButton > button:hover p,
    .stButton > button:hover div,
    .stButton > button:hover span {{
        color: #FFFFFF !important;
    }}
    .stButton > button:focus:not(:active) {{
        color: #FFFFFF !important;
        box-shadow: 0 0 0 3px rgba(27,42,74,0.2);
    }}
</style>
"""


def result_color(severity: float) -> str:
    """
    Maps a 0..1 severity/alert level to a hex color on the Current<->Signal
    axis (navy = calm/low, burgundy = alert/high) — used for the card's
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
    match the app's palette instead of Streamlit's default red. Bars animate
    growing in from 0 to their value (staggered slightly row-to-row) via the
    --bar-w custom property picked up by the .cc-bar-fill keyframes in
    CUSTOM_CSS.
    labels_and_probs: list of (label, prob) tuples, already sorted.
    colors: optional list of hex colors, same length/order.
    """
    rows = []
    for i, (label, prob) in enumerate(labels_and_probs):
        color = colors[i] if colors else BEACON_BLUE
        pct = max(0.0, min(1.0, prob)) * 100
        delay = i * 0.06
        rows.append(
            f"""
            <div class="cc-bar-row">
                <div class="cc-bar-label">{label}</div>
                <div class="cc-bar-track">
                    <div class="cc-bar-fill" style="--bar-w:{pct:.1f}%; background:{color}; animation-delay:{delay:.2f}s;"></div>
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
    """Applies the paper-card chart look: white bg, minimal spines, muted grid."""
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