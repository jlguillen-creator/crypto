"""
Crypto Predictor 5min — Streamlit App
Fuente de datos: Kraken Public API (sin API key)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from datetime import datetime, timezone
from crypto_predictor import (
    descargar_datos, calcular_indicadores,
    calcular_prediccion, BLOQUES_CRYPTO, PESOS_BASE,
    normalizar_symbol, scan_rapido
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Crypto Predictor 5m",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #08090c;
    color: #c8d0e0;
    font-weight: 500;
}
body::after {
    content: '';
    position: fixed; top:0; left:0; width:100%; height:100%;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
        rgba(255,180,0,0.008) 2px, rgba(255,180,0,0.008) 4px);
    pointer-events: none; z-index: 9998;
}

/* ── Hero ── */
.hero {
    background: #0c0e15;
    border: 1px solid #1e2432;
    border-top: 2px solid var(--acc, #f5a623);
    border-radius: 6px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    height: 100%;
    box-sizing: border-box;
}
.hero-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; letter-spacing: 4px;
    font-weight: 700; color: #8892a4;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.hero-dir {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.6rem; font-weight: 700; line-height: 1;
    color: var(--acc, #f5a623);
    text-shadow: 0 0 40px var(--acc-glow, rgba(245,166,35,0.3));
    letter-spacing: -2px;
}
.hero-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; letter-spacing: 3px; font-weight: 700;
    padding: 0.25rem 0.7rem;
    border: 1px solid currentColor; border-radius: 2px;
    display: inline-block; margin-top: 0.5rem;
}
.price-big {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 700; color: #e8edf5;
}
.price-chg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; margin-top: 0.15rem; font-weight: 600;
}
.target-box {
    background: #0a0c12; border: 1px solid #1e2432;
    border-left: 2px solid var(--acc, #f5a623);
    padding: 0.65rem 0.9rem; border-radius: 2px; margin-top: 0.6rem;
}
.target-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
}
.target-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem; color: #8892a4;
    font-weight: 600; margin-top: 0.2rem;
}
.score-big {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.2rem; font-weight: 700; line-height: 1;
}
.score-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem; color: #6b7894; margin-top: 0.15rem;
}
.counters {
    display: flex; gap: 1.2rem; margin-top: 0.7rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 700;
}

/* ── Prob boxes ── */
.pbox {
    background: #0c0e15; border: 1px solid #1e2432;
    border-radius: 6px; padding: 1rem 1.2rem;
}
.pbox-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem; letter-spacing: 3px;
    text-transform: uppercase; color: #8892a4;
    font-weight: 700; margin-bottom: 0.4rem;
}
.pbox-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.pbar-track {
    background: #141820; height: 5px;
    border-radius: 1px; margin-top: 0.5rem; overflow: hidden;
}
.pbar-fill { height: 100%; border-radius: 1px; }
.pbox-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem; color: #8892a4;
    font-weight: 600; margin-top: 0.35rem;
}

/* ── Info cards ── */
.icard {
    background: #0c0e15; border: 1px solid #1a1e2c;
    border-radius: 6px; padding: 0.85rem 1rem;
}
.icard-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem; letter-spacing: 3px;
    text-transform: uppercase; color: #8892a4;
    font-weight: 700; margin-bottom: 0.25rem;
}
.icard-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem; font-weight: 700; color: #e8edf5;
}

/* ── Indicator table ── */
.blk-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem; letter-spacing: 3px; font-weight: 700;
    color: #6b7894; text-transform: uppercase;
    margin: 1.1rem 0 0.4rem;
    border-bottom: 1px solid #141820; padding-bottom: 0.3rem;
}
.ind-row {
    display: grid;
    grid-template-columns: 7px 155px 1fr auto;
    align-items: center; gap: 0.5rem;
    padding: 0.4rem 0; border-bottom: 1px solid #0e1018;
}
.ind-dot  { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.ind-name { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; font-weight:600; color:#9aa0b4; }
.ind-val  { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; font-weight:500; color:#5a6480; }
.ind-sig  { font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700; text-align:right; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0b10;
    border-right: 1px solid #1a1e2c;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #c8d0e0 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    color: #8892a4 !important;
}

/* ── Top control bar ── */
.ctrl-bar {
    background: #0c0e15;
    border: 1px solid #1e2432;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.8rem;
    display: flex; align-items: center;
    gap: 0.6rem; flex-wrap: wrap;
}
.ctrl-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem; font-weight: 700;
    color: #f5a623; letter-spacing: 2px;
    white-space: nowrap;
}

/* ── Crypto chip buttons ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.4rem !important;
    border-radius: 20px !important;
    border: 1px solid #2a3040 !important;
    background: #141820 !important;
    color: #c8d0e0 !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: #f5a623 !important;
    color: #f5a623 !important;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #0c0e15; border: 1px solid #1e2432;
    border-left: 2px solid #f5a623;
    padding: 0.6rem 1rem; font-size: 0.68rem;
    color: #8892a4; font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; margin-top: 1.5rem; border-radius: 2px;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .hero { padding: 1rem 0.9rem; }
    .hero-dir { font-size: 2.4rem; }
    .price-big { font-size: 1.5rem; }
    .score-big { font-size: 2.2rem; }
    .pbox-val  { font-size: 1.8rem; }
    .ind-row { grid-template-columns: 7px 1fr auto; }
    .ind-val { display: none; }
    .hero-label { font-size: 0.58rem; letter-spacing: 2px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────
C = {
    "alcista":      "#00e87a",
    "alcista_leve": "#4dd68c",
    "bajista":      "#ff4f6a",
    "bajista_leve": "#f5a623",
    "neutro":       "#6b7894",
}
GLOW = {
    "alcista":      "rgba(0,232,122,0.3)",
    "alcista_leve": "rgba(77,214,140,0.2)",
    "bajista":      "rgba(255,79,106,0.3)",
    "bajista_leve": "rgba(245,166,35,0.2)",
    "neutro":       "rgba(107,120,148,0.1)",
}

CRYPTOS = [
    ("BTC","₿ BTC"), ("ETH","Ξ ETH"),   ("SOL","◎ SOL"),  ("XRP","✕ XRP"),
    ("BNB","⬡ BNB"), ("DOGE","Ð DOGE"), ("ADA","₳ ADA"),  ("AVAX","▲ AVAX"),
    ("DOT","● DOT"), ("LINK","⬡ LINK"),  ("LTC","Ł LTC"),  ("ATOM","⚛ ATOM"),
]

# ─────────────────────────────────────────────
# SCAN DE FONDO — se ejecuta al inicio de cada render
# Si el cache expiró (5 min), lanza los 12 scans y rerenderiza
# Al rerenderizar, los chips ya tienen los colores correctos
# ─────────────────────────────────────────────
import time as _time

_scan_ts   = st.session_state.get("scan_ts", 0)
_CACHE_TTL = 300   # 5 minutos

if _time.time() - _scan_ts > _CACHE_TTL:
    _progress = st.empty()
    _progress.caption("⟳ Escaneando señales…")
    _new_scores = {}
    for _sym, _ in CRYPTOS:
        _new_scores[_sym] = scan_rapido(_sym)
    st.session_state["scan_scores"] = _new_scores
    st.session_state["scan_ts"]     = _time.time()
    _progress.empty()
    st.rerun()

# ─────────────────────────────────────────────
# SIDEBAR (colapsado por defecto, útil en desktop)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("**⚡ CRYPTO PREDICTOR**")
    st.markdown("**5-MINUTE SIGNAL**")
    st.caption("Kraken API · sin registro · 20 indicadores")
    st.divider()

    sb_symbol = st.text_input(
        "Par de trading", value="BTC",
        placeholder="BTC, ETH, SOL…",
        key="sb_input"
    ).strip()
    auto_refresh = st.checkbox("Auto-refresh cada 60s", value=False)
    sb_analyze = st.button("⚡ ANALIZAR", use_container_width=True,
                           type="primary", key="sb_btn")
    st.divider()
    st.markdown("**TOP 12**")
    sb_c1, sb_c2 = st.columns(2)
    sb_selected = None
    for i, (sym_k, label) in enumerate(CRYPTOS):
        col = sb_c1 if i % 2 == 0 else sb_c2
        if col.button(label, key=f"sb_{sym_k}", use_container_width=True):
            sb_selected = sym_k
    st.divider()
    st.caption("FUENTE · Kraken REST API\nVELAS · 1m + 5m\nORDER BOOK · Top 20")

# ─────────────────────────────────────────────
# BARRA DE CONTROL SUPERIOR (mobile-first)
# ─────────────────────────────────────────────
top_c1, top_c2, top_c3 = st.columns([2, 4, 2])
with top_c1:
    st.markdown('<div class="ctrl-title">⚡ 5-MIN</div>', unsafe_allow_html=True)
with top_c2:
    top_symbol = st.text_input(
        "Par", value="BTC", label_visibility="collapsed",
        placeholder="BTC, ETH, SOL, XRP…", key="top_input"
    ).strip()
with top_c3:
    top_analyze = st.button("⚡ ANALIZAR", key="top_btn",
                            use_container_width=True, type="primary")

# ── Chips via query params — HTML grid puro, 2 col fijas en móvil ──
qp = st.query_params
chip_selected = qp.get("crypto", None)
if chip_selected:
    st.query_params.clear()

# Colores de señal del scan de fondo (session_state)
_SCAN_BG   = {"alcista": "#00e87a", "bajista": "#ff4f6a", "neutro": "#2a3040"}
_SCAN_BORD = {"alcista": "#00e87a", "bajista": "#ff4f6a", "neutro": "#2a3040"}
_SCAN_TXT  = {"alcista": "#00e87a", "bajista": "#ff4f6a", "neutro": "#c8d0e0"}

def _chip_style(sym_k):
    scan = st.session_state.get("scan_scores", {}).get(sym_k, {})
    color = scan.get("color", "neutro")
    bg   = "#0d1a0f" if color == "alcista" else ("#1a0d0f" if color == "bajista" else "#141820")
    bord = _SCAN_BORD[color]
    txt  = _SCAN_TXT[color]
    return bg, bord, txt

chip_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:0.6rem;">'
for sym_k, label in CRYPTOS:
    bg, bord, txt = _chip_style(sym_k)
    scan = st.session_state.get("scan_scores", {}).get(sym_k, {})
    prob = scan.get("prob_subida", None)
    prob_txt = f'<div style="font-size:0.6rem;opacity:0.8;margin-top:1px;">{prob:.0f}%</div>' if prob else ""
    chip_html += (
        f'<a href="?crypto={sym_k}" target="_self" style="' +
        f'display:block;text-align:center;text-decoration:none;' +
        f'font-family:IBM Plex Mono,monospace;font-size:0.78rem;font-weight:700;' +
        f'padding:0.45rem 0.3rem;border-radius:6px;' +
        f'background:{bg};border:1px solid {bord};color:{txt};' +
        f'-webkit-tap-highlight-color:transparent;'
        f'">{label}{prob_txt}</a>'
    )
chip_html += '</div>'
st.markdown(chip_html, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e2432; margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESOLVER SÍMBOLO Y TRIGGER
# ─────────────────────────────────────────────
if chip_selected:
    symbol_final = chip_selected
    do_analyze   = True
elif sb_selected:
    symbol_final = sb_selected
    do_analyze   = True
elif top_analyze or sb_analyze or auto_refresh:
    symbol_final = top_symbol if top_symbol else sb_symbol
    do_analyze   = True
else:
    symbol_final = ""
    do_analyze   = False

# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    st.markdown("""<script>setTimeout(()=>window.location.reload(),60000);</script>""",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PANTALLA INICIAL
# ─────────────────────────────────────────────
if not do_analyze:
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:3rem;
                    font-weight:700; color:#f5a623;
                    text-shadow:0 0 60px rgba(245,166,35,0.3); margin-bottom:0.4rem;">
            ⚡ 5-MIN SIGNAL
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                    color:#3a4055; letter-spacing:3px; font-weight:700; margin-bottom:2rem;">
            CRYPTO PREDICTOR · KRAKEN API
        </div>
        <div style="display:flex; gap:0.8rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:#0c0e15; border:1px solid #1e2432; border-radius:4px;
                         padding:0.5rem 0.9rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.62rem; color:#8892a4; font-weight:700; letter-spacing:2px;">
                📡 KRAKEN REAL-TIME
            </span>
            <span style="background:#0c0e15; border:1px solid #1e2432; border-radius:4px;
                         padding:0.5rem 0.9rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.62rem; color:#8892a4; font-weight:700; letter-spacing:2px;">
                🔑 SIN API KEY
            </span>
            <span style="background:#0c0e15; border:1px solid #1e2432; border-radius:4px;
                         padding:0.5rem 0.9rem; font-family:'IBM Plex Mono',monospace;
                         font-size:0.62rem; color:#8892a4; font-weight:700; letter-spacing:2px;">
                📊 20 INDICADORES
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                    color:#2a3040; letter-spacing:3px; font-weight:700; margin-top:1.5rem;">
            PULSA UNA CRIPTO ARRIBA ↑
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────
kraken_pair, sym_display = normalizar_symbol(symbol_final)

with st.spinner(f"Conectando con Kraken · {sym_display} · OKX · F&G…"):
    df, df5, book, futures_data, info, error, df15, df1h = descargar_datos(symbol_final)

if error or df is None:
    st.error(f"❌ {error}")
    st.stop()

with st.spinner("Calculando 20 indicadores…"):
    indicadores, señales, puntuaciones, atr_pct, regimen, hurst_val = calcular_indicadores(
        df, df5, book, futures_data, info, df15, df1h)
    pred = calcular_prediccion(puntuaciones, info["precio_actual"], indicadores,
                              regimen=regimen, fng_data=futures_data)

# ── Pre-calcular TODAS las variables antes de cualquier f-string HTML ──
precio         = info["precio_actual"]
cambio_pct     = info["cambio_pct"]
sc             = pred["señal_color"]
acento         = C[sc]
glow           = GLOW[sc]
ts             = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

direccion      = pred["direccion"]
señal_texto    = pred["señal_texto"]
precio_obj     = pred["precio_objetivo"]
mov_est        = pred["mov_estimado"]
atr_disp       = pred["atr_pct"]
score          = pred["score"]
prob_up        = pred["prob_subida"]
prob_dn        = pred["prob_bajada"]
n_alc          = pred["alcistas"]
n_neu          = pred["neutros"]
n_baj          = pred["bajistas"]

flecha_precio  = "&#9650;" if cambio_pct >= 0 else "&#9660;"
color_cambio   = "#00e87a" if cambio_pct >= 0 else "#ff4f6a"
flecha_score   = "&#8593;" if score > 0 else "&#8595;"

precio_fmt     = f"${precio:,.4f}"
precio_obj_fmt = f"${precio_obj:,.4f}"
cambio_fmt     = f"{cambio_pct:+.2f}%"
score_fmt      = f"{score:+.3f}"
mov_fmt        = f"{mov_est:.4f}%"
atr_fmt        = f"{atr_disp:.3f}%"
rng_lo_fmt     = f"${precio * (1 - mov_est / 100):,.4f}"
rng_hi_fmt     = f"${precio * (1 + mov_est / 100):,.4f}"
acento80       = acento + "80"

# ─────────────────────────────────────────────
# ══ HERO ══
# ─────────────────────────────────────────────
st.markdown(f'<style>.hero{{--acc:{acento};--acc-glow:{glow};}}.target-box{{--acc:{acento};}}</style>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════
# FILA 1 — PREDICCIÓN (lo más importante, primero)
# Dirección grande + Prob subida + Prob bajada + Rango
# ══════════════════════════════════════════════
c_up = "#00e87a" if prob_up > 55 else ("#ff4f6a" if prob_up < 45 else "#f5a623")
c_dn = "#ff4f6a" if prob_dn > 55 else ("#00e87a" if prob_dn < 45 else "#f5a623")

p1, p2, p3, p4 = st.columns([3, 2, 2, 2])

with p1:
    # Caja de dirección — texto más grande, es el resultado final
    st.markdown(
        '<div class="hero" style="height:100%;">'
        f'<div class="hero-label">&#9889; {sym_display} &middot; PREDICCI&Oacute;N +5 MINUTOS &middot; {ts}</div>'
        f'<div class="hero-dir" style="font-size:4.5rem;">{direccion}</div>'
        f'<div><span class="hero-badge" style="color:{acento}; border-color:{acento80}; font-size:0.82rem; padding:0.35rem 1rem;">'
        f'{señal_texto}</span></div>'
        '</div>',
        unsafe_allow_html=True
    )

with p2:
    st.markdown(
        '<div class="pbox" style="height:100%; box-sizing:border-box;">'
        '<div class="pbox-lbl" style="font-size:0.68rem; letter-spacing:3px;">PROB SUBIDA</div>'
        f'<div class="pbox-val" style="color:{c_up}; font-size:3rem;">{prob_up:.1f}%</div>'
        '<div class="pbar-track" style="height:7px; margin-top:0.7rem;">'
        f'<div class="pbar-fill" style="width:{prob_up}%; background:{c_up}; box-shadow:0 0 10px {c_up}60;"></div>'
        '</div>'
        '<div class="pbox-sub" style="font-size:0.68rem; margin-top:0.5rem;">pr&oacute;ximos 5 min</div>'
        '</div>',
        unsafe_allow_html=True
    )

with p3:
    st.markdown(
        '<div class="pbox" style="height:100%; box-sizing:border-box;">'
        '<div class="pbox-lbl" style="font-size:0.68rem; letter-spacing:3px;">PROB BAJADA</div>'
        f'<div class="pbox-val" style="color:{c_dn}; font-size:3rem;">{prob_dn:.1f}%</div>'
        '<div class="pbar-track" style="height:7px; margin-top:0.7rem;">'
        f'<div class="pbar-fill" style="width:{prob_dn}%; background:{c_dn}; box-shadow:0 0 10px {c_dn}60;"></div>'
        '</div>'
        '<div class="pbox-sub" style="font-size:0.68rem; margin-top:0.5rem;">pr&oacute;ximos 5 min</div>'
        '</div>',
        unsafe_allow_html=True
    )

with p4:
    st.markdown(
        '<div class="pbox" style="height:100%; box-sizing:border-box;">'
        '<div class="pbox-lbl" style="font-size:0.68rem; letter-spacing:3px;">RANGO ESTIMADO</div>'
        f'<div class="pbox-val" style="color:{acento}; font-size:2.4rem;">&plusmn;{mov_fmt}</div>'
        '<div style="margin-top:0.6rem; font-family:\'IBM Plex Mono\',monospace; font-weight:700; line-height:2.0;">'
        f'<div style="font-size:0.82rem; color:#00e87a;">&#9650; {rng_hi_fmt}</div>'
        f'<div style="font-size:0.82rem; color:#ff4f6a;">&#9660; {rng_lo_fmt}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='margin:0.7rem 0;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# FILA 2 — PRECIO ACTUAL + SCORE + OBJETIVO
# ══════════════════════════════════════════════
h2, h3 = st.columns([3, 2])

with h2:
    st.markdown(
        '<div class="hero">'
        '<div class="hero-label">PRECIO ACTUAL</div>'
        f'<div class="price-big">{precio_fmt}</div>'
        f'<div class="price-chg" style="color:{color_cambio};">'
        f'{flecha_precio} {cambio_fmt} (24h)</div>'
        '<div class="target-box">'
        '<div class="hero-label">PRECIO OBJETIVO +5m</div>'
        f'<div class="target-price" style="color:{acento};">{precio_obj_fmt}</div>'
        f'<div class="target-sub">{flecha_score} {mov_fmt} estimado &middot; ATR {atr_fmt}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with h3:
    st.markdown(
        '<div class="hero">'
        '<div class="hero-label">SCORE COMPUESTO</div>'
        f'<div class="score-big" style="color:{acento}; text-shadow:0 0 30px {glow};">'
        f'{score_fmt}</div>'
        '<div class="score-sub">escala &minus;1 a +1</div>'
        '<div class="counters">'
        f'<span style="color:#00e87a;">&#9650; {n_alc}</span>'
        f'<span style="color:#6b7894;">&#9711; {n_neu}</span>'
        f'<span style="color:#ff4f6a;">&#9660; {n_baj}</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='margin:0.7rem 0;'></div>", unsafe_allow_html=True)

# ── Info de mercado ──
vol_fmt  = f"${info['vol_24h']/1e6:.1f}M" if info["vol_24h"] > 1e6 else f"${info['vol_24h']:,.0f}"
fr_val   = futures_data.get("funding_rate")
fr_fmt   = f"{fr_val*100:+.4f}%" if fr_val is not None else "N/A"
fr_color = "#ff4f6a" if fr_val and fr_val > 0 else ("#00e87a" if fr_val and fr_val < 0 else "#8892a4")
oi_val   = futures_data.get("open_interest")
oi_fmt   = f"{oi_val:,.0f}" if oi_val else "N/A"
hl_fmt   = f"${info['high_24h']:,.4f} / ${info['low_24h']:,.4f}"

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f'<div class="icard"><div class="icard-lbl">VOL 24H</div><div class="icard-val">{vol_fmt}</div></div>',
                unsafe_allow_html=True)
with mc2:
    st.markdown(f'<div class="icard"><div class="icard-lbl">HIGH / LOW 24H</div><div class="icard-val" style="font-size:0.82rem;">{hl_fmt}</div></div>',
                unsafe_allow_html=True)
with mc3:
    st.markdown(f'<div class="icard"><div class="icard-lbl">FUNDING RATE</div><div class="icard-val" style="color:{fr_color};">{fr_fmt}</div></div>',
                unsafe_allow_html=True)
with mc4:
    st.markdown(f'<div class="icard"><div class="icard-lbl">OPEN INTEREST</div><div class="icard-val">{oi_fmt}</div></div>',
                unsafe_allow_html=True)

st.markdown("<div style='margin:0.7rem 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PANEL DE RÉGIMEN Y MODULADORES
# ─────────────────────────────────────────────
REGIME_LABELS = {
    "trending":       ("TENDENCIA",   "#4e9eff", "Indicadores de momentum amplificados"),
    "mean_reverting": ("REVERSIÓN",   "#f5a623", "Osciladores amplificados"),
    "noise":          ("RUIDO/LATERAL","#6b7894","Microestructura amplificada"),
}
reg_label, reg_color, reg_desc = REGIME_LABELS.get(
    regimen, ("?", "#6b7894", "")
)
hurst_fmt    = f"{hurst_val:.3f}"
fng_val_disp = futures_data.get("fng_value")
fng_cls_disp = futures_data.get("fng_class", "N/A")
fng_color    = ("#00e87a" if fng_val_disp and fng_val_disp <= 40 else
                "#ff4f6a" if fng_val_disp and fng_val_disp >= 60 else "#f5a623")
fng_disp     = f"{fng_val_disp} — {fng_cls_disp}" if fng_val_disp else "N/A"

book_src_disp  = futures_data.get("book_source", "kraken").upper()
tf_align_disp  = pred.get("tf_align", 0)
fng_mod_disp   = pred.get("fng_mod", 1.0)
tf_mod_disp    = pred.get("tf_mod", 1.0)
regime_mod_disp= pred.get("regime_mod", 1.0)
score_raw_disp = pred.get("score_raw", score)

st.markdown(
    '<div style="background:#0a0c12; border:1px solid #1e2432; border-left:3px solid #4e9eff;'
    'border-radius:6px; padding:0.9rem 1.2rem; margin-bottom:0.7rem;">'
    '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; letter-spacing:3px;'
    'font-weight:700; color:#8892a4; margin-bottom:0.6rem;">ANÁLISIS DE RÉGIMEN Y MODULADORES</div>'
    '<div style="display:flex; flex-wrap:wrap; gap:1.5rem; align-items:flex-start;">',
    unsafe_allow_html=True
)

rm1, rm2, rm3, rm4, rm5 = st.columns(5)

with rm1:
    st.markdown(
        f'<div class="icard"><div class="icard-lbl">RÉGIMEN (HURST={hurst_fmt})</div>'
        f'<div class="icard-val" style="color:{reg_color}; font-size:0.85rem;">{reg_label}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; color:#6b7894; margin-top:0.2rem;">{reg_desc}</div></div>',
        unsafe_allow_html=True
    )
with rm2:
    st.markdown(
        f'<div class="icard"><div class="icard-lbl">FEAR &amp; GREED</div>'
        f'<div class="icard-val" style="color:{fng_color};">{fng_disp}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; color:#6b7894; margin-top:0.2rem;">Mod F&G: ×{fng_mod_disp:.3f}</div></div>',
        unsafe_allow_html=True
    )
with rm3:
    tf_color = "#00e87a" if tf_align_disp == 2 else ("#f5a623" if tf_align_disp == 1 else "#6b7894")
    tf_txt   = ["Sin alineación", "15m confirma", "15m + 1h confirman"][tf_align_disp]
    st.markdown(
        f'<div class="icard"><div class="icard-lbl">ALINEACIÓN MULTI-TF</div>'
        f'<div class="icard-val" style="color:{tf_color}; font-size:0.85rem;">{tf_txt}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; color:#6b7894; margin-top:0.2rem;">Mod TF: ×{tf_mod_disp:.3f}</div></div>',
        unsafe_allow_html=True
    )
with rm4:
    st.markdown(
        f'<div class="icard"><div class="icard-lbl">ORDER BOOK FUENTE</div>'
        f'<div class="icard-val" style="color:#4e9eff; font-size:0.85rem;">{book_src_disp}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; color:#6b7894; margin-top:0.2rem;">Mod régimen: ×{regime_mod_disp:.3f}</div></div>',
        unsafe_allow_html=True
    )
with rm5:
    raw_color = "#00e87a" if score_raw_disp > 0 else ("#ff4f6a" if score_raw_disp < 0 else "#6b7894")
    st.markdown(
        f'<div class="icard"><div class="icard-lbl">SCORE BRUTO → FINAL</div>'
        f'<div class="icard-val" style="color:{raw_color};">{score_raw_disp:+.3f} → {score_fmt}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.58rem; color:#6b7894; margin-top:0.2rem;">Ajuste total: ×{fng_mod_disp*tf_mod_disp*regime_mod_disp:.3f}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<div style='margin:0.7rem 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_charts, tab_indicators, tab_book = st.tabs(["📈 Gráficos", "📋 Indicadores", "📖 Order Book"])

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
with tab_charts:
    BG   = "#08090c"
    PAN  = "#0c0e15"
    GRID = "#141820"
    CLR  = {"a":"#00e87a","b":"#ff4f6a","p":"#f5a623","blue":"#4e9eff","purple":"#a29bfe"}

    close_  = df["close"]
    high_   = df["high"]
    low_    = df["low"]
    vol_    = df["volume"]

    fig = plt.figure(figsize=(18, 16), facecolor=BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.3)

    def style_ax(ax, title=""):
        ax.set_facecolor(PAN)
        ax.tick_params(colors="#4a5568", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
        if title:
            ax.set_title(title, color="#6b7894", fontsize=9,
                         fontfamily="monospace", pad=5, fontweight="bold")

    # 1. Precio + EMA + BB
    ax1 = fig.add_subplot(gs[0, :])
    c80 = close_.tail(80); e7 = close_.ewm(span=7).mean().tail(80)
    e25 = close_.ewm(span=25).mean().tail(80)
    bm  = close_.rolling(20).mean().tail(80); bs = close_.rolling(20).std().tail(80)
    ax1.fill_between(range(80), (bm+2*bs).values, (bm-2*bs).values, alpha=0.06, color=CLR["p"])
    ax1.plot(range(80), (bm+2*bs).values, color=CLR["p"], lw=0.5, alpha=0.4)
    ax1.plot(range(80), (bm-2*bs).values, color=CLR["p"], lw=0.5, alpha=0.4)
    ax1.plot(range(80), c80.values,  color=CLR["blue"],   lw=1.8, zorder=5, label="Precio")
    ax1.plot(range(80), e7.values,   color=CLR["a"],      lw=1.2, alpha=0.9, label="EMA7")
    ax1.plot(range(80), e25.values,  color=CLR["purple"], lw=1.2, alpha=0.9, label="EMA25")
    ax1.axhline(precio, color="yellow", lw=0.8, ls="--", alpha=0.4)
    ax1.legend(loc="upper left", facecolor=PAN, labelcolor="#8892a4",
               fontsize=7.5, framealpha=0.9, edgecolor=GRID)
    style_ax(ax1, f"PRECIO 1M — {sym_display}  |  EMA7 · EMA25 · BOLLINGER (20)")

    # 2. Volumen
    ax2 = fig.add_subplot(gs[1, :2])
    v50 = vol_.tail(50); va = vol_.rolling(20).mean().tail(50)
    taker = df["taker_buy_base"].tail(50)
    vcols = [CLR["a"] if tb >= v*0.5 else CLR["b"] for tb, v in zip(taker.values, v50.values)]
    ax2.bar(range(len(v50)), v50.values, color=vcols, alpha=0.8, width=0.85)
    ax2.plot(range(len(va)), va.values, color="white", lw=1.2, alpha=0.5, label="MA20")
    ax2.set_xticks([]); ax2.legend(facecolor=PAN, labelcolor="#8892a4", fontsize=7.5, edgecolor=GRID)
    style_ax(ax2, "VOLUMEN  (verde=buy / rojo=sell)")

    # 3. Buy/Sell ratio
    ax3 = fig.add_subplot(gs[1, 2])
    tbq = df["taker_buy_quote"].tail(30)
    tsq = (df["quote_volume"] - df["taker_buy_quote"]).tail(30)
    ratio = tbq / (tbq + tsq) * 100
    rc = [CLR["a"] if v > 50 else CLR["b"] for v in ratio.values]
    ax3.bar(range(len(ratio)), ratio.values - 50, color=rc, alpha=0.8, width=0.85)
    ax3.axhline(0, color="#3a4055", lw=0.8); ax3.set_xticks([])
    ax3.set_ylabel("Buy% − 50", color="#6b7894", fontsize=7)
    style_ax(ax3, "BUY/SELL RATIO")

    # 4. RSI
    ax4 = fig.add_subplot(gs[2, 0])
    dlt  = close_.diff()
    rsi_s = (100 - 100 / (1 + dlt.clip(lower=0).rolling(9).mean() /
                           (-dlt.clip(upper=0)).rolling(9).mean())).tail(60)
    ax4.plot(range(len(rsi_s)), rsi_s.values, color=CLR["blue"], lw=1.5)
    ax4.axhline(70, color=CLR["b"], ls="--", lw=0.8, alpha=0.6)
    ax4.axhline(30, color=CLR["a"], ls="--", lw=0.8, alpha=0.6)
    ax4.fill_between(range(len(rsi_s)), rsi_s.values, 70, where=rsi_s.values>70, alpha=0.12, color=CLR["b"])
    ax4.fill_between(range(len(rsi_s)), rsi_s.values, 30, where=rsi_s.values<30, alpha=0.12, color=CLR["a"])
    ax4.set_ylim(0, 100); ax4.set_xticks([])
    style_ax(ax4, "RSI (9)")

    # 5. MACD
    ax5 = fig.add_subplot(gs[2, 1])
    ml = (close_.ewm(span=5).mean() - close_.ewm(span=13).mean()).tail(60)
    sl = ml.ewm(span=3).mean(); hl2 = ml - sl
    hc = [CLR["a"] if v >= 0 else CLR["b"] for v in hl2.values]
    ax5.bar(range(len(hl2)), hl2.values, color=hc, alpha=0.8, width=0.85)
    ax5.plot(range(len(ml)), ml.values, color=CLR["p"],      lw=1.4, label="MACD")
    ax5.plot(range(len(sl)), sl.values, color=CLR["purple"], lw=1.4, label="Signal")
    ax5.axhline(0, color="#3a4055", lw=0.5); ax5.set_xticks([])
    ax5.legend(facecolor=PAN, labelcolor="#8892a4", fontsize=7, edgecolor=GRID)
    style_ax(ax5, "MACD (5,13,3)")

    # 6. Stochastic
    ax6 = fig.add_subplot(gs[2, 2])
    l5 = low_.rolling(5).min(); h5 = high_.rolling(5).max()
    sk = ((close_ - l5) / (h5 - l5) * 100).rolling(3).mean().tail(60)
    sd = sk.rolling(3).mean()
    ax6.plot(range(len(sk)), sk.values, color=CLR["a"], lw=1.4, label="%K")
    ax6.plot(range(len(sd)), sd.values, color=CLR["b"], lw=1.4, label="%D")
    ax6.axhline(80, color=CLR["b"], ls="--", lw=0.7, alpha=0.5)
    ax6.axhline(20, color=CLR["a"], ls="--", lw=0.7, alpha=0.5)
    ax6.set_ylim(0, 100); ax6.set_xticks([])
    ax6.legend(facecolor=PAN, labelcolor="#8892a4", fontsize=7, edgecolor=GRID)
    style_ax(ax6, "STOCHASTIC (5,3)")

    # 7. Gauge
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.set_facecolor(PAN); ax7.set_aspect("equal")
    theta = np.linspace(np.pi, 0, 300)
    ax7.plot(np.cos(theta), np.sin(theta), color="#141820", lw=22, solid_capstyle="round")
    end_a  = np.pi - (prob_up / 100) * np.pi
    theta2 = np.linspace(np.pi, end_a, 300)
    gc = CLR["a"] if prob_up > 55 else (CLR["b"] if prob_up < 45 else CLR["p"])
    ax7.plot(np.cos(theta2), np.sin(theta2), color=gc, lw=22, solid_capstyle="round")
    ax7.text(0, 0.2, f"{prob_up:.1f}%", ha="center", va="center",
             fontsize=26, fontweight="bold", color=gc, fontfamily="monospace")
    ax7.text(0, -0.05, "PROB SUBIDA", ha="center", va="center",
             fontsize=7.5, color="#8892a4", fontfamily="monospace")
    ax7.text(-1.05, -0.18, "0%",   color="#3a4055", fontsize=7, fontfamily="monospace")
    ax7.text(0.75,  -0.18, "100%", color="#3a4055", fontsize=7, fontfamily="monospace")
    ax7.set_xlim(-1.3, 1.3); ax7.set_ylim(-0.3, 1.2); ax7.axis("off")
    ax7.set_title(f"SEÑAL: {direccion}", color="#6b7894",
                  fontsize=9, fontfamily="monospace", pad=5, fontweight="bold")
    for sp in ax7.spines.values(): sp.set_color(GRID)

    # 8. Scores
    ax8 = fig.add_subplot(gs[3, 1:])
    inds_s = sorted(puntuaciones.items(), key=lambda x: x[1])
    names  = [i[0][:24] for i in inds_s]
    vals   = [i[1] for i in inds_s]
    bc     = [CLR["a"] if v > 0 else (CLR["b"] if v < 0 else "#2a3040") for v in vals]
    ax8.barh(range(len(names)), vals, color=bc, alpha=0.85, height=0.65)
    ax8.set_yticks(range(len(names)))
    ax8.set_yticklabels(names, fontsize=7, color="#8892a4", fontfamily="monospace")
    ax8.axvline(0, color="#2a3040", lw=0.8); ax8.set_xlim(-1.2, 1.2)
    ax8.set_xlabel("← BAJISTA  ·  ALCISTA →", color="#4a5568", fontsize=7.5, fontfamily="monospace")
    style_ax(ax8, "SCORE POR INDICADOR")

    plt.suptitle(f"{sym_display}  ·  ANÁLISIS 5MIN  ·  {ts}",
                 color="#4a5568", fontsize=9, fontfamily="monospace", y=1.01, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
with tab_indicators:
    for bloque, inds_list in BLOQUES_CRYPTO.items():
        st.markdown(f'<div class="blk-title">{bloque}</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        for i, ind in enumerate(inds_list):
            val     = str(indicadores.get(ind, "N/A"))
            senal   = señales.get(ind, ("neutro", "N/A"))
            tipo, texto = senal if isinstance(senal, tuple) else ("neutro", str(senal))
            score_i = puntuaciones.get(ind, 0)
            dc      = C.get(tipo, C["neutro"])
            glow_i  = f"0 0 5px {dc}" if score_i != 0 else "none"
            col     = col_a if i % 2 == 0 else col_b
            # Pre-calcular estilos
            dot_sty = f"background:{dc}; box-shadow:{glow_i};"
            with col:
                st.markdown(
                    f'<div class="ind-row">'
                    f'<div class="ind-dot" style="{dot_sty}"></div>'
                    f'<div class="ind-name">{ind}</div>'
                    f'<div class="ind-val">{val}</div>'
                    f'<div class="ind-sig" style="color:{dc};">{texto}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ORDER BOOK
# ─────────────────────────────────────────────
with tab_book:
    if book and book.get("bids") and book.get("asks"):
        bids_raw = [(float(b[0]), float(b[1])) for b in book["bids"][:15]]
        asks_raw = [(float(a[0]), float(a[1])) for a in book["asks"][:15]]

        bid_total = sum(b[1] for b in bids_raw)
        ask_total = sum(a[1] for a in asks_raw)
        obi       = (bid_total - ask_total) / (bid_total + ask_total) * 100

        # Pre-calcular todo
        base_name     = info["nombre"].split("/")[0]
        spread_val    = asks_raw[0][0] - bids_raw[0][0]
        spread_fmt    = f"{spread_val:+.4f}"
        obi_color     = "#00e87a" if obi > 0 else "#ff4f6a"
        obi_fmt       = f"{obi:+.1f}%"
        bid_tot_fmt   = f"{bid_total:.4f}"
        ask_tot_fmt   = f"{ask_total:.4f}"

        cb, cm, ca = st.columns([5, 2, 5])

        with cb:
            st.markdown(
                f'<div class="blk-title" style="color:#00e87a;">BIDS (COMPRAS) &mdash; {bid_tot_fmt} {base_name}</div>',
                unsafe_allow_html=True
            )
            max_bv = max(b[1] for b in bids_raw)
            for price_b, vol_b in bids_raw:
                bw = vol_b / max_bv * 100
                pb = f"${price_b:,.4f}"; vb = f"{vol_b:.4f}"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.18rem 0;font-family:\'IBM Plex Mono\',monospace;">'
                    f'<div style="background:#00e87a18;border-radius:2px;width:{bw:.0f}%;min-width:4px;height:14px;border-right:2px solid #00e87a;"></div>'
                    f'<div style="font-size:0.7rem;font-weight:700;color:#00e87a;min-width:95px;">{pb}</div>'
                    f'<div style="font-size:0.65rem;font-weight:600;color:#8892a4;">{vb}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with cm:
            st.markdown(
                f'<div style="text-align:center;padding:1rem 0.4rem;">'
                f'<div class="icard-lbl">SPREAD</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.85rem;font-weight:700;color:#f5a623;margin-bottom:1rem;">{spread_fmt}</div>'
                f'<div class="icard-lbl">OBI</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.2rem;font-weight:700;color:{obi_color};">{obi_fmt}</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.58rem;font-weight:700;color:#8892a4;margin-top:0.3rem;">Order Book<br>Imbalance</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with ca:
            st.markdown(
                f'<div class="blk-title" style="color:#ff4f6a;text-align:right;">ASKS (VENTAS) &mdash; {ask_tot_fmt} {base_name}</div>',
                unsafe_allow_html=True
            )
            max_av = max(a[1] for a in asks_raw)
            for price_a, vol_a in asks_raw:
                aw = vol_a / max_av * 100
                pa = f"${price_a:,.4f}"; va = f"{vol_a:.4f}"
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:flex-end;gap:0.5rem;padding:0.18rem 0;font-family:\'IBM Plex Mono\',monospace;">'
                    f'<div style="font-size:0.65rem;font-weight:600;color:#8892a4;">{va}</div>'
                    f'<div style="font-size:0.7rem;font-weight:700;color:#ff4f6a;min-width:95px;text-align:right;">{pa}</div>'
                    f'<div style="background:#ff4f6a18;border-radius:2px;width:{aw:.0f}%;min-width:4px;height:14px;border-left:2px solid #ff4f6a;"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("Order book no disponible para este par.")

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '&#9888; AVISO LEGAL: An&aacute;lisis educativo e informativo. '
    'Las criptomonedas son activos de alto riesgo. '
    'Las predicciones a 5 minutos tienen fiabilidad limitada por naturaleza. '
    'No constituye asesoramiento financiero.'
    '</div>',
    unsafe_allow_html=True
)
