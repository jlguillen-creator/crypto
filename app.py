"""
Crypto Predictor 5min — Streamlit App
Predictor de movimiento de criptomonedas para los próximos 5 minutos
basado en 20 indicadores de microestructura, momentum y derivados.
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
    calcular_prediccion, BLOQUES_CRYPTO, PESOS_CRYPTO,
    normalizar_symbol
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Predictor 5m",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — Estética terminal / trading desk
# Tipografía: IBM Plex Mono + DM Sans
# Paleta: negro profundo + ámbar eléctrico
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #08090c;
}

/* ── Scanline texture overlay ── */
body::before {
    content: '';
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,180,0,0.012) 2px,
        rgba(255,180,0,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Hero card ── */
.hero {
    background: #0c0e14;
    border: 1px solid #1e2230;
    border-top: 2px solid var(--accent, #f5a623);
    border-radius: 4px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    position: relative;
}
.hero-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 4px;
    color: #4a5568;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.hero-direction {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    color: var(--accent, #f5a623);
    text-shadow: 0 0 40px var(--accent-glow, rgba(245,166,35,0.3));
    letter-spacing: -2px;
}
.hero-signal {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border: 1px solid currentColor;
    border-radius: 2px;
    display: inline-block;
    margin-top: 0.5rem;
}

/* ── Price display ── */
.price-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8eaf0;
}
.price-change {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}
.price-target-box {
    background: #0c0e14;
    border: 1px solid #1e2230;
    border-left: 2px solid var(--accent, #f5a623);
    padding: 0.7rem 1rem;
    border-radius: 2px;
    margin-top: 0.6rem;
}

/* ── Probability bars ── */
.prob-box {
    background: #0c0e14;
    border: 1px solid #1e2230;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
}
.prob-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3a4055;
    margin-bottom: 0.5rem;
}
.prob-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
}
.prob-track {
    background: #141620;
    height: 6px;
    border-radius: 1px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 1px;
}

/* ── Score ring ── */
.score-box {
    background: #0c0e14;
    border: 1px solid #1e2230;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.score-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
}

/* ── Counters row ── */
.counter-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}

/* ── Info cards ── */
.info-card {
    background: #0c0e14;
    border: 1px solid #1a1e2a;
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
}
.info-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3a4055;
    margin-bottom: 0.3rem;
}
.info-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c8ccdb;
}

/* ── Indicator rows ── */
.ind-bloque-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    color: #3a4055;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #141620;
    padding-bottom: 0.3rem;
}
.ind-row {
    display: grid;
    grid-template-columns: 6px 160px 1fr auto;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid #0e1018;
}
.ind-dot { width:6px; height:6px; border-radius:50%; }
.ind-name { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#4a5568; }
.ind-val  { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#2d3448; text-align:left; }
.ind-sig  { font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:500; text-align:right; }

/* ── Refresh badge ── */
.refresh-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #3a4055;
    letter-spacing: 2px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #08090c;
    border-right: 1px solid #141620;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #0c0e14;
    border: 1px solid #1e2230;
    border-left: 2px solid #f5a623;
    padding: 0.6rem 1rem;
    font-size: 0.72rem;
    color: #3a4055;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 1.5rem;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────
C = {
    "alcista":      "#00e87a",
    "alcista_leve": "#68d391",
    "bajista":      "#ff4f6a",
    "bajista_leve": "#f5a623",
    "neutro":       "#3a4055",
}
GLOW = {
    "alcista":      "rgba(0,232,122,0.25)",
    "alcista_leve": "rgba(104,211,145,0.15)",
    "bajista":      "rgba(255,79,106,0.25)",
    "bajista_leve": "rgba(245,166,35,0.15)",
    "neutro":       "rgba(58,64,85,0.1)",
}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.62rem;
                letter-spacing:4px; color:#3a4055; margin-bottom:0.5rem;'>
    ⚡ CRYPTO PREDICTOR
    </div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:1.2rem;
                font-weight:700; color:#f5a623; margin-bottom:1rem;'>
    5-MINUTE SIGNAL
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.75rem; color:#3a4055; line-height:1.7; margin-bottom:1rem;'>
    Análisis de 20 indicadores de microestructura, momentum y derivados
    sobre datos de Kraken en tiempo real.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    symbol_input = st.text_input(
        "Par de trading",
        value="BTCUSDT",
        placeholder="BTCUSDT, ETHUSDT, SOLUSDT…",
        help="Formato: BTCUSDT, BTC/USDT o simplemente BTC"
    ).strip()

    auto_refresh = st.checkbox("Auto-refresh cada 60s", value=False)
    analizar_btn = st.button("⚡ ANALIZAR", use_container_width=True, type="primary")

    st.divider()

    # Sugerencias rápidas
    st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.6rem; letter-spacing:3px; color:#3a4055; margin-bottom:0.5rem;'>PARES POPULARES</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    sugerencias = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"]
    for i, sug in enumerate(sugerencias):
        col = col_s1 if i % 2 == 0 else col_s2
        if col.button(sug, key=f"sug_{sug}", use_container_width=True):
            symbol_input = sug
            analizar_btn = True

    st.divider()
    st.markdown("""
    <div style='font-size:0.65rem; color:#2a2f40; line-height:1.8; font-family:"IBM Plex Mono",monospace;'>
    FUENTE: Kraken API<br>
    AUTH: No requerida<br>
    VELAS: 1m + 5m<br>
    ORDER BOOK: Top 20<br>
    FUTUROS: Perpetuos USDT
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    import time
    st.markdown("""
    <script>
    setTimeout(() => window.location.reload(), 60000);
    </script>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PANTALLA INICIAL
# ─────────────────────────────────────────────
if not analizar_btn and not auto_refresh:
    st.markdown("""
    <div style='text-align:center; padding:5rem 2rem;'>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem;
                    letter-spacing:5px; color:#2a2f40; margin-bottom:1rem;'>
            SISTEMA DE ANÁLISIS CRYPTO
        </div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:3.5rem;
                    font-weight:700; color:#f5a623; line-height:1;
                    text-shadow:0 0 60px rgba(245,166,35,0.3); margin-bottom:0.5rem;'>
            ⚡ 5-MIN
        </div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:1.2rem;
                    color:#1e2230; font-weight:600; letter-spacing:3px; margin-bottom:2rem;'>
            CRYPTO PREDICTOR
        </div>
        <div style='display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-bottom:3rem;'>
            <div style='background:#0c0e14; border:1px solid #1a1e2a; border-radius:4px;
                        padding:0.8rem 1.2rem; font-family:"IBM Plex Mono",monospace;
                        font-size:0.7rem; color:#3a4055; letter-spacing:2px;'>
                📡 KRAKEN REAL-TIME
            </div>
            <div style='background:#0c0e14; border:1px solid #1a1e2a; border-radius:4px;
                        padding:0.8rem 1.2rem; font-family:"IBM Plex Mono",monospace;
                        font-size:0.7rem; color:#3a4055; letter-spacing:2px;'>
                🔑 SIN API KEY
            </div>
            <div style='background:#0c0e14; border:1px solid #1a1e2a; border-radius:4px;
                        padding:0.8rem 1.2rem; font-family:"IBM Plex Mono",monospace;
                        font-size:0.7rem; color:#3a4055; letter-spacing:2px;'>
                📊 20 INDICADORES
            </div>
            <div style='background:#0c0e14; border:1px solid #1a1e2a; border-radius:4px;
                        padding:0.8rem 1.2rem; font-family:"IBM Plex Mono",monospace;
                        font-size:0.7rem; color:#3a4055; letter-spacing:2px;'>
                🏦 ORDER BOOK + FUTUROS
            </div>
        </div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.62rem;
                    color:#2a2f40; letter-spacing:3px;'>
            ← SELECCIONA UN PAR EN EL PANEL LATERAL
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────
sym = normalizar_symbol(symbol_input)

with st.spinner(f"Conectando con Kraken · {sym}…"):
    df, df5, book, futures_data, info, error = descargar_datos(symbol_input)

if error or df is None:
    st.error(f"❌ {error}")
    st.stop()

with st.spinner("Calculando 20 indicadores…"):
    indicadores, señales, puntuaciones, atr_pct = calcular_indicadores(
        df, df5, book, futures_data, info)
    pred = calcular_prediccion(puntuaciones, info["precio_actual"], indicadores)

precio     = info["precio_actual"]
cambio_pct = info["cambio_pct"]
sc         = pred["señal_color"]
acento     = C[sc]
glow       = GLOW[sc]
ts         = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

# ─────────────────────────────────────────────
# ══ HERO: PREDICCIÓN 5 MINUTOS ══
# ─────────────────────────────────────────────

st.markdown(f"""
<style>
.hero {{ --accent: {acento}; --accent-glow: {glow}; }}
</style>
<div class="hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:2rem;">

    <!-- Dirección + señal -->
    <div>
      <div class="hero-ticker">⚡ {sym} · PREDICCIÓN +5 MINUTOS · {ts}</div>
      <div class="hero-direction">{pred['direccion']}</div>
      <div>
        <span class="hero-signal" style="color:{acento}; border-color:{acento}60;">
          {pred['señal_texto']}
        </span>
      </div>
    </div>

    <!-- Precio + objetivo -->
    <div>
      <div class="hero-ticker">PRECIO ACTUAL</div>
      <div class="price-main">${precio:,.4f}</div>
      <div class="price-change" style="color:{'#00e87a' if cambio_pct >= 0 else '#ff4f6a'};">
        {'▲' if cambio_pct >= 0 else '▼'} {cambio_pct:+.2f}% (24h)
      </div>
      <div class="price-target-box">
        <div class="hero-ticker">PRECIO OBJETIVO +5m</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
                    font-weight:700; color:{acento};">
          ${pred['precio_objetivo']:,.4f}
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#2a2f40; margin-top:0.2rem;">
          {'↑' if pred['score'] > 0 else '↓'} {pred['mov_estimado']:.4f}% estimado
          · ATR {pred['atr_pct']:.3f}%
        </div>
      </div>
    </div>

    <!-- Score + contadores -->
    <div style="min-width:150px;">
      <div class="hero-ticker">SCORE COMPUESTO</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:3.5rem;
                  font-weight:700; line-height:1; color:{acento};
                  text-shadow: 0 0 30px {glow};">
        {pred['score']:+.3f}
      </div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                  color:#2a2f40; margin-top:0.2rem;">escala −1 a +1</div>
      <div class="counter-row" style="margin-top:0.8rem;">
        <span style="color:#00e87a;">▲ {pred['alcistas']}</span>
        <span style="color:#3a4055;">◎ {pred['neutros']}</span>
        <span style="color:#ff4f6a;">▼ {pred['bajistas']}</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Barras de probabilidad ──
c1, c2, c3 = st.columns(3)

def prob_box(label, pct, color, sublabel=""):
    fill_color = color
    return f"""
    <div class="prob-box">
      <div class="prob-lbl">{label}</div>
      <div class="prob-val" style="color:{fill_color};">{pct:.1f}%</div>
      <div class="prob-track">
        <div class="prob-fill" style="width:{pct}%; background:{fill_color};
             box-shadow: 0 0 8px {fill_color}60;"></div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                  color:#2a2f40; margin-top:0.4rem;">{sublabel}</div>
    </div>
    """

with c1:
    c_up = "#00e87a" if pred["prob_subida"] > 55 else ("#ff4f6a" if pred["prob_subida"] < 45 else "#f5a623")
    st.markdown(prob_box("PROB SUBIDA", pred["prob_subida"], c_up, "próximos 5 min"), unsafe_allow_html=True)

with c2:
    c_dn = "#ff4f6a" if pred["prob_bajada"] > 55 else ("#00e87a" if pred["prob_bajada"] < 45 else "#f5a623")
    st.markdown(prob_box("PROB BAJADA", pred["prob_bajada"], c_dn, "próximos 5 min"), unsafe_allow_html=True)

with c3:
    rng_lo = precio * (1 - pred["mov_estimado"] / 100)
    rng_hi = precio * (1 + pred["mov_estimado"] / 100)
    st.markdown(f"""
    <div class="prob-box">
      <div class="prob-lbl">RANGO ESTIMADO</div>
      <div class="prob-val" style="color:{acento};">±{pred['mov_estimado']:.4f}%</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
                  color:#3a4055; margin-top:0.4rem; line-height:1.8;">
        <span style="color:#00e87a;">${rng_hi:,.4f}</span><br>
        <span style="color:#ff4f6a;">${rng_lo:,.4f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

# ── Info de mercado ──
c1, c2, c3, c4 = st.columns(4)
vol_fmt  = f"${info['vol_24h']/1e6:.1f}M" if info["vol_24h"] > 1e6 else f"${info['vol_24h']:,.0f}"
fr_val   = futures_data.get("funding_rate")
fr_fmt   = f"{fr_val*100:+.4f}%" if fr_val is not None else "N/A"
oi_fmt   = f"{futures_data.get('open_interest', 0):,.0f}" if futures_data.get("open_interest") else "N/A"

with c1:
    st.markdown(f"""<div class="info-card">
    <div class="info-lbl">VOLUMEN 24H</div>
    <div class="info-val">{vol_fmt}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="info-card">
    <div class="info-lbl">HIGH / LOW 24H</div>
    <div class="info-val" style="font-size:0.85rem;">${info['high_24h']:,.4f} / ${info['low_24h']:,.4f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="info-card">
    <div class="info-lbl">FUNDING RATE</div>
    <div class="info-val" style="color:{'#ff4f6a' if fr_val and fr_val > 0 else '#00e87a' if fr_val and fr_val < 0 else '#3a4055'};">{fr_fmt}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="info-card">
    <div class="info-lbl">OPEN INTEREST</div>
    <div class="info-val">{oi_fmt}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_charts, tab_indicators, tab_book = st.tabs(["📈 Gráficos", "📋 Indicadores", "📖 Order Book"])

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
with tab_charts:
    BG   = "#08090c"
    PAN  = "#0c0e14"
    GRID = "#141620"
    CLR  = {"a": "#00e87a", "b": "#ff4f6a", "p": "#f5a623", "blue": "#4e9eff"}

    close  = df["close"]
    high_  = df["high"]
    low_   = df["low"]
    vol    = df["volume"]

    fig = plt.figure(figsize=(18, 16), facecolor=BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.3)

    def style(ax, title=""):
        ax.set_facecolor(PAN)
        ax.tick_params(colors="#2a2f40", labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
        if title:
            ax.set_title(title, color="#3a4055", fontsize=9,
                         fontfamily="monospace", pad=5)

    # ── 1. Precio + EMA7/25 + Bollinger ──
    ax1 = fig.add_subplot(gs[0, :])
    c50 = close.tail(80)
    e7  = close.ewm(span=7).mean().tail(80)
    e25 = close.ewm(span=25).mean().tail(80)
    bm  = close.rolling(20).mean().tail(80)
    bs  = close.rolling(20).std().tail(80)
    ax1.fill_between(bm.index, (bm+2*bs).values, (bm-2*bs).values,
                     alpha=0.06, color=CLR["p"])
    ax1.plot(bm.index, (bm+2*bs).values, color=CLR["p"], lw=0.5, alpha=0.4)
    ax1.plot(bm.index, (bm-2*bs).values, color=CLR["p"], lw=0.5, alpha=0.4)
    ax1.plot(c50.index, c50.values, color=CLR["blue"], lw=1.8, zorder=5, label="Precio")
    ax1.plot(e7.index,  e7.values,  color=CLR["a"],    lw=1.2, alpha=0.9, label="EMA7")
    ax1.plot(e25.index, e25.values, color="#a29bfe",   lw=1.2, alpha=0.9, label="EMA25")
    ax1.axhline(precio, color="yellow", lw=0.8, ls="--", alpha=0.4)
    ax1.legend(loc="upper left", facecolor=PAN, labelcolor="#3a4055",
               fontsize=7.5, framealpha=0.9, edgecolor=GRID)
    style(ax1, f"PRECIO 1M — {sym}  |  EMA7 · EMA25 · BOLLINGER (20)")

    # ── 2. Volumen con color compra/venta ──
    ax2 = fig.add_subplot(gs[1, :2])
    v50   = vol.tail(50)
    va    = vol.rolling(20).mean().tail(50)
    taker = df["taker_buy_base"].tail(50)
    cols  = [CLR["a"] if tb >= v*0.5 else CLR["b"]
             for tb, v in zip(taker.values, v50.values)]
    ax2.bar(range(len(v50)), v50.values, color=cols, alpha=0.8, width=0.85)
    ax2.plot(range(len(va)), va.values, color="white", lw=1.2, alpha=0.6, label="MA20")
    ax2.set_xticks([])
    ax2.legend(facecolor=PAN, labelcolor="#3a4055", fontsize=7.5, edgecolor=GRID)
    style(ax2, "VOLUMEN (verde=taker buy, rojo=taker sell)")

    # ── 3. Buy/Sell ratio acumulado ──
    ax3 = fig.add_subplot(gs[1, 2])
    tbq   = df["taker_buy_quote"].tail(30)
    tsq   = (df["quote_volume"] - df["taker_buy_quote"]).tail(30)
    ratio = (tbq / (tbq + tsq) * 100)
    ratio_c = [CLR["a"] if v > 50 else CLR["b"] for v in ratio.values]
    ax3.bar(range(len(ratio)), ratio.values - 50, color=ratio_c, alpha=0.8, width=0.85)
    ax3.axhline(0, color="#3a4055", lw=0.8)
    ax3.set_xticks([])
    ax3.set_ylabel("Buy% − 50", color="#3a4055", fontsize=7)
    style(ax3, "BUY/SELL RATIO")

    # ── 4. RSI ──
    ax4 = fig.add_subplot(gs[2, 0])
    delta_r = close.diff()
    rsi_s = (100 - 100 / (1 + delta_r.clip(lower=0).rolling(9).mean() /
                            (-delta_r.clip(upper=0)).rolling(9).mean())).tail(60)
    ax4.plot(range(len(rsi_s)), rsi_s.values, color=CLR["blue"], lw=1.5)
    ax4.axhline(70, color=CLR["b"], ls="--", lw=0.8, alpha=0.6)
    ax4.axhline(30, color=CLR["a"], ls="--", lw=0.8, alpha=0.6)
    ax4.fill_between(range(len(rsi_s)), rsi_s.values, 70,
                     where=rsi_s.values > 70, alpha=0.15, color=CLR["b"])
    ax4.fill_between(range(len(rsi_s)), rsi_s.values, 30,
                     where=rsi_s.values < 30, alpha=0.15, color=CLR["a"])
    ax4.set_ylim(0, 100)
    ax4.set_xticks([])
    style(ax4, "RSI (9)")

    # ── 5. MACD ──
    ax5 = fig.add_subplot(gs[2, 1])
    ml = (close.ewm(span=5).mean()  - close.ewm(span=13).mean()).tail(60)
    sl = ml.ewm(span=3).mean()
    hl = (ml - sl)
    hc = [CLR["a"] if v >= 0 else CLR["b"] for v in hl.values]
    ax5.bar(range(len(hl)), hl.values, color=hc, alpha=0.8, width=0.85)
    ax5.plot(range(len(ml)), ml.values, color=CLR["p"],   lw=1.4, label="MACD")
    ax5.plot(range(len(sl)), sl.values, color="#a29bfe", lw=1.4, label="Signal")
    ax5.axhline(0, color="#3a4055", lw=0.5)
    ax5.set_xticks([])
    ax5.legend(facecolor=PAN, labelcolor="#3a4055", fontsize=7, edgecolor=GRID)
    style(ax5, "MACD (5,13,3)")

    # ── 6. Stochastic ──
    ax6 = fig.add_subplot(gs[2, 2])
    l5  = low_.rolling(5).min()
    h5  = high_.rolling(5).max()
    sk  = ((close - l5) / (h5 - l5) * 100).rolling(3).mean().tail(60)
    sd  = sk.rolling(3).mean()
    ax6.plot(range(len(sk)), sk.values, color=CLR["a"],  lw=1.4, label="%K")
    ax6.plot(range(len(sd)), sd.values, color=CLR["b"],  lw=1.4, label="%D")
    ax6.axhline(80, color=CLR["b"], ls="--", lw=0.7, alpha=0.5)
    ax6.axhline(20, color=CLR["a"], ls="--", lw=0.7, alpha=0.5)
    ax6.set_ylim(0, 100)
    ax6.set_xticks([])
    ax6.legend(facecolor=PAN, labelcolor="#3a4055", fontsize=7, edgecolor=GRID)
    style(ax6, "STOCHASTIC (5,3)")

    # ── 7. Gauge probabilidad ──
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.set_facecolor(PAN)
    ax7.set_aspect("equal")
    theta  = np.linspace(np.pi, 0, 300)
    ax7.plot(np.cos(theta), np.sin(theta), color="#141620", lw=22, solid_capstyle="round")
    end_a  = np.pi - (pred["prob_subida"] / 100) * np.pi
    theta2 = np.linspace(np.pi, end_a, 300)
    gc     = CLR["a"] if pred["prob_subida"] > 55 else (CLR["b"] if pred["prob_subida"] < 45 else CLR["p"])
    ax7.plot(np.cos(theta2), np.sin(theta2), color=gc, lw=22, solid_capstyle="round")
    ax7.text(0, 0.2, f"{pred['prob_subida']:.1f}%",
             ha="center", va="center", fontsize=26, fontweight="bold",
             color=gc, fontfamily="monospace")
    ax7.text(0, -0.05, "PROB SUBIDA",
             ha="center", va="center", fontsize=7.5, color="#3a4055", fontfamily="monospace")
    ax7.text(-1.05, -0.18, "0%",  color="#2a2f40", fontsize=7, fontfamily="monospace")
    ax7.text(0.75,  -0.18, "100%", color="#2a2f40", fontsize=7, fontfamily="monospace")
    ax7.set_xlim(-1.3, 1.3); ax7.set_ylim(-0.3, 1.2); ax7.axis("off")
    ax7.set_title(f"SEÑAL: {pred['direccion']}", color="#3a4055",
                  fontsize=9, fontfamily="monospace", pad=5)
    for sp in ax7.spines.values(): sp.set_color(GRID)

    # ── 8. Score por indicador ──
    ax8 = fig.add_subplot(gs[3, 1:])
    inds_sorted = sorted(puntuaciones.items(), key=lambda x: x[1])
    names = [i[0][:24] for i in inds_sorted]
    vals  = [i[1] for i in inds_sorted]
    bc    = [CLR["a"] if v > 0 else (CLR["b"] if v < 0 else "#2a2f40") for v in vals]
    ax8.barh(range(len(names)), vals, color=bc, alpha=0.85, height=0.65)
    ax8.set_yticks(range(len(names)))
    ax8.set_yticklabels(names, fontsize=7, color="#3a4055", fontfamily="monospace")
    ax8.axvline(0, color="#2a2f40", lw=0.8)
    ax8.set_xlim(-1.2, 1.2)
    ax8.set_xlabel("← BAJISTA  ·  ALCISTA →", color="#2a2f40",
                   fontsize=7.5, fontfamily="monospace")
    style(ax8, "SCORE POR INDICADOR")

    plt.suptitle(f"{sym}  ·  ANÁLISIS 5MIN  ·  {ts}",
                 color="#2a2f40", fontsize=9, fontfamily="monospace", y=1.01)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
with tab_indicators:
    for bloque, inds_list in BLOQUES_CRYPTO.items():
        st.markdown(f'<div class="ind-bloque-title">{bloque}</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        for i, ind in enumerate(inds_list):
            val   = indicadores.get(ind, "N/A")
            senal = señales.get(ind, ("neutro", "N/A"))
            tipo, texto = senal if isinstance(senal, tuple) else ("neutro", str(senal))
            score = puntuaciones.get(ind, 0)
            peso  = PESOS_CRYPTO.get(ind, 1)
            dc    = C.get(tipo, C["neutro"])
            sc_c  = C.get(tipo, C["neutro"])
            col   = col_a if i % 2 == 0 else col_b
            with col:
                st.markdown(f"""
                <div class="ind-row">
                  <div class="ind-dot" style="background:{dc};
                       box-shadow:{'0 0 4px ' + dc if score != 0 else 'none'};"></div>
                  <div class="ind-name">{ind}</div>
                  <div class="ind-val">{val}</div>
                  <div class="ind-sig" style="color:{sc_c};">{texto}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ORDER BOOK
# ─────────────────────────────────────────────
with tab_book:
    if book and "bids" in book and "asks" in book:
        bids_raw = [(float(b[0]), float(b[1])) for b in book["bids"][:15]]
        asks_raw = [(float(a[0]), float(a[1])) for a in book["asks"][:15]]

        bid_total = sum(b[1] for b in bids_raw)
        ask_total = sum(a[1] for a in asks_raw)
        obi = (bid_total - ask_total) / (bid_total + ask_total) * 100

        col_b, col_mid, col_a = st.columns([5, 2, 5])

        with col_b:
            st.markdown(f"""
            <div class="ind-bloque-title" style="color:#00e87a;">
              BIDS (COMPRAS) — {bid_total:.4f} {sym.replace('USDT','')}
            </div>""", unsafe_allow_html=True)
            max_bid_vol = max(b[1] for b in bids_raw)
            for price_b, vol_b in bids_raw:
                bar_w = vol_b / max_bid_vol * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.5rem;
                            padding:0.2rem 0; font-family:'IBM Plex Mono',monospace;">
                  <div style="background:#00e87a18; border-radius:2px;
                              width:{bar_w:.0f}%; min-width:4px; height:16px;
                              border-right:2px solid #00e87a;"></div>
                  <div style="font-size:0.72rem; color:#00e87a; min-width:90px;">${price_b:,.4f}</div>
                  <div style="font-size:0.65rem; color:#3a4055;">{vol_b:.4f}</div>
                </div>""", unsafe_allow_html=True)

        with col_mid:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem 0.5rem;">
              <div class="info-lbl">SPREAD</div>
              <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
                          color:#f5a623; margin-bottom:1rem;">
                {(asks_raw[0][0] - bids_raw[0][0]):+.4f}
              </div>
              <div class="info-lbl">OBI</div>
              <div style="font-family:'IBM Plex Mono',monospace; font-size:1.2rem;
                          font-weight:700; color:{'#00e87a' if obi > 0 else '#ff4f6a'};">
                {obi:+.1f}%
              </div>
              <div style="font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
                          color:#2a2f40; margin-top:0.3rem;">Order Book<br>Imbalance</div>
            </div>
            """, unsafe_allow_html=True)

        with col_a:
            st.markdown(f"""
            <div class="ind-bloque-title" style="color:#ff4f6a; text-align:right;">
              ASKS (VENTAS) — {ask_total:.4f} {sym.replace('USDT','')}
            </div>""", unsafe_allow_html=True)
            max_ask_vol = max(a[1] for a in asks_raw)
            for price_a, vol_a in asks_raw:
                bar_w = vol_a / max_ask_vol * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:flex-end;
                            gap:0.5rem; padding:0.2rem 0;
                            font-family:'IBM Plex Mono',monospace;">
                  <div style="font-size:0.65rem; color:#3a4055;">{vol_a:.4f}</div>
                  <div style="font-size:0.72rem; color:#ff4f6a; min-width:90px;
                              text-align:right;">${price_a:,.4f}</div>
                  <div style="background:#ff4f6a18; border-radius:2px;
                              width:{bar_w:.0f}%; min-width:4px; height:16px;
                              border-left:2px solid #ff4f6a;"></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Order book no disponible para este par.")

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠ AVISO LEGAL: Análisis educativo e informativo. Las criptomonedas son activos
de alto riesgo y su precio puede variar drásticamente en segundos. Las predicciones
a 5 minutos tienen una fiabilidad limitada inherente. No inviertas dinero que no
puedas permitirte perder. No constituye asesoramiento financiero.
</div>
""", unsafe_allow_html=True)
