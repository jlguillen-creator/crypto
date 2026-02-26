"""
Módulo de lógica central — Crypto Predictor 5min
Fuente de datos: Binance Public API (sin API key)
20 indicadores adaptados a trading intraday de criptomonedas
"""

import requests
import pandas as pd
import numpy as np
import warnings
import ssl
import os
import urllib3
from datetime import datetime, timezone

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

BINANCE_BASE    = "https://api.binance.com/api/v3"
BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1"
BINANCE_FDATA   = "https://fapi.binance.com/futures/data"


# ─────────────────────────────────────────────
# CLIENTE BINANCE (sin autenticación)
# ─────────────────────────────────────────────

# Proxy opcional — configura si estás detrás de un proxy corporativo
# Ejemplo: HTTPS_PROXY=http://proxy.empresa.com:8080
_PROXIES = None
if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
    _proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    _PROXIES = {"https": _proxy_url, "http": _proxy_url}

def _get(url, params=None):
    try:
        r = requests.get(url, params=params, verify=False, timeout=10,
                         proxies=_PROXIES)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 451:
            # Binance bloqueada por región — marcar para fallback
            raise ConnectionError("BINANCE_GEO_BLOCKED")
        return None
    except ConnectionError:
        raise
    except Exception:
        return None


def _get_safe(url, params=None):
    """Wrapper que nunca lanza excepción — devuelve None si falla."""
    try:
        return _get(url, params)
    except Exception:
        return None


def normalizar_symbol(symbol: str) -> str:
    """Convierte BTC, BTCUSDT, BTC/USDT → BTCUSDT"""
    s = symbol.upper().replace("/", "").replace("-", "")
    if not s.endswith("USDT") and not s.endswith("BUSD") and not s.endswith("BTC"):
        s = s + "USDT"
    return s


# ─────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────

def _test_conectividad():
    """Comprueba si Binance es alcanzable. Devuelve (ok, mensaje)."""
    try:
        r = requests.get(f"{BINANCE_BASE}/ping", verify=False, timeout=6,
                         proxies=_PROXIES)
        if r.status_code == 200:
            return True, ""
        return False, f"Binance respondió HTTP {r.status_code}"
    except Exception as e:
        msg = str(e)
        if "Max retries" in msg or "NameResolution" in msg or "ConnectionError" in msg:
            return False, (
                "No se puede conectar con Binance. Posibles causas:\n"
                "• Binance bloqueada en tu red/region\n"
                "• Proxy corporativo: configura HTTPS_PROXY\n"
                "• Sin acceso a internet\n\n"
                "En Streamlit Cloud funciona sin restricciones."
            )
        return False, f"Error de red: {msg[:200]}"


def descargar_datos(symbol: str):
    sym = normalizar_symbol(symbol)

    # ── Test de conectividad previo ──
    ok, err_msg = _test_conectividad()
    if not ok:
        return None, None, None, None, None, err_msg

    # ── Velas 1 minuto: últimas 100 (para calcular indicadores) ──
    klines_raw = _get_safe(f"{BINANCE_BASE}/klines",
                           {"symbol": sym, "interval": "1m", "limit": 100})
    if not klines_raw:
        # Distinguir símbolo inválido de fallo de red
        exchange_info = _get_safe(f"{BINANCE_BASE}/exchangeInfo")
        if exchange_info:
            symbols_validos = [s["symbol"] for s in exchange_info.get("symbols", [])]
            if sym not in symbols_validos:
                sugerencia = next((s for s in symbols_validos if s.startswith(sym[:3])), None)
                msg = f"Símbolo '{sym}' no encontrado en Binance."
                if sugerencia:
                    msg += f" ¿Quisiste decir {sugerencia}?"
                return None, None, None, None, None, msg
        return None, None, None, None, None, f"No se pudieron obtener datos para {sym}. Intenta de nuevo."

    df = pd.DataFrame(klines_raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    for col in ["open","high","low","close","volume","quote_volume",
                "taker_buy_base","taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)

    # ── Velas 5 minutos: últimas 50 (contexto más amplio) ──
    klines_5m = _get_safe(f"{BINANCE_BASE}/klines",
                     {"symbol": sym, "interval": "5m", "limit": 50})
    df5 = None
    if klines_5m:
        df5 = pd.DataFrame(klines_5m, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy_base",
            "taker_buy_quote","ignore"
        ])
        df5["open_time"] = pd.to_datetime(df5["open_time"], unit="ms")
        df5 = df5.set_index("open_time")
        for col in ["open","high","low","close","volume","quote_volume"]:
            df5[col] = df5[col].astype(float)

    # ── Order book (depth 20) ──
    book = _get_safe(f"{BINANCE_BASE}/depth", {"symbol": sym, "limit": 20})

    # ── Ticker 24h ──
    ticker = _get_safe(f"{BINANCE_BASE}/ticker/24hr", {"symbol": sym})

    # ── Precio actual ──
    price_raw = _get_safe(f"{BINANCE_BASE}/ticker/price", {"symbol": sym})
    precio_actual = float(price_raw["price"]) if price_raw else df["close"].iloc[-1]

    # ── Datos de futuros (funding rate, open interest) ── opcional
    futures_data = {}
    # Intentar símbolo perpetuo
    perp_sym = sym if sym.endswith("USDT") else sym + "USDT"
    funding = _get_safe(f"{BINANCE_FUTURES}/premiumIndex", {"symbol": perp_sym})
    if funding:
        futures_data["funding_rate"]    = float(funding.get("lastFundingRate", 0))
        futures_data["mark_price"]      = float(funding.get("markPrice", precio_actual))
        futures_data["index_price"]     = float(funding.get("indexPrice", precio_actual))

    oi = _get_safe(f"{BINANCE_FUTURES}/openInterest", {"symbol": perp_sym})
    if oi:
        futures_data["open_interest"] = float(oi.get("openInterest", 0))

    # OI histórico (cambio)
    oi_hist = _get_safe(f"{BINANCE_FDATA}/openInterestHist",
                   {"symbol": perp_sym, "period": "5m", "limit": 6})
    if oi_hist and isinstance(oi_hist, list) and len(oi_hist) >= 2:
        oi_vals = [float(x["sumOpenInterest"]) for x in oi_hist]
        futures_data["oi_change_pct"] = (oi_vals[-1] / oi_vals[0] - 1) * 100 if oi_vals[0] else 0

    # Long/Short ratio
    ls = _get_safe(f"{BINANCE_FDATA}/globalLongShortAccountRatio",
              {"symbol": perp_sym, "period": "5m", "limit": 1})
    if ls and isinstance(ls, list) and ls:
        futures_data["long_ratio"]  = float(ls[0].get("longAccount", 0.5))
        futures_data["short_ratio"] = float(ls[0].get("shortAccount", 0.5))

    info = {
        "symbol":        sym,
        "nombre":        sym,
        "precio_actual": precio_actual,
        "precio_prev":   float(ticker["prevClosePrice"]) if ticker else df["close"].iloc[-2],
        "cambio_pct":    float(ticker["priceChangePercent"]) if ticker else 0,
        "vol_24h":       float(ticker["quoteVolume"]) if ticker else 0,
        "high_24h":      float(ticker["highPrice"]) if ticker else 0,
        "low_24h":       float(ticker["lowPrice"]) if ticker else 0,
    }

    return df, df5, book, futures_data, info, None


# ─────────────────────────────────────────────
# 20 INDICADORES PARA 5 MINUTOS
# ─────────────────────────────────────────────

def calcular_indicadores(df, df5, book, futures_data, info):
    indicadores  = {}
    señales      = {}
    puntuaciones = {}

    close   = df["close"]
    high    = df["high"]
    low     = df["low"]
    volume  = df["volume"]
    precio  = info["precio_actual"]

    # ── 1. RSI (9) — rápido para 1m ──
    delta    = close.diff()
    avg_g    = delta.clip(lower=0).rolling(9).mean()
    avg_l    = (-delta.clip(upper=0)).rolling(9).mean()
    rsi      = (100 - 100 / (1 + avg_g / avg_l)).iloc[-1]
    indicadores["RSI (9)"] = round(rsi, 1)
    if rsi < 30:
        señales["RSI (9)"] = ("alcista", f"Sobreventa ({rsi:.1f})")
        puntuaciones["RSI (9)"] = 1.0
    elif rsi > 70:
        señales["RSI (9)"] = ("bajista", f"Sobrecompra ({rsi:.1f})")
        puntuaciones["RSI (9)"] = -1.0
    elif rsi < 45:
        señales["RSI (9)"] = ("alcista_leve", f"Zona baja ({rsi:.1f})")
        puntuaciones["RSI (9)"] = 0.4
    elif rsi > 55:
        señales["RSI (9)"] = ("bajista_leve", f"Zona alta ({rsi:.1f})")
        puntuaciones["RSI (9)"] = -0.4
    else:
        señales["RSI (9)"] = ("neutro", f"Neutro ({rsi:.1f})")
        puntuaciones["RSI (9)"] = 0.0

    # ── 2. MACD (5, 13, 3) — parámetros intraday ──
    macd_line   = close.ewm(span=5).mean()  - close.ewm(span=13).mean()
    signal_line = macd_line.ewm(span=3).mean()
    hist        = (macd_line - signal_line)
    m_val, s_val, h_val = macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
    h_prev = hist.iloc[-2]
    indicadores["MACD (5,13,3)"] = f"{m_val:.4f} / {s_val:.4f}"
    if h_val > 0 and h_val > h_prev:
        señales["MACD (5,13,3)"] = ("alcista", "Histograma subiendo")
        puntuaciones["MACD (5,13,3)"] = 1.0
    elif h_val > 0 and h_val <= h_prev:
        señales["MACD (5,13,3)"] = ("alcista_leve", "MACD+ perdiendo fuerza")
        puntuaciones["MACD (5,13,3)"] = 0.3
    elif h_val < 0 and h_val < h_prev:
        señales["MACD (5,13,3)"] = ("bajista", "Histograma bajando")
        puntuaciones["MACD (5,13,3)"] = -1.0
    elif h_val < 0 and h_val >= h_prev:
        señales["MACD (5,13,3)"] = ("bajista_leve", "MACD- perdiendo fuerza")
        puntuaciones["MACD (5,13,3)"] = -0.3
    else:
        señales["MACD (5,13,3)"] = ("neutro", "Cruce en zona 0")
        puntuaciones["MACD (5,13,3)"] = 0.0

    # ── 3. EMA 7 / EMA 25 ──
    ema7  = close.ewm(span=7).mean().iloc[-1]
    ema25 = close.ewm(span=25).mean().iloc[-1]
    ema7_prev  = close.ewm(span=7).mean().iloc[-2]
    ema25_prev = close.ewm(span=25).mean().iloc[-2]
    indicadores["EMA 7/25"] = f"{ema7:.4f} / {ema25:.4f}"
    if ema7 > ema25 and precio > ema7:
        señales["EMA 7/25"] = ("alcista", "Precio > EMA7 > EMA25")
        puntuaciones["EMA 7/25"] = 1.0
    elif ema7 > ema25 and precio < ema7:
        señales["EMA 7/25"] = ("alcista_leve", "EMA7 > EMA25, retroceso")
        puntuaciones["EMA 7/25"] = 0.3
    elif ema7 < ema25 and precio < ema7:
        señales["EMA 7/25"] = ("bajista", "Precio < EMA7 < EMA25")
        puntuaciones["EMA 7/25"] = -1.0
    elif ema7 < ema25 and precio > ema7:
        señales["EMA 7/25"] = ("bajista_leve", "EMA7 < EMA25, rebote")
        puntuaciones["EMA 7/25"] = -0.3
    else:
        señales["EMA 7/25"] = ("neutro", "EMAs entrelazadas")
        puntuaciones["EMA 7/25"] = 0.0

    # ── 4. Bollinger Bands (20, 2) ──
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]
    bb_mid_v = bb_mid.iloc[-1]
    pct_b    = (precio - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
    bw       = (bb_upper - bb_lower) / bb_mid_v * 100  # Bandwidth
    indicadores["Bollinger %B"] = f"{pct_b:.1f}% (BW {bw:.2f}%)"
    if pct_b < 5:
        señales["Bollinger %B"] = ("alcista", f"Banda inferior ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = 1.0
    elif pct_b > 95:
        señales["Bollinger %B"] = ("bajista", f"Banda superior ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = -1.0
    elif pct_b < 35:
        señales["Bollinger %B"] = ("alcista_leve", f"Zona baja ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = 0.4
    elif pct_b > 65:
        señales["Bollinger %B"] = ("bajista_leve", f"Zona alta ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = -0.4
    else:
        señales["Bollinger %B"] = ("neutro", f"Centro banda ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = 0.0

    # ── 5. Stochastic (5, 3) — ultrarrápido ──
    low5  = low.rolling(5).min()
    high5 = high.rolling(5).max()
    stoch_k = ((close - low5) / (high5 - low5) * 100).rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    k, d = stoch_k.iloc[-1], stoch_d.iloc[-1]
    indicadores["Stochastic (5,3)"] = f"K={k:.1f} D={d:.1f}"
    if k < 20 and d < 20:
        señales["Stochastic (5,3)"] = ("alcista", f"Sobreventa K={k:.0f}")
        puntuaciones["Stochastic (5,3)"] = 1.0
    elif k > 80 and d > 80:
        señales["Stochastic (5,3)"] = ("bajista", f"Sobrecompra K={k:.0f}")
        puntuaciones["Stochastic (5,3)"] = -1.0
    elif k > d and k < 50:
        señales["Stochastic (5,3)"] = ("alcista_leve", "K cruza D desde abajo")
        puntuaciones["Stochastic (5,3)"] = 0.5
    elif k < d and k > 50:
        señales["Stochastic (5,3)"] = ("bajista_leve", "K cruza D desde arriba")
        puntuaciones["Stochastic (5,3)"] = -0.5
    else:
        señales["Stochastic (5,3)"] = ("neutro", f"Zona media K={k:.0f}")
        puntuaciones["Stochastic (5,3)"] = 0.0

    # ── 6. Williams %R (14) ──
    hh = high.rolling(14).max()
    ll = low.rolling(14).min()
    wr = ((hh - close) / (hh - ll) * -100).iloc[-1]
    indicadores["Williams %R"] = f"{wr:.1f}"
    if wr < -80:
        señales["Williams %R"] = ("alcista", f"Sobreventa ({wr:.0f})")
        puntuaciones["Williams %R"] = 1.0
    elif wr > -20:
        señales["Williams %R"] = ("bajista", f"Sobrecompra ({wr:.0f})")
        puntuaciones["Williams %R"] = -1.0
    else:
        señales["Williams %R"] = ("neutro", f"Zona media ({wr:.0f})")
        puntuaciones["Williams %R"] = 0.0

    # ── 7. ATR (9) — volatilidad real esperada 5m ──
    tr     = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr    = tr.rolling(9).mean().iloc[-1]
    atr_pct = atr / precio * 100
    indicadores["ATR (9)"] = f"±{atr:.4f} (±{atr_pct:.3f}%)"
    señales["ATR (9)"]     = ("neutro", f"Volatilidad: ±{atr_pct:.3f}% por vela")
    puntuaciones["ATR (9)"] = 0.0

    # ── 8. Rate of Change — 5 velas (momentum puro) ──
    roc5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    roc3 = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
    indicadores["Rate of Change"] = f"3m: {roc3:+.3f}% | 5m: {roc5:+.3f}%"
    if roc5 > 0.15 and roc3 > 0:
        señales["Rate of Change"] = ("alcista", f"Momentum positivo +{roc5:.3f}%")
        puntuaciones["Rate of Change"] = min(1.0, roc5 / 0.3)
    elif roc5 < -0.15 and roc3 < 0:
        señales["Rate of Change"] = ("bajista", f"Momentum negativo {roc5:.3f}%")
        puntuaciones["Rate of Change"] = max(-1.0, roc5 / 0.3)
    else:
        señales["Rate of Change"] = ("neutro", f"Sin momentum claro ({roc5:+.3f}%)")
        puntuaciones["Rate of Change"] = 0.0

    # ── 9. OBV Tendencia ──
    obv        = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema    = obv.ewm(span=10).mean()
    obv_trend  = obv.iloc[-1] - obv.iloc[-5]
    obv_vs_ema = obv.iloc[-1] - obv_ema.iloc[-1]
    indicadores["OBV"] = f"Δ5m: {obv_trend:+.0f}"
    if obv_vs_ema > 0 and obv_trend > 0:
        señales["OBV"] = ("alcista", "OBV > EMA y subiendo")
        puntuaciones["OBV"] = 1.0
    elif obv_vs_ema < 0 and obv_trend < 0:
        señales["OBV"] = ("bajista", "OBV < EMA y bajando")
        puntuaciones["OBV"] = -1.0
    else:
        señales["OBV"] = ("neutro", "OBV mixto")
        puntuaciones["OBV"] = 0.0

    # ── 10. VWAP Desviación ──
    vwap_num   = (close * volume).cumsum()
    vwap_den   = volume.cumsum()
    vwap       = (vwap_num / vwap_den).iloc[-1]
    vwap_dev   = (precio - vwap) / vwap * 100
    indicadores["VWAP Desviación"] = f"VWAP={vwap:.4f} ({vwap_dev:+.3f}%)"
    if vwap_dev > 0.1:
        señales["VWAP Desviación"] = ("bajista_leve", f"Precio {vwap_dev:+.2f}% sobre VWAP")
        puntuaciones["VWAP Desviación"] = -0.5
    elif vwap_dev < -0.1:
        señales["VWAP Desviación"] = ("alcista_leve", f"Precio {vwap_dev:+.2f}% bajo VWAP")
        puntuaciones["VWAP Desviación"] = 0.5
    else:
        señales["VWAP Desviación"] = ("neutro", f"Precio ≈ VWAP ({vwap_dev:+.3f}%)")
        puntuaciones["VWAP Desviación"] = 0.0

    # ── 11. Volumen Relativo + Tendencia ──
    vol_ma   = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
    cambio_1m = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    indicadores["Volumen Relativo"] = f"{vol_ratio:.2f}x (vs media 20m)"
    if vol_ratio > 2.0 and cambio_1m > 0:
        señales["Volumen Relativo"] = ("alcista", f"Vol {vol_ratio:.1f}x en vela alcista")
        puntuaciones["Volumen Relativo"] = 1.0
    elif vol_ratio > 2.0 and cambio_1m < 0:
        señales["Volumen Relativo"] = ("bajista", f"Vol {vol_ratio:.1f}x en vela bajista")
        puntuaciones["Volumen Relativo"] = -1.0
    elif vol_ratio > 1.3 and cambio_1m > 0:
        señales["Volumen Relativo"] = ("alcista_leve", f"Vol elevado alcista ({vol_ratio:.1f}x)")
        puntuaciones["Volumen Relativo"] = 0.5
    elif vol_ratio > 1.3 and cambio_1m < 0:
        señales["Volumen Relativo"] = ("bajista_leve", f"Vol elevado bajista ({vol_ratio:.1f}x)")
        puntuaciones["Volumen Relativo"] = -0.5
    else:
        señales["Volumen Relativo"] = ("neutro", f"Volumen normal ({vol_ratio:.1f}x)")
        puntuaciones["Volumen Relativo"] = 0.0

    # ── 12. Patrón Vela Última (1m) ──
    o, c, h, l = df["open"].iloc[-1], close.iloc[-1], high.iloc[-1], low.iloc[-1]
    body   = abs(c - o)
    rango  = h - l
    ls     = min(o, c) - l   # sombra inferior
    us     = h - max(o, c)   # sombra superior
    patron, p_score = "Neutro", 0.0
    if rango > 0:
        if body > rango * 0.8 and c > o:
            patron, p_score = "Marubozu alcista", 1.0
        elif body > rango * 0.8 and c < o:
            patron, p_score = "Marubozu bajista", -1.0
        elif ls > body * 2 and us < body * 0.5:
            patron, p_score = "Hammer", 1.0
        elif us > body * 2 and ls < body * 0.5:
            patron, p_score = "Shooting Star", -1.0
        elif body < rango * 0.1:
            patron, p_score = "Doji", 0.0
        elif c > o:
            patron, p_score = "Alcista", 0.5
        else:
            patron, p_score = "Bajista", -0.5
    indicadores["Patrón Vela 1m"] = patron
    tipo_v = "alcista" if p_score > 0.5 else ("bajista" if p_score < -0.5 else
             "alcista_leve" if p_score > 0 else ("bajista_leve" if p_score < 0 else "neutro"))
    señales["Patrón Vela 1m"] = (tipo_v, patron)
    puntuaciones["Patrón Vela 1m"] = p_score

    # ── 13. Order Book Imbalance ──
    obi_score, obi_texto = 0.0, "N/A"
    if book and "bids" in book and "asks" in book:
        bids = book["bids"][:10]
        asks = book["asks"][:10]
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total   = bid_vol + ask_vol
        obi     = (bid_vol - ask_vol) / total * 100 if total > 0 else 0
        obi_texto = f"Bids {bid_vol:.2f} / Asks {ask_vol:.2f} ({obi:+.1f}%)"
        indicadores["Order Book Imbalance"] = f"OBI={obi:+.1f}%"
        if obi > 15:
            señales["Order Book Imbalance"] = ("alcista", f"Presión compradora ({obi:+.0f}%)")
            puntuaciones["Order Book Imbalance"] = min(1.0, obi / 30)
        elif obi < -15:
            señales["Order Book Imbalance"] = ("bajista", f"Presión vendedora ({obi:+.0f}%)")
            puntuaciones["Order Book Imbalance"] = max(-1.0, obi / 30)
        else:
            señales["Order Book Imbalance"] = ("neutro", f"Equilibrado ({obi:+.0f}%)")
            puntuaciones["Order Book Imbalance"] = obi / 100
    else:
        indicadores["Order Book Imbalance"] = "N/A"
        señales["Order Book Imbalance"] = ("neutro", "Sin datos")
        puntuaciones["Order Book Imbalance"] = 0.0

    # ── 14. Bid/Ask Spread ──
    spread_score, spread_txt = 0.0, "N/A"
    if book and "bids" in book and "asks" in book and book["bids"] and book["asks"]:
        best_bid = float(book["bids"][0][0])
        best_ask = float(book["asks"][0][0])
        spread   = (best_ask - best_bid) / best_bid * 100
        indicadores["Bid/Ask Spread"] = f"{spread:.4f}%"
        if spread < 0.01:
            señales["Bid/Ask Spread"] = ("alcista_leve", f"Spread ajustado ({spread:.4f}%)")
            puntuaciones["Bid/Ask Spread"] = 0.2
        elif spread > 0.05:
            señales["Bid/Ask Spread"] = ("bajista_leve", f"Spread amplio ({spread:.4f}%)")
            puntuaciones["Bid/Ask Spread"] = -0.3
        else:
            señales["Bid/Ask Spread"] = ("neutro", f"Spread normal ({spread:.4f}%)")
            puntuaciones["Bid/Ask Spread"] = 0.0
    else:
        indicadores["Bid/Ask Spread"] = "N/A"
        señales["Bid/Ask Spread"] = ("neutro", "Sin datos")
        puntuaciones["Bid/Ask Spread"] = 0.0

    # ── 15. Buy/Sell Ratio (Taker) ──
    taker_buy  = df["taker_buy_quote"].tail(10).sum()
    taker_sell = (df["quote_volume"] - df["taker_buy_quote"]).tail(10).sum()
    total_taker = taker_buy + taker_sell
    bs_ratio    = taker_buy / total_taker * 100 if total_taker > 0 else 50
    indicadores["Buy/Sell Ratio"] = f"{bs_ratio:.1f}% compras (10m)"
    if bs_ratio > 60:
        señales["Buy/Sell Ratio"] = ("alcista", f"Dominan compradores ({bs_ratio:.0f}%)")
        puntuaciones["Buy/Sell Ratio"] = min(1.0, (bs_ratio - 50) / 25)
    elif bs_ratio < 40:
        señales["Buy/Sell Ratio"] = ("bajista", f"Dominan vendedores ({bs_ratio:.0f}%)")
        puntuaciones["Buy/Sell Ratio"] = max(-1.0, (bs_ratio - 50) / 25)
    else:
        señales["Buy/Sell Ratio"] = ("neutro", f"Equilibrio compra/venta ({bs_ratio:.0f}%)")
        puntuaciones["Buy/Sell Ratio"] = (bs_ratio - 50) / 50

    # ── 16. Trades por minuto (actividad de mercado) ──
    trades_pm   = df["trades"].tail(5).mean()
    trades_max  = df["trades"].max()
    trades_pct  = trades_pm / trades_max * 100 if trades_max > 0 else 50
    indicadores["Actividad Trades"] = f"{trades_pm:.0f} trades/min (media 5m)"
    if trades_pct > 70:
        cambio_5m = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        if cambio_5m > 0:
            señales["Actividad Trades"] = ("alcista", f"Alta actividad en subida ({trades_pct:.0f}%)")
            puntuaciones["Actividad Trades"] = 0.7
        else:
            señales["Actividad Trades"] = ("bajista", f"Alta actividad en bajada ({trades_pct:.0f}%)")
            puntuaciones["Actividad Trades"] = -0.7
    else:
        señales["Actividad Trades"] = ("neutro", f"Actividad normal ({trades_pct:.0f}%)")
        puntuaciones["Actividad Trades"] = 0.0

    # ── 17. Funding Rate (futuros perpetuos) ──
    if futures_data.get("funding_rate") is not None:
        fr    = futures_data["funding_rate"] * 100
        indicadores["Funding Rate"] = f"{fr:+.4f}%"
        if fr > 0.05:
            señales["Funding Rate"] = ("bajista_leve", f"FR positivo alto → longs pagando ({fr:+.4f}%)")
            puntuaciones["Funding Rate"] = -0.5
        elif fr < -0.05:
            señales["Funding Rate"] = ("alcista_leve", f"FR negativo → shorts pagando ({fr:+.4f}%)")
            puntuaciones["Funding Rate"] = 0.5
        else:
            señales["Funding Rate"] = ("neutro", f"FR neutro ({fr:+.4f}%)")
            puntuaciones["Funding Rate"] = 0.0
    else:
        indicadores["Funding Rate"] = "N/A (spot)"
        señales["Funding Rate"] = ("neutro", "Solo disponible en futuros")
        puntuaciones["Funding Rate"] = 0.0

    # ── 18. Open Interest Cambio ──
    if "oi_change_pct" in futures_data:
        oi_chg = futures_data["oi_change_pct"]
        indicadores["Open Interest Δ"] = f"{oi_chg:+.3f}% (30m)"
        cambio_precio = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        if oi_chg > 0.5 and cambio_precio > 0:
            señales["Open Interest Δ"] = ("alcista", f"OI sube + precio sube → tendencia real")
            puntuaciones["Open Interest Δ"] = 0.8
        elif oi_chg > 0.5 and cambio_precio < 0:
            señales["Open Interest Δ"] = ("bajista", f"OI sube + precio baja → más shorts")
            puntuaciones["Open Interest Δ"] = -0.8
        elif oi_chg < -0.5 and cambio_precio > 0:
            señales["Open Interest Δ"] = ("alcista_leve", f"OI baja + precio sube → shorts cerrando")
            puntuaciones["Open Interest Δ"] = 0.5
        else:
            señales["Open Interest Δ"] = ("neutro", f"OI sin tendencia clara ({oi_chg:+.3f}%)")
            puntuaciones["Open Interest Δ"] = 0.0
    else:
        indicadores["Open Interest Δ"] = "N/A (spot)"
        señales["Open Interest Δ"] = ("neutro", "Solo disponible en futuros")
        puntuaciones["Open Interest Δ"] = 0.0

    # ── 19. Long/Short Ratio ──
    if "long_ratio" in futures_data:
        lr   = futures_data["long_ratio"] * 100
        sr   = futures_data["short_ratio"] * 100
        indicadores["Long/Short Ratio"] = f"L={lr:.1f}% / S={sr:.1f}%"
        if lr > 60:
            señales["Long/Short Ratio"] = ("bajista_leve", f"Exceso longs ({lr:.0f}%) → contrarian")
            puntuaciones["Long/Short Ratio"] = -0.4
        elif sr > 60:
            señales["Long/Short Ratio"] = ("alcista_leve", f"Exceso shorts ({sr:.0f}%) → contrarian")
            puntuaciones["Long/Short Ratio"] = 0.4
        else:
            señales["Long/Short Ratio"] = ("neutro", f"Ratio equilibrado ({lr:.0f}/{sr:.0f})")
            puntuaciones["Long/Short Ratio"] = 0.0
    else:
        indicadores["Long/Short Ratio"] = "N/A (spot)"
        señales["Long/Short Ratio"] = ("neutro", "Solo disponible en futuros")
        puntuaciones["Long/Short Ratio"] = 0.0

    # ── 20. Contexto 5m (macro micro — desde df5) ──
    if df5 is not None and len(df5) >= 5:
        close5  = df5["close"]
        ema7_5m = close5.ewm(span=7).mean().iloc[-1]
        trend5  = (close5.iloc[-1] / close5.iloc[-5] - 1) * 100
        indicadores["Tendencia 5m TF"] = f"EMA7={ema7_5m:.4f} Δ5v={trend5:+.3f}%"
        if close5.iloc[-1] > ema7_5m and trend5 > 0.1:
            señales["Tendencia 5m TF"] = ("alcista", f"TF 5m alcista ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = 0.8
        elif close5.iloc[-1] < ema7_5m and trend5 < -0.1:
            señales["Tendencia 5m TF"] = ("bajista", f"TF 5m bajista ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = -0.8
        else:
            señales["Tendencia 5m TF"] = ("neutro", f"TF 5m lateral ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = 0.0
    else:
        indicadores["Tendencia 5m TF"] = "N/A"
        señales["Tendencia 5m TF"] = ("neutro", "Sin datos")
        puntuaciones["Tendencia 5m TF"] = 0.0

    return indicadores, señales, puntuaciones, atr_pct


# ─────────────────────────────────────────────
# PREDICCIÓN 5 MINUTOS
# ─────────────────────────────────────────────

PESOS_CRYPTO = {
    "RSI (9)":              1.5,
    "MACD (5,13,3)":        1.5,
    "EMA 7/25":             1.2,
    "Bollinger %B":         1.0,
    "Stochastic (5,3)":     1.0,
    "Williams %R":          0.8,
    "ATR (9)":              0.0,   # solo informativo
    "Rate of Change":       1.8,   # clave para 5m
    "OBV":                  1.0,
    "VWAP Desviación":      1.2,
    "Volumen Relativo":     1.3,
    "Patrón Vela 1m":       0.7,
    "Order Book Imbalance": 2.0,   # máximo peso — microestructura
    "Bid/Ask Spread":       0.5,
    "Buy/Sell Ratio":       2.0,   # máximo peso — flujo real
    "Actividad Trades":     0.8,
    "Funding Rate":         1.0,
    "Open Interest Δ":      1.0,
    "Long/Short Ratio":     0.6,
    "Tendencia 5m TF":      1.5,
}

BLOQUES_CRYPTO = {
    "⚡ Momentum Técnico":     ["RSI (9)", "MACD (5,13,3)", "EMA 7/25",
                                "Bollinger %B", "Stochastic (5,3)", "Williams %R"],
    "📊 Volatilidad y Precio": ["ATR (9)", "Rate of Change", "VWAP Desviación",
                                "Patrón Vela 1m"],
    "📦 Volumen y Flujo":      ["OBV", "Volumen Relativo", "Buy/Sell Ratio",
                                "Actividad Trades"],
    "🏦 Microestructura":      ["Order Book Imbalance", "Bid/Ask Spread"],
    "🔮 Futuros / Derivados":  ["Funding Rate", "Open Interest Δ", "Long/Short Ratio"],
    "🌐 Contexto Macro":       ["Tendencia 5m TF"],
}


def calcular_prediccion(puntuaciones, precio_actual, indicadores):
    # Extraer ATR%
    atr_str = indicadores.get("ATR (9)", "±0 (±0%)")
    try:
        atr_pct = float(atr_str.split("±")[2].replace("%)", "").replace("%", ""))
    except Exception:
        atr_pct = 0.1

    score_pond = sum(puntuaciones.get(k, 0) * PESOS_CRYPTO.get(k, 1)
                     for k in puntuaciones)
    peso_total = sum(v for v in PESOS_CRYPTO.values() if v > 0)
    score_norm = max(-1.0, min(1.0, score_pond / peso_total)) if peso_total else 0

    # Probabilidad: 50% ± 40% máximo
    prob_subida = max(5.0, min(95.0, 50.0 + score_norm * 40.0))

    # Movimiento estimado basado en ATR × factor de confianza
    mov_est = atr_pct * (0.6 + abs(score_norm) * 0.6)
    precio_obj = precio_actual * (1 + (mov_est if score_norm > 0 else -mov_est) / 100)

    if score_norm > 0.4:
        señal_texto = "ALCISTA FUERTE"
        señal_color = "alcista"
    elif score_norm > 0.15:
        señal_texto = "TENDENCIA ALCISTA"
        señal_color = "alcista_leve"
    elif score_norm < -0.4:
        señal_texto = "BAJISTA FUERTE"
        señal_color = "bajista"
    elif score_norm < -0.15:
        señal_texto = "TENDENCIA BAJISTA"
        señal_color = "bajista_leve"
    else:
        señal_texto = "LATERAL / INDECISO"
        señal_color = "neutro"

    alcistas = sum(1 for v in puntuaciones.values() if v > 0)
    bajistas = sum(1 for v in puntuaciones.values() if v < 0)
    neutros  = sum(1 for v in puntuaciones.values() if v == 0)

    return {
        "score":           score_norm,
        "prob_subida":     prob_subida,
        "prob_bajada":     100 - prob_subida,
        "direccion":       "↑ SUBE" if score_norm > 0 else "↓ BAJA",
        "mov_estimado":    mov_est,
        "precio_objetivo": precio_obj,
        "señal_texto":     señal_texto,
        "señal_color":     señal_color,
        "atr_pct":         atr_pct,
        "alcistas":        alcistas,
        "bajistas":        bajistas,
        "neutros":         neutros,
    }
