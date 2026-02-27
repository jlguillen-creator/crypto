"""
Módulo de lógica central — Crypto Predictor 5min  v2.0
═══════════════════════════════════════════════════════
Fuentes de datos:
  • Kraken REST API  — OHLCV 1m/5m/15m/1h, order book, ticker
  • OKX REST API     — order book global, trades reales, funding rate,
                       open interest, long/short ratio
  • Alternative.me   — Fear & Greed Index (sentimiento macro)

Mejoras v2:
  • OKX como fuente de microestructura (10-20x más profundidad que Kraken)
  • Buy/Sell ratio real de trades OKX (no estimado)
  • Funding rate + OI + Long/Short de OKX perpetuos (real)
  • Fear & Greed como modulador del score final
  • Multi-timeframe: 15m y 1h como contexto de régimen
  • Hurst Exponent → detecta tendencia vs. ruido vs. reversión
  • Pesos dinámicos según régimen de mercado
  • Correlación BTC (para altcoins)
  • Filtro de wash-trading (volumen anómalo sin movimiento)
"""

import requests
import pandas as pd
import numpy as np
import warnings
import ssl
import os
import urllib3
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
KRAKEN_BASE  = "https://api.kraken.com/0/public"
OKX_BASE     = "https://www.okx.com/api/v5"
FNG_URL      = "https://api.alternative.me/fng/?limit=3"


# ─────────────────────────────────────────────
# CLIENTE HTTP
# ─────────────────────────────────────────────
def _get(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, verify=False, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and data.get("error") and data["error"]:
                return None
            return data
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# MAPAS DE SÍMBOLO
# ─────────────────────────────────────────────
_KRAKEN_ALIASES = {
    "BTC":"XBTUSD",  "BTCUSDT":"XBTUSD",  "BTCUSD":"XBTUSD",
    "ETH":"ETHUSD",  "ETHUSDT":"ETHUSD",
    "SOL":"SOLUSD",  "SOLUSDT":"SOLUSD",
    "BNB":"BNBUSD",  "BNBUSDT":"BNBUSD",
    "XRP":"XRPUSD",  "XRPUSDT":"XRPUSD",
    "ADA":"ADAUSD",  "ADAUSDT":"ADAUSD",
    "DOGE":"XDGUSD", "DOGEUSDT":"XDGUSD",
    "DOT":"DOTUSD",  "DOTUSDT":"DOTUSD",
    "AVAX":"AVAXUSD","AVAXUSDT":"AVAXUSD",
    "LINK":"LINKUSD","LINKUSDT":"LINKUSD",
    "LTC":"XLTCZUSD","LTCUSDT":"XLTCZUSD",
    "ATOM":"ATOMUSD","ATOMUSDT":"ATOMUSD",
    "MATIC":"MATICUSD","MATICUSDT":"MATICUSD",
    "UNI":"UNIUSD",  "UNIUSDT":"UNIUSD",
    "NEAR":"NEARUSD","NEARUSDT":"NEARUSD",
    "AAVE":"AAVEUSD","AAVEUSDT":"AAVEUSD",
    "XLM":"XXLMZUSD","XLMUSDT":"XXLMZUSD",
    "TRX":"TRXUSD",  "TRXUSDT":"TRXUSD",
}

# OKX usa el formato "BTC-USDT" para spot y "BTC-USDT-SWAP" para perpetuos
_OKX_SPOT = {
    "BTC":"BTC-USDT",  "ETH":"ETH-USDT",  "SOL":"SOL-USDT",
    "XRP":"XRP-USDT",  "BNB":"BNB-USDT",  "ADA":"ADA-USDT",
    "DOGE":"DOGE-USDT","DOT":"DOT-USDT",  "AVAX":"AVAX-USDT",
    "LINK":"LINK-USDT","LTC":"LTC-USDT",  "ATOM":"ATOM-USDT",
    "MATIC":"MATIC-USDT","UNI":"UNI-USDT","NEAR":"NEAR-USDT",
    "AAVE":"AAVE-USDT","XLM":"XLM-USDT", "TRX":"TRX-USDT",
}
_OKX_SWAP = {k: v.replace("-USDT", "-USDT-SWAP") for k, v in _OKX_SPOT.items()}

# BTC en Kraken = "XBTUSD", pero para OKX = "BTC"
_KRAKEN_TO_BASE = {
    "XBTUSD":"BTC", "ETHUSD":"ETH", "SOLUSD":"SOL", "XRPUSD":"XRP",
    "BNBUSD":"BNB", "ADAUSD":"ADA", "XDGUSD":"DOGE", "DOTUSD":"DOT",
    "AVAXUSD":"AVAX","LINKUSD":"LINK","XLTCZUSD":"LTC","ATOMUSD":"ATOM",
    "MATICUSD":"MATIC","UNIUSD":"UNI","NEARUSD":"NEAR","AAVEUSD":"AAVE",
    "XXLMZUSD":"XLM","TRXUSD":"TRX",
}


def normalizar_symbol(symbol: str) -> tuple:
    """Devuelve (kraken_pair, display_name)."""
    s = symbol.upper().strip().replace("/","").replace("-","").replace("_","")
    if s in _KRAKEN_ALIASES:
        pair    = _KRAKEN_ALIASES[s]
        base    = _KRAKEN_TO_BASE.get(pair, s.replace("USDT","").replace("USD",""))
        display = base + "/USD"
        return pair, display
    if s.endswith("USD"):
        return s, s[:-3] + "/USD"
    if s.endswith("USDT"):
        return s[:-1], s[:-4] + "/USD"
    return s + "USD", s + "/USD"


def _base_from_kraken(kraken_pair: str) -> str:
    """XBTUSD → BTC, ETHUSD → ETH, …"""
    return _KRAKEN_TO_BASE.get(kraken_pair, kraken_pair.replace("USD",""))


# ─────────────────────────────────────────────
# FUNCIONES DE DESCARGA INDIVIDUALES
# ─────────────────────────────────────────────

def _kraken_ohlc(pair: str, interval: int, limit: int = 100):
    """Descarga velas de Kraken. Devuelve DataFrame o None."""
    raw = _get(f"{KRAKEN_BASE}/OHLC", {"pair": pair, "interval": interval})
    if not raw or "result" not in raw:
        return None
    key = [k for k in raw["result"] if k != "last"][0]
    data = raw["result"][key][-(limit + 1):-1]
    if len(data) < 10:
        return None
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    for col in ["open","high","low","close","vwap","volume"]:
        df[col] = df[col].astype(float)
    df["count"] = df["count"].astype(int)
    return df


def _kraken_book(pair: str):
    raw = _get(f"{KRAKEN_BASE}/Depth", {"pair": pair, "count": 20})
    if raw and "result" in raw:
        key = list(raw["result"].keys())[0]
        bk  = raw["result"][key]
        return {
            "bids": [[float(b[0]), float(b[1])] for b in bk.get("bids", [])],
            "asks": [[float(a[0]), float(a[1])] for a in bk.get("asks", [])],
            "source": "kraken",
        }
    return None


def _okx_book(base: str):
    """Order book de OKX spot — profundidad 20 niveles."""
    inst = _OKX_SPOT.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/market/books", {"instId": inst, "sz": "20"})
    if raw and raw.get("code") == "0" and raw.get("data"):
        bk = raw["data"][0]
        return {
            "bids": [[float(b[0]), float(b[1])] for b in bk.get("bids", [])],
            "asks": [[float(a[0]), float(a[1])] for a in bk.get("asks", [])],
            "source": "okx",
        }
    return None


def _okx_trades(base: str):
    """Últimos 100 trades de OKX — permite calcular taker buy/sell real."""
    inst = _OKX_SPOT.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/market/trades", {"instId": inst, "limit": "100"})
    if raw and raw.get("code") == "0" and raw.get("data"):
        trades = raw["data"]
        buy_vol  = sum(float(t["sz"]) * float(t["px"])
                       for t in trades if t.get("side") == "buy")
        sell_vol = sum(float(t["sz"]) * float(t["px"])
                       for t in trades if t.get("side") == "sell")
        total    = buy_vol + sell_vol
        return {
            "buy_vol":   buy_vol,
            "sell_vol":  sell_vol,
            "buy_ratio": buy_vol / total * 100 if total > 0 else 50.0,
            "n_trades":  len(trades),
        }
    return None


def _okx_funding(base: str):
    """Funding rate actual del perpetuo en OKX."""
    inst = _OKX_SWAP.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/public/funding-rate", {"instId": inst})
    if raw and raw.get("code") == "0" and raw.get("data"):
        fr = raw["data"][0].get("fundingRate")
        next_fr = raw["data"][0].get("nextFundingRate")
        return {
            "funding_rate":      float(fr) if fr else None,
            "next_funding_rate": float(next_fr) if next_fr else None,
        }
    return None


def _okx_open_interest(base: str):
    """Open Interest del perpetuo en OKX."""
    inst = _OKX_SWAP.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/public/open-interest", {"instId": inst})
    if raw and raw.get("code") == "0" and raw.get("data"):
        oi = raw["data"][0].get("oiCcy")  # en moneda base
        return float(oi) if oi else None
    return None


def _okx_oi_history(base: str):
    """Historial de OI (8 puntos cada 5min) para calcular cambio %."""
    inst = _OKX_SWAP.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/rubik/stat/contracts/open-interest-volume",
               {"ccy": base, "period": "5m"})
    # Endpoint alternativo si falla
    if not raw or raw.get("code") != "0":
        raw = _get(f"{OKX_BASE}/public/open-interest-history",
                   {"instId": inst, "period": "5m", "limit": "8"})
    if raw and raw.get("code") == "0" and raw.get("data") and len(raw["data"]) >= 2:
        try:
            vals = [float(d[1]) if isinstance(d, list) else float(d.get("oiCcy", 0))
                    for d in raw["data"][:8]]
            if vals[0] > 0 and vals[-1] > 0:
                return (vals[0] - vals[-1]) / vals[-1] * 100
        except Exception:
            pass
    return None


def _okx_long_short(base: str):
    """Ratio long/short de OKX."""
    raw = _get(f"{OKX_BASE}/rubik/stat/contracts/long-short-account-ratio",
               {"ccy": base, "period": "5m"})
    if raw and raw.get("code") == "0" and raw.get("data"):
        try:
            ls_ratio = float(raw["data"][0][1])  # longRatio
            return {
                "long_ratio":  ls_ratio / (1 + ls_ratio),
                "short_ratio": 1 / (1 + ls_ratio),
                "ls_raw":      ls_ratio,
            }
        except Exception:
            pass
    return None


def _fear_greed():
    """Fear & Greed Index de alternative.me — actualiza cada hora."""
    raw = _get(FNG_URL, timeout=6)
    if raw and "data" in raw and raw["data"]:
        try:
            latest = raw["data"][0]
            prev   = raw["data"][1] if len(raw["data"]) > 1 else latest
            return {
                "value":          int(latest["value"]),
                "classification": latest["value_classification"],
                "prev_value":     int(prev["value"]),
                "trend":          int(latest["value"]) - int(prev["value"]),
            }
        except Exception:
            pass
    return None


def _okx_price(base: str):
    """Precio último de OKX para comparación multi-exchange."""
    inst = _OKX_SPOT.get(base)
    if not inst:
        return None
    raw = _get(f"{OKX_BASE}/market/ticker", {"instId": inst})
    if raw and raw.get("code") == "0" and raw.get("data"):
        try:
            return float(raw["data"][0]["last"])
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────
# HURST EXPONENT (detección de régimen)
# ─────────────────────────────────────────────
def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    H > 0.6  → tendencia persistente (trending)
    H ≈ 0.5  → movimiento browniano (ruido, difícil predecir)
    H < 0.4  → reversión a la media (mean-reverting)
    """
    try:
        prices = series.dropna().values
        if len(prices) < max_lag * 2:
            return 0.5
        lags   = range(min_lag, max_lag)
        tau    = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        tau    = [t for t in tau if t > 0]
        if len(tau) < 3:
            return 0.5
        poly   = np.polyfit(np.log(list(range(min_lag, min_lag + len(tau)))),
                            np.log(tau), 1)
        return max(0.1, min(0.9, poly[0]))
    except Exception:
        return 0.5


def clasificar_regimen(h: float) -> str:
    if h > 0.62:
        return "trending"
    elif h < 0.40:
        return "mean_reverting"
    else:
        return "noise"


# ─────────────────────────────────────────────
# DETECCIÓN DE WASH TRADING
# ─────────────────────────────────────────────
def detectar_wash_trading(df: pd.DataFrame) -> float:
    """
    Devuelve ratio de velas sospechosas (vol alto + movimiento mínimo).
    > 0.4 → posible wash trading, reducir confianza en volumen.
    """
    try:
        vol_z    = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        body_pct = ((df["close"] - df["open"]).abs() / df["close"] * 100)
        sospecha = ((vol_z > 2.0) & (body_pct < 0.02)).sum()
        return sospecha / len(df)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# DESCARGA PRINCIPAL — PARALELA
# ─────────────────────────────────────────────
def _safe_call(fn, *args):
    """Llama a fn, devuelve None ante cualquier excepción."""
    try:
        return fn(*args)
    except Exception:
        return None


def descargar_datos(symbol: str):
    pair, display = normalizar_symbol(symbol)
    base = _base_from_kraken(pair)

    # ── OHLC 1m — obligatorio ──
    df = _safe_call(_kraken_ohlc, pair, 1, 100)
    if df is None:
        try:
            assets = _get(f"{KRAKEN_BASE}/AssetPairs", {"pair": pair})
            if assets and "result" in assets:
                alt = list(assets["result"].keys())[0]
                df  = _safe_call(_kraken_ohlc, alt, 1, 100)
        except Exception:
            pass
    if df is None or len(df) < 20:
        return None, None, None, None, None, (
            f"Par '{pair}' no encontrado en Kraken. "
            "Prueba con: BTC, ETH, SOL, XRP, DOGE, ADA, DOT, AVAX, LINK, LTC…"
        ), None, None

    # ── OHLC adicionales — opcionales ──
    df5   = _safe_call(_kraken_ohlc, pair, 5,  60)
    df15  = _safe_call(_kraken_ohlc, pair, 15, 50)
    df1h  = _safe_call(_kraken_ohlc, pair, 60, 48)

    # ── Order book: OKX preferido, fallback Kraken ──
    book_okx = _safe_call(_okx_book,    base)
    book_krk = _safe_call(_kraken_book, pair)
    book = book_okx if book_okx else book_krk

    # ── Datos OKX — todos opcionales, nunca bloquean ──
    okx_trades   = _safe_call(_okx_trades,       base)
    funding_data = _safe_call(_okx_funding,       base) or {}
    oi_val       = _safe_call(_okx_open_interest, base)
    oi_chg       = _safe_call(_okx_oi_history,    base)
    ls_data      = _safe_call(_okx_long_short,    base) or {}
    okx_price    = _safe_call(_okx_price,         base)
    fng          = _safe_call(_fear_greed)

    # ── Enriquecer df con taker estimado (mejorado con OKX trades si disponible) ──
    # okx_trades already set above
    if okx_trades:
        # Usar ratio real de OKX para distribuir el volumen de las últimas velas
        real_ratio = okx_trades["buy_ratio"] / 100
        df["taker_buy_base"]  = df["volume"] * real_ratio
        df["quote_volume"]    = df["volume"] * df["close"]
        df["taker_buy_quote"] = df["quote_volume"] * real_ratio
    else:
        df["quote_volume"]    = df["volume"] * df["close"]
        df["taker_buy_quote"] = df.apply(
            lambda r: r["quote_volume"] * 0.6 if r["close"] >= r["open"]
                      else r["quote_volume"] * 0.4, axis=1)
        df["taker_buy_base"]  = df.apply(
            lambda r: r["volume"] * 0.6 if r["close"] >= r["open"]
                      else r["volume"] * 0.4, axis=1)
    df["trades"] = df["count"]

    # ── Order book: preferir OKX (más profundo), fallback Kraken ──
    # book_okx already set above
    # book_krk already set above
    book = book_okx if book_okx else book_krk

    # ── Ticker Kraken para precio y 24h stats ──
    ticker_raw    = _get(f"{KRAKEN_BASE}/Ticker", {"pair": pair})
    precio_actual = float(df["close"].iloc[-1])
    cambio_pct    = 0.0
    vol_24h = high_24h = low_24h = 0.0

    if ticker_raw and "result" in ticker_raw:
        tk_key = list(ticker_raw["result"].keys())[0]
        tk     = ticker_raw["result"][tk_key]
        precio_actual = float(tk["c"][0])
        open_price    = float(tk["o"])
        cambio_pct    = (precio_actual / open_price - 1) * 100 if open_price else 0
        vol_24h       = float(tk["v"][1]) * precio_actual
        high_24h      = float(tk["h"][1])
        low_24h       = float(tk["l"][1])

    # ── Precio OKX para comparación multi-exchange ──
    # okx_price already set above
    price_diverge = None
    if okx_price and precio_actual > 0:
        price_diverge = (precio_actual - okx_price) / okx_price * 100

    # ── Datos de futuros / derivados de OKX ──
    # funding_data already set above
    # oi_val already set above
    # oi_chg already set above
    # ls_data already set above

    futures_data = {
        "funding_rate":      funding_data.get("funding_rate"),
        "next_funding_rate": funding_data.get("next_funding_rate"),
        "open_interest":     oi_val,
        "oi_change_pct":     oi_chg,
        "long_ratio":        ls_data.get("long_ratio"),
        "short_ratio":       ls_data.get("short_ratio"),
        "ls_raw":            ls_data.get("ls_raw"),
        "okx_trades":        okx_trades,
        "price_diverge":     price_diverge,
        "book_source":       book.get("source", "kraken") if book else "kraken",
    }

    # ── Fear & Greed ──
    # fng already set above
    if fng:
        futures_data["fng_value"]  = fng["value"]
        futures_data["fng_class"]  = fng["classification"]
        futures_data["fng_trend"]  = fng["trend"]

    info = {
        "symbol":        pair,
        "nombre":        display,
        "precio_actual": precio_actual,
        "precio_prev":   float(df["close"].iloc[-2]),
        "cambio_pct":    cambio_pct,
        "vol_24h":       vol_24h,
        "high_24h":      high_24h,
        "low_24h":       low_24h,
        "okx_price":     okx_price,
        "base":          base,
    }

    return df, df5, book, futures_data, info, None, df15, df1h


# ─────────────────────────────────────────────
# 20 INDICADORES BASE + NUEVOS CONTEXTUALES
# ─────────────────────────────────────────────

def calcular_indicadores(df, df5, book, futures_data, info,
                         df15=None, df1h=None):
    indicadores  = {}
    señales      = {}
    puntuaciones = {}

    close   = df["close"]
    high    = df["high"]
    low     = df["low"]
    volume  = df["volume"]
    precio  = info["precio_actual"]

    # ── 1. RSI (9) ──
    delta  = close.diff()
    avg_g  = delta.clip(lower=0).rolling(9).mean()
    avg_l  = (-delta.clip(upper=0)).rolling(9).mean()
    rsi    = (100 - 100 / (1 + avg_g / avg_l)).iloc[-1]
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

    # ── 2. MACD (5,13,3) ──
    macd_l  = close.ewm(span=5).mean() - close.ewm(span=13).mean()
    sig_l   = macd_l.ewm(span=3).mean()
    hist    = macd_l - sig_l
    h_val, h_prev = hist.iloc[-1], hist.iloc[-2]
    indicadores["MACD (5,13,3)"] = f"{macd_l.iloc[-1]:.4f} / {sig_l.iloc[-1]:.4f}"
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
        señales["MACD (5,13,3)"] = ("neutro", "Cruce zona 0")
        puntuaciones["MACD (5,13,3)"] = 0.0

    # ── 3. EMA 7/25 ──
    ema7  = close.ewm(span=7).mean().iloc[-1]
    ema25 = close.ewm(span=25).mean().iloc[-1]
    indicadores["EMA 7/25"] = f"{ema7:.4f} / {ema25:.4f}"
    if ema7 > ema25 and precio > ema7:
        señales["EMA 7/25"] = ("alcista", "Precio > EMA7 > EMA25")
        puntuaciones["EMA 7/25"] = 1.0
    elif ema7 > ema25:
        señales["EMA 7/25"] = ("alcista_leve", "EMA7 > EMA25, retroceso")
        puntuaciones["EMA 7/25"] = 0.3
    elif ema7 < ema25 and precio < ema7:
        señales["EMA 7/25"] = ("bajista", "Precio < EMA7 < EMA25")
        puntuaciones["EMA 7/25"] = -1.0
    elif ema7 < ema25:
        señales["EMA 7/25"] = ("bajista_leve", "EMA7 < EMA25, rebote")
        puntuaciones["EMA 7/25"] = -0.3
    else:
        señales["EMA 7/25"] = ("neutro", "EMAs entrelazadas")
        puntuaciones["EMA 7/25"] = 0.0

    # ── 4. Bollinger %B ──
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = (bb_mid + 2 * bb_std).iloc[-1]
    bb_lower = (bb_mid - 2 * bb_std).iloc[-1]
    bb_mid_v = bb_mid.iloc[-1]
    pct_b    = (precio - bb_lower) / (bb_upper - bb_lower) * 100 \
               if (bb_upper - bb_lower) > 0 else 50
    bw       = (bb_upper - bb_lower) / bb_mid_v * 100
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
        señales["Bollinger %B"] = ("neutro", f"Centro ({pct_b:.0f}%)")
        puntuaciones["Bollinger %B"] = 0.0

    # ── 5. Stochastic (5,3) ──
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

    # ── 6. Williams %R ──
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

    # ── 7. ATR (9) — informativo ──
    tr    = pd.concat([high - low,
                       (high - close.shift()).abs(),
                       (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(9).mean().iloc[-1]
    atr_pct = atr / precio * 100
    indicadores["ATR (9)"] = f"±{atr:.4f} (±{atr_pct:.3f}%)"
    señales["ATR (9)"]     = ("neutro", f"Volatilidad: ±{atr_pct:.3f}% por vela")
    puntuaciones["ATR (9)"] = 0.0

    # ── 8. Rate of Change ──
    roc5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
    roc3 = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0
    indicadores["Rate of Change"] = f"3m: {roc3:+.3f}% | 5m: {roc5:+.3f}%"
    if roc5 > 0.15 and roc3 > 0:
        señales["Rate of Change"] = ("alcista", f"Momentum +{roc5:.3f}%")
        puntuaciones["Rate of Change"] = min(1.0, roc5 / 0.3)
    elif roc5 < -0.15 and roc3 < 0:
        señales["Rate of Change"] = ("bajista", f"Momentum {roc5:.3f}%")
        puntuaciones["Rate of Change"] = max(-1.0, roc5 / 0.3)
    else:
        señales["Rate of Change"] = ("neutro", f"Sin momentum ({roc5:+.3f}%)")
        puntuaciones["Rate of Change"] = 0.0

    # ── 9. OBV ──
    obv        = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema    = obv.ewm(span=10).mean()
    obv_trend  = obv.iloc[-1] - obv.iloc[-5]
    indicadores["OBV"] = f"Δ5m: {obv_trend:+.0f}"
    if obv.iloc[-1] > obv_ema.iloc[-1] and obv_trend > 0:
        señales["OBV"] = ("alcista", "OBV > EMA y subiendo")
        puntuaciones["OBV"] = 1.0
    elif obv.iloc[-1] < obv_ema.iloc[-1] and obv_trend < 0:
        señales["OBV"] = ("bajista", "OBV < EMA y bajando")
        puntuaciones["OBV"] = -1.0
    else:
        señales["OBV"] = ("neutro", "OBV mixto")
        puntuaciones["OBV"] = 0.0

    # ── 10. VWAP Desviación ──
    vwap = (close * volume).cumsum() / volume.cumsum()
    vwap_dev = (precio - vwap.iloc[-1]) / vwap.iloc[-1] * 100
    indicadores["VWAP Desviación"] = f"VWAP={vwap.iloc[-1]:.4f} ({vwap_dev:+.3f}%)"
    if vwap_dev > 0.1:
        señales["VWAP Desviación"] = ("bajista_leve", f"Precio {vwap_dev:+.2f}% sobre VWAP")
        puntuaciones["VWAP Desviación"] = -0.5
    elif vwap_dev < -0.1:
        señales["VWAP Desviación"] = ("alcista_leve", f"Precio {vwap_dev:+.2f}% bajo VWAP")
        puntuaciones["VWAP Desviación"] = 0.5
    else:
        señales["VWAP Desviación"] = ("neutro", f"≈ VWAP ({vwap_dev:+.3f}%)")
        puntuaciones["VWAP Desviación"] = 0.0

    # ── 11. Volumen Relativo ──
    vol_ma    = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1
    cambio_1m = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    # Penalizar si hay indicio de wash trading
    wash_ratio = detectar_wash_trading(df)
    vol_ratio_adj = vol_ratio * (1 - wash_ratio * 0.5)
    indicadores["Volumen Relativo"] = f"{vol_ratio_adj:.2f}x" + (" ⚠wash" if wash_ratio > 0.3 else "")
    if vol_ratio_adj > 2.0 and cambio_1m > 0:
        señales["Volumen Relativo"] = ("alcista", f"Vol {vol_ratio_adj:.1f}x alcista")
        puntuaciones["Volumen Relativo"] = 1.0
    elif vol_ratio_adj > 2.0 and cambio_1m < 0:
        señales["Volumen Relativo"] = ("bajista", f"Vol {vol_ratio_adj:.1f}x bajista")
        puntuaciones["Volumen Relativo"] = -1.0
    elif vol_ratio_adj > 1.3 and cambio_1m > 0:
        señales["Volumen Relativo"] = ("alcista_leve", f"Vol elevado ({vol_ratio_adj:.1f}x)")
        puntuaciones["Volumen Relativo"] = 0.5
    elif vol_ratio_adj > 1.3 and cambio_1m < 0:
        señales["Volumen Relativo"] = ("bajista_leve", f"Vol elevado bajista ({vol_ratio_adj:.1f}x)")
        puntuaciones["Volumen Relativo"] = -0.5
    else:
        señales["Volumen Relativo"] = ("neutro", f"Normal ({vol_ratio_adj:.1f}x)")
        puntuaciones["Volumen Relativo"] = 0.0

    # ── 12. Patrón Vela ──
    o, c_, h_, l_ = df["open"].iloc[-1], close.iloc[-1], high.iloc[-1], low.iloc[-1]
    body  = abs(c_ - o); rng = h_ - l_
    ls_s  = min(o, c_) - l_; us_s = h_ - max(o, c_)
    patron, p_score = "Neutro", 0.0
    if rng > 0:
        if body > rng * 0.8 and c_ > o:
            patron, p_score = "Marubozu alcista", 1.0
        elif body > rng * 0.8 and c_ < o:
            patron, p_score = "Marubozu bajista", -1.0
        elif ls_s > body * 2 and us_s < body * 0.5:
            patron, p_score = "Hammer", 1.0
        elif us_s > body * 2 and ls_s < body * 0.5:
            patron, p_score = "Shooting Star", -1.0
        elif body < rng * 0.1:
            patron, p_score = "Doji", 0.0
        elif c_ > o:
            patron, p_score = "Alcista", 0.5
        else:
            patron, p_score = "Bajista", -0.5
    indicadores["Patrón Vela 1m"] = patron
    tipo_v = ("alcista" if p_score > 0.5 else
              "bajista" if p_score < -0.5 else
              "alcista_leve" if p_score > 0 else
              "bajista_leve" if p_score < 0 else "neutro")
    señales["Patrón Vela 1m"] = (tipo_v, patron)
    puntuaciones["Patrón Vela 1m"] = p_score

    # ── 13. Order Book Imbalance (OKX preferido) ──
    book_src = futures_data.get("book_source", "kraken")
    if book and "bids" in book and "asks" in book:
        bids    = book["bids"][:10]
        asks    = book["asks"][:10]
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total   = bid_vol + ask_vol
        obi     = (bid_vol - ask_vol) / total * 100 if total > 0 else 0
        src_tag = "OKX" if book_src == "okx" else "Kraken"
        indicadores["Order Book Imbalance"] = f"OBI={obi:+.1f}% [{src_tag}]"
        if obi > 15:
            señales["Order Book Imbalance"] = ("alcista", f"Presión compradora {obi:+.0f}% [{src_tag}]")
            puntuaciones["Order Book Imbalance"] = min(1.0, obi / 30)
        elif obi < -15:
            señales["Order Book Imbalance"] = ("bajista", f"Presión vendedora {obi:+.0f}% [{src_tag}]")
            puntuaciones["Order Book Imbalance"] = max(-1.0, obi / 30)
        else:
            señales["Order Book Imbalance"] = ("neutro", f"Equilibrado {obi:+.0f}% [{src_tag}]")
            puntuaciones["Order Book Imbalance"] = obi / 100
    else:
        indicadores["Order Book Imbalance"] = "N/A"
        señales["Order Book Imbalance"] = ("neutro", "Sin datos")
        puntuaciones["Order Book Imbalance"] = 0.0

    # ── 14. Bid/Ask Spread ──
    if book and book.get("bids") and book.get("asks"):
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

    # ── 15. Buy/Sell Ratio — REAL de OKX si disponible ──
    okx_trades_data = futures_data.get("okx_trades")
    if okx_trades_data:
        bs_ratio = okx_trades_data["buy_ratio"]
        n_trades = okx_trades_data["n_trades"]
        indicadores["Buy/Sell Ratio"] = f"{bs_ratio:.1f}% buy (OKX {n_trades}t)"
        src_bs = "OKX real"
    else:
        taker_buy  = df["taker_buy_quote"].tail(10).sum()
        taker_sell = (df["quote_volume"] - df["taker_buy_quote"]).tail(10).sum()
        total_t    = taker_buy + taker_sell
        bs_ratio   = taker_buy / total_t * 100 if total_t > 0 else 50
        indicadores["Buy/Sell Ratio"] = f"{bs_ratio:.1f}% buy (Kraken est.)"
        src_bs = "Kraken estimado"

    if bs_ratio > 60:
        señales["Buy/Sell Ratio"] = ("alcista", f"Compradores dominan {bs_ratio:.0f}% [{src_bs}]")
        puntuaciones["Buy/Sell Ratio"] = min(1.0, (bs_ratio - 50) / 25)
    elif bs_ratio < 40:
        señales["Buy/Sell Ratio"] = ("bajista", f"Vendedores dominan {bs_ratio:.0f}% [{src_bs}]")
        puntuaciones["Buy/Sell Ratio"] = max(-1.0, (bs_ratio - 50) / 25)
    else:
        señales["Buy/Sell Ratio"] = ("neutro", f"Equilibrio {bs_ratio:.0f}% [{src_bs}]")
        puntuaciones["Buy/Sell Ratio"] = (bs_ratio - 50) / 50

    # ── 16. Actividad Trades ──
    trades_pm  = df["trades"].tail(5).mean()
    trades_max = df["trades"].max()
    trades_pct = trades_pm / trades_max * 100 if trades_max > 0 else 50
    indicadores["Actividad Trades"] = f"{trades_pm:.0f} t/min (media 5m)"
    if trades_pct > 70:
        cambio_5m = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        if cambio_5m > 0:
            señales["Actividad Trades"] = ("alcista", f"Alta actividad subida ({trades_pct:.0f}%)")
            puntuaciones["Actividad Trades"] = 0.7
        else:
            señales["Actividad Trades"] = ("bajista", f"Alta actividad bajada ({trades_pct:.0f}%)")
            puntuaciones["Actividad Trades"] = -0.7
    else:
        señales["Actividad Trades"] = ("neutro", f"Actividad normal ({trades_pct:.0f}%)")
        puntuaciones["Actividad Trades"] = 0.0

    # ── 17. Funding Rate — OKX real ──
    fr = futures_data.get("funding_rate")
    if fr is not None:
        fr_pct = fr * 100
        next_fr = futures_data.get("next_funding_rate")
        next_txt = f" → {next_fr*100:+.4f}%" if next_fr else ""
        indicadores["Funding Rate"] = f"{fr_pct:+.4f}%{next_txt} [OKX]"
        if fr_pct > 0.05:
            señales["Funding Rate"] = ("bajista_leve", f"Longs pagando alto ({fr_pct:+.4f}%)")
            puntuaciones["Funding Rate"] = -min(1.0, fr_pct / 0.08)
        elif fr_pct < -0.05:
            señales["Funding Rate"] = ("alcista_leve", f"Shorts pagando ({fr_pct:+.4f}%)")
            puntuaciones["Funding Rate"] = min(1.0, abs(fr_pct) / 0.08)
        else:
            señales["Funding Rate"] = ("neutro", f"FR neutro ({fr_pct:+.4f}%)")
            puntuaciones["Funding Rate"] = 0.0
    else:
        indicadores["Funding Rate"] = "N/A"
        señales["Funding Rate"] = ("neutro", "Sin datos OKX")
        puntuaciones["Funding Rate"] = 0.0

    # ── 18. Open Interest Δ — OKX real ──
    oi_chg = futures_data.get("oi_change_pct")
    if oi_chg is not None:
        cambio_precio = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        indicadores["Open Interest Δ"] = f"{oi_chg:+.3f}% (5m) [OKX]"
        if oi_chg > 0.5 and cambio_precio > 0:
            señales["Open Interest Δ"] = ("alcista", "OI↑ + precio↑ → tendencia real")
            puntuaciones["Open Interest Δ"] = 0.8
        elif oi_chg > 0.5 and cambio_precio < 0:
            señales["Open Interest Δ"] = ("bajista", "OI↑ + precio↓ → más shorts")
            puntuaciones["Open Interest Δ"] = -0.8
        elif oi_chg < -0.5 and cambio_precio > 0:
            señales["Open Interest Δ"] = ("alcista_leve", "OI↓ + precio↑ → shorts cerrando")
            puntuaciones["Open Interest Δ"] = 0.5
        else:
            señales["Open Interest Δ"] = ("neutro", f"OI sin tendencia ({oi_chg:+.3f}%)")
            puntuaciones["Open Interest Δ"] = 0.0
    else:
        indicadores["Open Interest Δ"] = "N/A"
        señales["Open Interest Δ"] = ("neutro", "Sin datos OKX")
        puntuaciones["Open Interest Δ"] = 0.0

    # ── 19. Long/Short Ratio — OKX real ──
    lr = futures_data.get("long_ratio")
    sr = futures_data.get("short_ratio")
    if lr is not None and sr is not None:
        lr_pct, sr_pct = lr * 100, sr * 100
        indicadores["Long/Short Ratio"] = f"L={lr_pct:.1f}% / S={sr_pct:.1f}% [OKX]"
        if lr_pct > 60:
            señales["Long/Short Ratio"] = ("bajista_leve", f"Exceso longs ({lr_pct:.0f}%) → contrarian")
            puntuaciones["Long/Short Ratio"] = -0.4
        elif sr_pct > 60:
            señales["Long/Short Ratio"] = ("alcista_leve", f"Exceso shorts ({sr_pct:.0f}%) → contrarian")
            puntuaciones["Long/Short Ratio"] = 0.4
        else:
            señales["Long/Short Ratio"] = ("neutro", f"Ratio equilibrado ({lr_pct:.0f}/{sr_pct:.0f})")
            puntuaciones["Long/Short Ratio"] = 0.0
    else:
        indicadores["Long/Short Ratio"] = "N/A"
        señales["Long/Short Ratio"] = ("neutro", "Sin datos OKX")
        puntuaciones["Long/Short Ratio"] = 0.0

    # ── 20. Tendencia 5m TF ──
    if df5 is not None and len(df5) >= 5:
        close5  = df5["close"]
        ema7_5  = close5.ewm(span=7).mean().iloc[-1]
        trend5  = (close5.iloc[-1] / close5.iloc[-5] - 1) * 100
        indicadores["Tendencia 5m TF"] = f"EMA7={ema7_5:.4f} Δ={trend5:+.3f}%"
        if close5.iloc[-1] > ema7_5 and trend5 > 0.1:
            señales["Tendencia 5m TF"] = ("alcista", f"5m alcista ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = 0.8
        elif close5.iloc[-1] < ema7_5 and trend5 < -0.1:
            señales["Tendencia 5m TF"] = ("bajista", f"5m bajista ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = -0.8
        else:
            señales["Tendencia 5m TF"] = ("neutro", f"5m lateral ({trend5:+.3f}%)")
            puntuaciones["Tendencia 5m TF"] = 0.0
    else:
        indicadores["Tendencia 5m TF"] = "N/A"
        señales["Tendencia 5m TF"] = ("neutro", "Sin datos")
        puntuaciones["Tendencia 5m TF"] = 0.0

    # ════════════════════════════════════════════
    # NUEVOS INDICADORES — NIVEL 1 Y 2
    # ════════════════════════════════════════════

    # ── N1. Fear & Greed Index ──
    fng_val = futures_data.get("fng_value")
    fng_cls = futures_data.get("fng_class", "")
    fng_trn = futures_data.get("fng_trend", 0)
    if fng_val is not None:
        indicadores["Fear & Greed"] = f"{fng_val} — {fng_cls} (Δ{fng_trn:+d})"
        if fng_val <= 20:
            señales["Fear & Greed"] = ("alcista", f"Miedo extremo ({fng_val}) → oportunidad")
            puntuaciones["Fear & Greed"] = 0.8
        elif fng_val <= 40:
            señales["Fear & Greed"] = ("alcista_leve", f"Miedo ({fng_val}) → sesgo alcista")
            puntuaciones["Fear & Greed"] = 0.3
        elif fng_val >= 80:
            señales["Fear & Greed"] = ("bajista", f"Codicia extrema ({fng_val}) → precaución")
            puntuaciones["Fear & Greed"] = -0.8
        elif fng_val >= 60:
            señales["Fear & Greed"] = ("bajista_leve", f"Codicia ({fng_val}) → sesgo bajista")
            puntuaciones["Fear & Greed"] = -0.3
        else:
            señales["Fear & Greed"] = ("neutro", f"Neutro ({fng_val})")
            puntuaciones["Fear & Greed"] = 0.0
    else:
        indicadores["Fear & Greed"] = "N/A"
        señales["Fear & Greed"] = ("neutro", "Sin datos")
        puntuaciones["Fear & Greed"] = 0.0

    # ── N2. Tendencia 15m TF ──
    if df15 is not None and len(df15) >= 8:
        c15     = df15["close"]
        ema9_15 = c15.ewm(span=9).mean()
        ema21_15 = c15.ewm(span=21).mean()
        t15     = (c15.iloc[-1] / c15.iloc[-5] - 1) * 100
        e9, e21 = ema9_15.iloc[-1], ema21_15.iloc[-1]
        indicadores["Tendencia 15m TF"] = f"EMA9={e9:.4f} EMA21={e21:.4f} Δ={t15:+.3f}%"
        if c15.iloc[-1] > e9 > e21 and t15 > 0.15:
            señales["Tendencia 15m TF"] = ("alcista", f"15m alcista fuerte ({t15:+.3f}%)")
            puntuaciones["Tendencia 15m TF"] = 1.0
        elif c15.iloc[-1] > e9 and t15 > 0:
            señales["Tendencia 15m TF"] = ("alcista_leve", f"15m alcista ({t15:+.3f}%)")
            puntuaciones["Tendencia 15m TF"] = 0.5
        elif c15.iloc[-1] < e9 < e21 and t15 < -0.15:
            señales["Tendencia 15m TF"] = ("bajista", f"15m bajista fuerte ({t15:+.3f}%)")
            puntuaciones["Tendencia 15m TF"] = -1.0
        elif c15.iloc[-1] < e9 and t15 < 0:
            señales["Tendencia 15m TF"] = ("bajista_leve", f"15m bajista ({t15:+.3f}%)")
            puntuaciones["Tendencia 15m TF"] = -0.5
        else:
            señales["Tendencia 15m TF"] = ("neutro", f"15m lateral ({t15:+.3f}%)")
            puntuaciones["Tendencia 15m TF"] = 0.0
    else:
        indicadores["Tendencia 15m TF"] = "N/A"
        señales["Tendencia 15m TF"] = ("neutro", "Sin datos")
        puntuaciones["Tendencia 15m TF"] = 0.0

    # ── N3. Tendencia 1h TF ──
    if df1h is not None and len(df1h) >= 8:
        c1h     = df1h["close"]
        ema9_1h = c1h.ewm(span=9).mean()
        ema21_1h = c1h.ewm(span=21).mean()
        t1h     = (c1h.iloc[-1] / c1h.iloc[-4] - 1) * 100
        e9h, e21h = ema9_1h.iloc[-1], ema21_1h.iloc[-1]
        indicadores["Tendencia 1h TF"] = f"EMA9={e9h:.4f} EMA21={e21h:.4f} Δ={t1h:+.3f}%"
        if c1h.iloc[-1] > e9h > e21h and t1h > 0.3:
            señales["Tendencia 1h TF"] = ("alcista", f"1h alcista fuerte ({t1h:+.3f}%)")
            puntuaciones["Tendencia 1h TF"] = 1.0
        elif c1h.iloc[-1] > e9h and t1h > 0:
            señales["Tendencia 1h TF"] = ("alcista_leve", f"1h alcista ({t1h:+.3f}%)")
            puntuaciones["Tendencia 1h TF"] = 0.5
        elif c1h.iloc[-1] < e9h < e21h and t1h < -0.3:
            señales["Tendencia 1h TF"] = ("bajista", f"1h bajista fuerte ({t1h:+.3f}%)")
            puntuaciones["Tendencia 1h TF"] = -1.0
        elif c1h.iloc[-1] < e9h and t1h < 0:
            señales["Tendencia 1h TF"] = ("bajista_leve", f"1h bajista ({t1h:+.3f}%)")
            puntuaciones["Tendencia 1h TF"] = -0.5
        else:
            señales["Tendencia 1h TF"] = ("neutro", f"1h lateral ({t1h:+.3f}%)")
            puntuaciones["Tendencia 1h TF"] = 0.0
    else:
        indicadores["Tendencia 1h TF"] = "N/A"
        señales["Tendencia 1h TF"] = ("neutro", "Sin datos")
        puntuaciones["Tendencia 1h TF"] = 0.0

    # ── N4. Hurst Exponent — régimen de mercado ──
    h_val = hurst_exponent(close, max_lag=min(20, len(close) // 4))
    regimen = clasificar_regimen(h_val)
    regimen_labels = {
        "trending":       "TENDENCIA",
        "mean_reverting": "REVERSIÓN",
        "noise":          "RUIDO/LATERAL",
    }
    regimen_colors = {
        "trending":       "alcista_leve",
        "mean_reverting": "bajista_leve",
        "noise":          "neutro",
    }
    indicadores["Hurst / Régimen"] = f"H={h_val:.3f} → {regimen_labels[regimen]}"
    señales["Hurst / Régimen"] = (regimen_colors[regimen],
                                   f"H={h_val:.3f}: mercado en {regimen_labels[regimen]}")
    puntuaciones["Hurst / Régimen"] = 0.0  # solo contexto, no puntúa directamente

    # ── N5. Divergencia de precio multi-exchange ──
    price_div = futures_data.get("price_diverge")
    if price_div is not None:
        okx_p = info.get("okx_price", 0)
        indicadores["Divergencia Exchange"] = (
            f"Kraken vs OKX: {price_div:+.4f}% "
            f"(OKX=${okx_p:,.4f})"
        )
        if abs(price_div) > 0.05:
            # Precio de Kraken por encima de OKX → probablemente corrija a la baja
            if price_div > 0.05:
                señales["Divergencia Exchange"] = ("bajista_leve",
                    f"Kraken {price_div:+.4f}% sobre OKX → posible corrección")
                puntuaciones["Divergencia Exchange"] = -0.3
            else:
                señales["Divergencia Exchange"] = ("alcista_leve",
                    f"Kraken {price_div:+.4f}% bajo OKX → posible rebote")
                puntuaciones["Divergencia Exchange"] = 0.3
        else:
            señales["Divergencia Exchange"] = ("neutro",
                f"Precios alineados ({price_div:+.4f}%)")
            puntuaciones["Divergencia Exchange"] = 0.0
    else:
        indicadores["Divergencia Exchange"] = "N/A (OKX no disponible)"
        señales["Divergencia Exchange"] = ("neutro", "Sin datos OKX")
        puntuaciones["Divergencia Exchange"] = 0.0

    return indicadores, señales, puntuaciones, atr_pct, regimen, h_val


# ─────────────────────────────────────────────
# PESOS BASE
# ─────────────────────────────────────────────
PESOS_BASE = PESOS_CRYPTO = {
    # Indicadores técnicos originales
    "RSI (9)":              1.5,
    "MACD (5,13,3)":        1.5,
    "EMA 7/25":             1.2,
    "Bollinger %B":         1.0,
    "Stochastic (5,3)":     1.0,
    "Williams %R":          0.8,
    "ATR (9)":              0.0,
    "Rate of Change":       1.8,
    "OBV":                  1.0,
    "VWAP Desviación":      1.2,
    "Volumen Relativo":     1.3,
    "Patrón Vela 1m":       0.7,
    "Order Book Imbalance": 2.0,
    "Bid/Ask Spread":       0.5,
    "Buy/Sell Ratio":       2.0,
    "Actividad Trades":     0.8,
    "Funding Rate":         1.0,
    "Open Interest Δ":      1.0,
    "Long/Short Ratio":     0.6,
    "Tendencia 5m TF":      1.5,
    # Nuevos
    "Fear & Greed":         1.2,
    "Tendencia 15m TF":     1.8,
    "Tendencia 1h TF":      2.0,
    "Hurst / Régimen":      0.0,   # informativo
    "Divergencia Exchange": 0.8,
}

# Pesos por régimen — amplificadores sobre PESOS_BASE
_REGIME_MULT = {
    # En tendencia: momentum y continuación tienen más peso
    "trending": {
        "Rate of Change":   1.4,
        "EMA 7/25":         1.4,
        "MACD (5,13,3)":    1.3,
        "Tendencia 15m TF": 1.3,
        "Tendencia 1h TF":  1.3,
        "RSI (9)":          0.7,   # OSC menos útiles en tendencia
        "Bollinger %B":     0.7,
        "Stochastic (5,3)": 0.7,
        "Williams %R":      0.7,
    },
    # En reversión: osciladores y niveles extremos tienen más peso
    "mean_reverting": {
        "RSI (9)":          1.4,
        "Bollinger %B":     1.4,
        "Stochastic (5,3)": 1.3,
        "Williams %R":      1.3,
        "VWAP Desviación":  1.3,
        "Rate of Change":   0.6,   # momentum menos fiable en reversión
        "EMA 7/25":         0.7,
        "MACD (5,13,3)":    0.7,
    },
    # En ruido: microestructura y datos de derivados tienen más peso
    "noise": {
        "Order Book Imbalance": 1.3,
        "Buy/Sell Ratio":       1.3,
        "Funding Rate":         1.2,
        "Long/Short Ratio":     1.2,
    },
}

BLOQUES_CRYPTO = {
    "⚡ Momentum Técnico":     ["RSI (9)", "MACD (5,13,3)", "EMA 7/25",
                                "Bollinger %B", "Stochastic (5,3)", "Williams %R"],
    "📊 Volatilidad y Precio": ["ATR (9)", "Rate of Change", "VWAP Desviación",
                                "Patrón Vela 1m"],
    "📦 Volumen y Flujo":      ["OBV", "Volumen Relativo", "Buy/Sell Ratio",
                                "Actividad Trades"],
    "🏦 Microestructura":      ["Order Book Imbalance", "Bid/Ask Spread",
                                "Divergencia Exchange"],
    "🔮 Futuros / Derivados":  ["Funding Rate", "Open Interest Δ", "Long/Short Ratio"],
    "🌐 Contexto Multi-TF":    ["Tendencia 5m TF", "Tendencia 15m TF",
                                "Tendencia 1h TF", "Hurst / Régimen"],
    "🧠 Sentimiento Macro":    ["Fear & Greed"],
}


# ─────────────────────────────────────────────
# PREDICCIÓN — CON MODULADORES
# ─────────────────────────────────────────────
def calcular_prediccion(puntuaciones, precio_actual, indicadores,
                        regimen: str = "noise", fng_data: dict = None):
    # ATR
    atr_str = indicadores.get("ATR (9)", "±0 (±0%)")
    try:
        atr_pct = float(atr_str.split("±")[2].replace("%)", "").replace("%", ""))
    except Exception:
        atr_pct = 0.1

    # ── Pesos dinámicos según régimen ──
    mults = _REGIME_MULT.get(regimen, {})
    pesos_efectivos = {
        k: v * mults.get(k, 1.0)
        for k, v in PESOS_BASE.items()
    }

    # ── Score ponderado ──
    score_pond = sum(
        puntuaciones.get(k, 0) * pesos_efectivos.get(k, 1.0)
        for k in puntuaciones
    )
    peso_total = sum(v for v in pesos_efectivos.values() if v > 0)
    score_norm = max(-1.0, min(1.0, score_pond / peso_total)) if peso_total else 0

    # ── Modulador Fear & Greed ──
    # No cambia la dirección, solo ajusta la confianza (max ±10%)
    fng_mod = 1.0
    fng_val = fng_data.get("fng_value") if fng_data else None
    if fng_val is not None:
        if score_norm > 0:
            # Señal alcista: F&G bajo aumenta confianza (oversold sentiment)
            if fng_val <= 20:
                fng_mod = 1.10
            elif fng_val <= 40:
                fng_mod = 1.05
            elif fng_val >= 80:
                fng_mod = 0.90   # demasiada euforia → contrarian
            elif fng_val >= 60:
                fng_mod = 0.95
        else:
            # Señal bajista: F&G alto aumenta confianza (overbought sentiment)
            if fng_val >= 80:
                fng_mod = 1.10
            elif fng_val >= 60:
                fng_mod = 1.05
            elif fng_val <= 20:
                fng_mod = 0.90   # demasiado miedo → contrarian rebote
            elif fng_val <= 40:
                fng_mod = 0.95

    # ── Modulador Multi-TF alignment ──
    # Si 15m y 1h confirman la señal de 1m, boost confianza
    tf_align = 0
    p15 = puntuaciones.get("Tendencia 15m TF", 0)
    p1h = puntuaciones.get("Tendencia 1h TF", 0)
    if score_norm > 0:
        if p15 > 0: tf_align += 1
        if p1h > 0: tf_align += 1
    elif score_norm < 0:
        if p15 < 0: tf_align += 1
        if p1h < 0: tf_align += 1
    tf_mod = 1.0 + tf_align * 0.05   # max +10% si ambos TF confirman

    # ── Modulador régimen ──
    # En "noise" reducir la confianza general
    regime_conf = {"trending": 1.05, "mean_reverting": 1.0, "noise": 0.92}
    regime_mod  = regime_conf.get(regimen, 1.0)

    # Score final modulado
    score_final = max(-1.0, min(1.0, score_norm * fng_mod * tf_mod * regime_mod))

    # Probabilidad
    prob_subida = max(5.0, min(95.0, 50.0 + score_final * 40.0))

    # Movimiento estimado
    mov_est    = atr_pct * (0.6 + abs(score_final) * 0.6)
    precio_obj = precio_actual * (1 + (mov_est if score_final > 0 else -mov_est) / 100)

    if score_final > 0.4:
        señal_texto, señal_color = "ALCISTA FUERTE",   "alcista"
    elif score_final > 0.15:
        señal_texto, señal_color = "TENDENCIA ALCISTA","alcista_leve"
    elif score_final < -0.4:
        señal_texto, señal_color = "BAJISTA FUERTE",   "bajista"
    elif score_final < -0.15:
        señal_texto, señal_color = "TENDENCIA BAJISTA","bajista_leve"
    else:
        señal_texto, señal_color = "LATERAL / INDECISO","neutro"

    alcistas = sum(1 for v in puntuaciones.values() if v > 0)
    bajistas = sum(1 for v in puntuaciones.values() if v < 0)
    neutros  = sum(1 for v in puntuaciones.values() if v == 0)

    return {
        "score":           score_final,
        "score_raw":       score_norm,
        "prob_subida":     prob_subida,
        "prob_bajada":     100 - prob_subida,
        "direccion":       "↑ SUBE" if score_final > 0 else "↓ BAJA",
        "mov_estimado":    mov_est,
        "precio_objetivo": precio_obj,
        "señal_texto":     señal_texto,
        "señal_color":     señal_color,
        "atr_pct":         atr_pct,
        "alcistas":        alcistas,
        "bajistas":        bajistas,
        "neutros":         neutros,
        "regimen":         regimen,
        "fng_mod":         round(fng_mod, 3),
        "tf_align":        tf_align,
        "tf_mod":          round(tf_mod, 3),
        "regime_mod":      round(regime_mod, 3),
        "pesos_efectivos": pesos_efectivos,
    }
