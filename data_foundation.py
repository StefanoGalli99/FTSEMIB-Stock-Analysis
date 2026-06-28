"""
Fase 1 — Fondamenta dati per lo studio di Granger causality liquido <-> illiquido
sul FTSE MIB (Borsa Italiana).

Cosa fa questo modulo, e SOLO questo (niente Granger ancora):
  1. Scarica i prezzi GIORNALIERI (auto_adjust=False -> abbiamo Close grezzo + Adj Close + Volume).
  2. Calcola la illiquidita' di Amihud (2002) per ogni titolo, sull'intero campione.
  3. Ordina i titoli dal piu' liquido al piu' illiquido.
  4. Costruisce i rendimenti SETTIMANALI (venerdi') da usare poi nel test di Granger.
  5. Mette in cache i dati grezzi su disco, cosi' i passi successivi non riscaricano.

Definizioni (le fissiamo qui, una volta, in modo trasparente):
  - rendimento giornaliero  r_t  = Adj_Close_t / Adj_Close_{t-1} - 1     (total return)
  - controvalore € giornaliero  DVOL_t = Close_grezzo_t * Volume_t       (euro scambiati)
  - Amihud illiquidity per titolo:  ILLIQ = media_t( |r_t| / DVOL_t )
        valori ALTI  -> titolo ILLIQUIDO (il prezzo si muove molto per ogni € scambiato)
        valori BASSI -> titolo LIQUIDO
    Riportiamo ILLIQ * 1e9 solo per leggibilita' (la scala non cambia il ranking).
"""

from __future__ import annotations
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Cartella di cache dei dati grezzi (pickle)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DAILY_CACHE = os.path.join(DATA_DIR, "daily_raw.pkl")


TICKERS = [
    "BMPS.MI", "TEN.MI", "LDO.MI", "SPM.MI", "BAMI.MI", "FBK.MI", "G.MI", "IP.MI",
    "BC.MI", "AZM.MI", "A2A.MI", "IVG.MI", "RACE.MI", "PST.MI", "PIRC.MI", "DIA.MI", "PRY.MI",
    "SRG.MI", "STMMI.MI", "UNI.MI", "HER.MI", "REC.MI", "TRN.MI", "BPE.MI", "ENEL.MI", "NEXI.MI",
    "TIT.MI", "ERG.MI", "IG.MI", "BMED.MI", "CPR.MI", "MONC.MI", "INW.MI", "STLAM.MI",
    "ENI.MI", "AMP.MI", "MB.MI", "ISP.MI", "UCG.MI",
]


# ---------------------------------------------------------------------------
# 1. DOWNLOAD DATI GIORNALIERI (con cache)
# ---------------------------------------------------------------------------

def download_daily(tickers=TICKERS, start=START_DATE, end=END_DATE,
                   use_cache=True, cache_path=DAILY_CACHE) -> pd.DataFrame:
    """Scarica i daily per tutti i ticker. Ritorna un DataFrame con colonne MultiIndex
    (campo, ticker). Mette/legge una cache pickle per non riscaricare ogni volta.
    cache_path permette universi diversi (es. MIB-39 vs universo ampio)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if use_cache and os.path.exists(cache_path):
        print(f"[cache] leggo i daily da {cache_path}")
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"[!] Errore nel caricamento della cache ({e}). Scarico nuovamente...")

    print(f"[download] {len(tickers)} ticker, {start} -> {end} ...")
    df = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, group_by="column",
    )
    # scarta i ticker per cui Adj Close e' interamente NaN (download fallito)
    adj = df["Adj Close"]
    dead = [tk for tk in adj.columns if adj[tk].notna().sum() == 0]
    if dead:
        print(f"[!] scarto ticker senza dati: {', '.join(dead)}")
        keep = [tk for tk in adj.columns if tk not in dead]
        df = df.loc[:, (slice(None), keep)]
    df.to_pickle(cache_path)
    print(f"[cache] salvato in {cache_path}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. AMIHUD ILLIQUIDITY + COPERTURA DATI
# ---------------------------------------------------------------------------
def compute_amihud(daily: pd.DataFrame) -> pd.DataFrame:
    """Per ogni ticker calcola la illiquidita' di Amihud sull'intero campione e
    quante osservazioni valide ha (copertura). Ritorna un DataFrame ordinato dal
    piu' liquido al piu' illiquido."""
    adj = daily["Adj Close"]
    close = daily["Close"]
    vol = daily["Volume"]

    rows = []
    for tk in adj.columns:
        r = adj[tk].pct_change()
        dvol = close[tk] * vol[tk]          # controvalore in €
        # tieni solo giorni con rendimento definito e controvalore > 0
        mask = r.notna() & (dvol > 0)
        n = int(mask.sum())
        if n < 60:                          # meno di ~3 mesi di dati: inaffidabile
            rows.append((tk, np.nan, n, _first_valid(adj[tk]), _last_valid(adj[tk])))
            continue
        illiq = (r[mask].abs() / dvol[mask]).mean()
        rows.append((tk, illiq, n, _first_valid(adj[tk]), _last_valid(adj[tk])))

    out = pd.DataFrame(rows, columns=["ticker", "amihud", "n_obs", "start", "end"])
    out["amihud_x1e9"] = out["amihud"] * 1e9
    out = out.sort_values("amihud").reset_index(drop=True)
    out.insert(0, "rank_liquid", range(1, len(out) + 1))  # 1 = piu' liquido
    return out


def _first_valid(s: pd.Series):
    idx = s.first_valid_index()
    return None if idx is None else pd.Timestamp(idx).date()


def _last_valid(s: pd.Series):
    idx = s.last_valid_index()
    return None if idx is None else pd.Timestamp(idx).date()


# ---------------------------------------------------------------------------
# 3. RENDIMENTI SETTIMANALI (per il test di Granger, fase successiva)
# ---------------------------------------------------------------------------
def weekly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti settimanali (venerdi') da Adj Close: prendiamo l'ultimo prezzo
    aggiustato della settimana e calcoliamo la variazione %. Colonne = ticker."""
    adj = daily["Adj Close"]
    weekly_price = adj.resample("W-FRI").last()
    return weekly_price.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# ESECUZIONE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    daily = download_daily()

    rank = compute_amihud(daily)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 100)

    print("\n=== RANKING LIQUIDITA' (Amihud, intero campione) ===")
    print("rank_liquid=1 -> piu' LIQUIDO ; amihud alto -> piu' ILLIQUIDO\n")
    print(rank[["rank_liquid", "ticker", "amihud_x1e9", "n_obs", "start", "end"]].to_string(index=False))

    # ticker senza dati o con copertura insufficiente
    bad = rank[rank["amihud"].isna()]
    if len(bad):
        print("\n[!] Ticker scartati (dati assenti o copertura < 60 giorni):")
        print("    " + ", ".join(bad["ticker"].tolist()))

    wk = weekly_returns(daily)
    print(f"\n=== RENDIMENTI SETTIMANALI === shape={wk.shape} "
          f"({wk.index.min().date()} -> {wk.index.max().date()})")
