"""
Fase 5a — Universo ampio (Borsa Italiana, ~100 nomi) + misura di staleness.

Allarghiamo oltre il FTSE MIB perche' le "illiquide" del MIB sono comunque liquide in
assoluto (sono il fondo della classe dei piu' grandi). Con mid/small cap otteniamo
illiquidita' VERA e uno spread di liquidita' piu' ampio -> piu' potenza per il test
lead-lag. Prezzo da pagare: i prezzi stantii (non-synchronous trading) peggiorano, per
questo qui calcoliamo gia' la misura di staleness che useremo come controllo (Fase 5b).

STALENESS (Lesmond-Ogden-Trzcinka 1999): frazione di giorni di borsa con rendimento
ESATTAMENTE zero. Un titolo che non scambia ha prezzo invariato -> rendimento zero.
Tanti zeri = prezzi stantii = rischio di lead-lag spurio (meccanico, non informativo).

Nota survivorship bias: la lista e' "di oggi", quindi esclude i delistati (es. Tod's,
Saras spariti nel 2024). E' un limite noto e dichiarato, non eliminabile con yfinance.
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd

from data_foundation import download_daily, compute_amihud, weekly_returns, DATA_DIR

EXTENDED_CACHE = os.path.join(DATA_DIR, "daily_extended.pkl")

# Universo candidato (~100). I ticker che non scaricano o con copertura corta verranno
# filtrati a valle: la lista e' volutamente generosa.
_CANDIDATES = """
BMPS TEN LDO SPM BAMI FBK G IP BC AZM A2A IVG RACE PST PIRC DIA PRY SRG STMMI UNI HER REC
TRN BPE ENEL NEXI TIT ERG IG BMED CPR MONC INW STLAM ENI AMP MB ISP UCG
ANIM BFF BGN CE DLG BZU IGD IRE ZV ELC ENAV FILA WIIT SES REY TNXT CALT CEM GVS MN TGYM
SRS MARR BSS LTMC FCT OVS BRE IF ACE DAN FNM SOL ASC EM CMB TIP DEA CIR ELN TES PINF
PIA TXT TOD SFER AEF GEO BAN ILTY DOV JUVE SSL ASR AVIO RCS EXAI IWB MT DAL FOS BPSO
""".split()
EXTENDED_TICKERS = sorted({c + ".MI" for c in _CANDIDATES})

MIN_OBS = 2900  # copertura piena dal 2015 (~590+ settimane)


def load_universe():
    """Scarica/carica l'universo ampio e ritorna (daily, ranking_amihud_filtrato)."""
    daily = download_daily(tickers=EXTENDED_TICKERS, cache_path=EXTENDED_CACHE)
    rank = compute_amihud(daily)
    full = rank[(rank["n_obs"] >= MIN_OBS) & rank["amihud"].notna()].reset_index(drop=True)
    return daily, full


def staleness(daily: pd.DataFrame, tickers) -> pd.Series:
    """Frazione di giorni con rendimento esattamente zero (proxy di prezzi stantii)."""
    adj = daily["Adj Close"]
    out = {}
    for tk in tickers:
        r = adj[tk].pct_change().dropna()
        out[tk] = float((r == 0).mean())
    return pd.Series(out, name="staleness")


if __name__ == "__main__":
    daily, full = load_universe()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)

    full = full.copy()
    full["staleness"] = staleness(daily, full["ticker"]).reindex(full["ticker"]).values

    print(f"\n=== UNIVERSO PULITO: {len(full)} titoli (copertura piena dal 2015) ===\n")
    print("--- 10 PIU' LIQUIDI ---")
    print(full.head(10)[["rank_liquid", "ticker", "amihud_x1e9", "staleness"]].to_string(index=False))
    print("\n--- 10 PIU' ILLIQUIDI ---")
    print(full.tail(10)[["rank_liquid", "ticker", "amihud_x1e9", "staleness"]].to_string(index=False))

    # confronto staleness tra terzili di liquidita'
    k = len(full) // 3
    liq_grp = full.head(k)
    ill_grp = full.tail(k)
    print(f"\n=== STALENESS: liquidi vs illiquidi (terzili, k={k}) ===")
    print(f"  liquidi   : staleness media = {liq_grp['staleness'].mean():.4f}")
    print(f"  illiquidi : staleness media = {ill_grp['staleness'].mean():.4f}")
    print(f"  rapporto  : {ill_grp['staleness'].mean() / max(liq_grp['staleness'].mean(),1e-9):.1f}x")

    print(f"\n=== SPREAD DI LIQUIDITA' (Amihud x1e9) ===")
    print(f"  mediana liquidi   : {liq_grp['amihud_x1e9'].median():.3f}")
    print(f"  mediana illiquidi : {ill_grp['amihud_x1e9'].median():.3f}")
    print(f"  rapporto          : {ill_grp['amihud_x1e9'].median()/liq_grp['amihud_x1e9'].median():.0f}x")
