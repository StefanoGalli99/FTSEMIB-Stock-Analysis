"""
Fase 2 — Diagnostiche pre-Granger sulle serie settimanali.

Il test di Granger ha due prerequisiti che lo script vecchio saltava (ed e' la causa
piu' comune di risultati spuri):

  A) STAZIONARIETA'. Granger assume serie stazionarie. La verifichiamo con DUE test
     che hanno ipotesi nulle opposte, e li incrociamo:
       - ADF  (Augmented Dickey-Fuller):  H0 = "c'e' radice unitaria" (NON stazionaria).
              p-value basso  -> RIFIUTO H0 -> serie stazionaria.
       - KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = "serie stazionaria".
              p-value basso  -> RIFIUTO H0 -> serie NON stazionaria.
     Verdetto solido di stazionarieta': ADF rifiuta E KPSS non rifiuta.

  B) SCELTA DEL LAG. Invece di prendere il lag col p-value piu' basso (data snooping,
     errore del vecchio codice), adattiamo un VAR alla coppia e lasciamo scegliere il
     numero di lag a AIC/BIC. Quel lag lo useremo poi nel test di Granger (Fase 3).

Questo modulo SOLO diagnostica: non esegue ancora Granger.
"""

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR

from data_foundation import download_daily, weekly_returns

# ---------------------------------------------------------------------------
# Coppie scelte con l'utente: stesso settore + cross-settore.
# (liquida, illiquida)
# ---------------------------------------------------------------------------
PAIRS = [
    # --- stesso settore ---
    ("ENI.MI",  "ERG.MI",  "energia"),
    ("ENEL.MI", "HER.MI",  "utility"),
    ("RACE.MI", "BC.MI",   "lusso"),
    # --- cross-settore ---
    ("ISP.MI",  "BC.MI",   "banca/lusso"),
    ("UCG.MI",  "DIA.MI",  "banca/healthcare"),
    ("ENI.MI",  "AMP.MI",  "energia/healthcare"),
]

MAX_LAG = 8  # lag massimo (settimane) considerato nella selezione del VAR


# ---------------------------------------------------------------------------
# A) STAZIONARIETA'
# ---------------------------------------------------------------------------
def stationarity_report(series: pd.Series) -> dict:
    """ADF + KPSS su una serie. Ritorna p-value di entrambi e un verdetto incrociato."""
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS avvisa quando il p-value e' fuori tabella
        kpss_p = kpss(s, regression="c", nlags="auto")[1]

    adf_stationary = adf_p < 0.05            # ADF rifiuta radice unitaria
    kpss_stationary = kpss_p >= 0.05         # KPSS NON rifiuta stazionarieta'
    if adf_stationary and kpss_stationary:
        verdict = "STAZIONARIA"
    elif not adf_stationary and not kpss_stationary:
        verdict = "NON stazionaria"
    else:
        verdict = "ambigua"
    return {"n": len(s), "adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict}


# ---------------------------------------------------------------------------
# B) SCELTA DEL LAG (VAR + AIC/BIC) su una coppia allineata
# ---------------------------------------------------------------------------
def lag_selection(pair_df: pd.DataFrame, maxlag=MAX_LAG) -> dict:
    """Adatta un VAR alla coppia (gia' allineata, senza NaN) e ritorna i lag
    suggeriti da AIC e BIC."""
    sel = VAR(pair_df).select_order(maxlags=maxlag)
    return {"aic": int(sel.aic), "bic": int(sel.bic),
            "fpe": int(sel.fpe), "hqic": int(sel.hqic)}


# ---------------------------------------------------------------------------
# ESECUZIONE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    daily = download_daily()
    wk = weekly_returns(daily)

    # --- A) stazionarieta' su ogni titolo coinvolto nelle coppie ---
    used = sorted({t for liq, ill, _ in PAIRS for t in (liq, ill)})
    print("=== A) STAZIONARIETA' (rendimenti settimanali) ===")
    print(f"{'ticker':<10}{'n':>5}{'ADF p':>10}{'KPSS p':>10}   verdetto")
    for tk in used:
        r = stationarity_report(wk[tk])
        print(f"{tk:<10}{r['n']:>5}{r['adf_p']:>10.4f}{r['kpss_p']:>10.4f}   {r['verdict']}")

    # --- B) scelta del lag per ogni coppia ---
    print("\n=== B) SCELTA DEL LAG (VAR, max 8 settimane) ===")
    print(f"{'coppia':<22}{'settore':<20}{'n_comune':>9}{'AIC':>6}{'BIC':>6}{'HQIC':>6}")
    for liq, ill, settore in PAIRS:
        pair = wk[[liq, ill]].dropna()
        sel = lag_selection(pair)
        name = f"{liq}->{ill}"
        print(f"{name:<22}{settore:<20}{len(pair):>9}{sel['aic']:>6}{sel['bic']:>6}{sel['hqic']:>6}")
