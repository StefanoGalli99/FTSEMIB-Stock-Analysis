"""
Fase 4 — Test SISTEMATICO dell'asimmetria lead-lag su molte coppie.

Logica: una singola coppia significativa puo' essere fortuna. Ma se prendiamo TUTTE
le coppie liquida x illiquida e l'ipotesi di Chordia-Swaminathan e' vera, allora la
direzione liquida->illiquida deve risultare significativa MOLTO piu' spesso della
direzione inversa. Sotto l'ipotesi nulla (nessun lead-lag) entrambe dovrebbero
risultare significative solo nel ~5% dei casi, in modo simmetrico.

Disegno (pre-registrato, niente cherry-picking):
  - Titoli: i 6 piu' liquidi e gli 8 piu' illiquidi del ranking Amihud, MA solo quelli
    con copertura piena dal 2015 (n_obs alto) -> campioni settimanali allineati ~599.
  - Per ogni coppia (liquida, illiquida) e per lag = 1,2,3,4:
        forward  = Granger  liquida -> illiquida
        backward = Granger  illiquida -> liquida   (HAC, dalla Fase 3)
  - Aggregazione per lag: quante forward e backward significative al 5%, grezze e dopo FDR.
    Confronto col valore atteso sotto H0 (~5% delle coppie).
  - SIGN TEST sull'asimmetria: tra le coppie in cui esattamente UNA direzione e'
    significativa, quante sono "forward"? Sotto H0 e' una binomiale(50%).
"""

from __future__ import annotations
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from data_foundation import download_daily, weekly_returns, compute_amihud
from phase3_granger import granger_hac

N_LIQUID = 6
N_ILLIQUID = 8
LAGS = [1, 2, 3, 4]
ALPHA = 0.05


def select_universe(daily) -> tuple[list[str], list[str]]:
    """Sceglie i titoli liquidi e illiquidi con copertura piena dal 2015."""
    rank = compute_amihud(daily)
    full = rank[(rank["start"] <= date(2015, 2, 1)) &
                (rank["n_obs"] >= 2900) &
                rank["amihud"].notna()].reset_index(drop=True)
    liquid = full.head(N_LIQUID)["ticker"].tolist()
    illiquid = full.tail(N_ILLIQUID)["ticker"].tolist()
    return liquid, illiquid


if __name__ == "__main__":
    daily = download_daily()
    wk = weekly_returns(daily)
    liquid, illiquid = select_universe(daily)

    print(f"LIQUIDE   ({len(liquid)}): {', '.join(liquid)}")
    print(f"ILLIQUIDE ({len(illiquid)}): {', '.join(illiquid)}")
    pairs = [(l, i) for l in liquid for i in illiquid]
    print(f"Coppie totali: {len(pairs)}  (x2 direzioni x {len(LAGS)} lag)\n")

    print(f"{'lag':>4}{'#pairs':>8}{'fwd sig':>9}{'bwd sig':>9}{'atteso H0':>11}"
          f"{'fwd FDR':>9}{'bwd FDR':>9}{'sign test p':>13}")

    for L in LAGS:
        fwd_p, bwd_p = [], []
        for liq, ill in pairs:
            fwd_p.append(granger_hac(wk[ill], wk[liq], L)["p"])  # liquida -> illiquida
            bwd_p.append(granger_hac(wk[liq], wk[ill], L)["p"])  # illiquida -> liquida
        fwd_p = np.array(fwd_p); bwd_p = np.array(bwd_p)

        fwd_sig = int((fwd_p < ALPHA).sum())
        bwd_sig = int((bwd_p < ALPHA).sum())
        expected = ALPHA * len(pairs)
        fwd_fdr = int(multipletests(fwd_p, alpha=ALPHA, method="fdr_bh")[0].sum())
        bwd_fdr = int(multipletests(bwd_p, alpha=ALPHA, method="fdr_bh")[0].sum())

        # sign test: coppie con esattamente una direzione significativa
        only_fwd = int(((fwd_p < ALPHA) & (bwd_p >= ALPHA)).sum())
        only_bwd = int(((bwd_p < ALPHA) & (fwd_p >= ALPHA)).sum())
        m = only_fwd + only_bwd
        sign_p = binomtest(only_fwd, m, 0.5).pvalue if m > 0 else np.nan

        print(f"{L:>4}{len(pairs):>8}{fwd_sig:>9}{bwd_sig:>9}{expected:>11.1f}"
              f"{fwd_fdr:>9}{bwd_fdr:>9}{sign_p:>13.4f}"
              f"   (solo-fwd={only_fwd}, solo-bwd={only_bwd})")
