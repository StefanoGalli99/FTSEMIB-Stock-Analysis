"""
Fase 5b — Test sistematico dell'asimmetria CON controllo per i prezzi stantii.

Idea del controllo (kill-test): la causalita' spuria da non-synchronous trading vive nei
prezzi STANTII. Amihud (impatto) e staleness (giorni fermi) sono correlati ma non identici:
esistono illiquidi-per-impatto che scambiano ogni giorno. Allora spacchiamo il gruppo
illiquido in due meta' per staleness e rifacciamo il test:
  - se l'asimmetria liquida->illiquida vive SOLO negli illiquidi STANTII -> meccanica (brutto).
  - se sopravvive anche negli illiquidi FRESCHI (poco stantii) -> diffusione vera (bello).

Per ogni sottogruppo riportiamo, sui lag 1..4 messi insieme:
  forward  = quante volte liquida->illiquida e' significativa (5%)
  backward = quante volte illiquida->liquida e' significativa (placebo, atteso ~5%)
piu' un sign test sull'asimmetria.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import binomtest

from data_foundation import weekly_returns
from phase3_granger import granger_hac
from phase5_universe import load_universe, staleness

N_GROUP = 20      # quanti liquidi e quanti illiquidi
LAGS = [1, 2, 3, 4]
ALPHA = 0.05


def asymmetry(liquid, illiquid, wk):
    """Conta forward/backward significativi su tutte le coppie liquid x illiquid, lag 1..4."""
    fwd_hits = bwd_hits = total = 0
    only_fwd = only_bwd = 0
    for liq in liquid:
        for ill in illiquid:
            f_any = b_any = False
            for L in LAGS:
                pf = granger_hac(wk[ill], wk[liq], L)["p"]   # liquida -> illiquida
                pb = granger_hac(wk[liq], wk[ill], L)["p"]   # illiquida -> liquida
                total += 1
                if pf < ALPHA:
                    fwd_hits += 1; f_any = True
                if pb < ALPHA:
                    bwd_hits += 1; b_any = True
            only_fwd += int(f_any and not b_any)
            only_bwd += int(b_any and not f_any)
    m = only_fwd + only_bwd
    sign_p = binomtest(only_fwd, m, 0.5).pvalue if m > 0 else np.nan
    return {"npairs": len(liquid) * len(illiquid), "total_tests": total,
            "fwd_pct": 100 * fwd_hits / total, "bwd_pct": 100 * bwd_hits / total,
            "only_fwd": only_fwd, "only_bwd": only_bwd, "sign_p": sign_p}


if __name__ == "__main__":
    daily, full = load_universe()
    wk = weekly_returns(daily)

    full = full.copy()
    full["staleness"] = staleness(daily, full["ticker"]).reindex(full["ticker"]).values

    liquid = full.head(N_GROUP)["ticker"].tolist()
    ill_df = full.tail(N_GROUP).copy()
    med = ill_df["staleness"].median()
    fresh = ill_df[ill_df["staleness"] <= med]["ticker"].tolist()   # poco stantii
    stale = ill_df[ill_df["staleness"] > med]["ticker"].tolist()    # molto stantii

    print(f"LIQUIDI ({len(liquid)}): {', '.join(t.replace('.MI','') for t in liquid)}")
    print(f"ILLIQUIDI FRESCHI ({len(fresh)}, staleness<= {med:.3f}): "
          f"{', '.join(t.replace('.MI','') for t in fresh)}")
    print(f"ILLIQUIDI STANTII ({len(stale)}, staleness>  {med:.3f}): "
          f"{', '.join(t.replace('.MI','') for t in stale)}\n")

    print(f"{'sottogruppo illiquido':<26}{'coppie':>7}{'fwd %':>8}{'bwd %':>8}"
          f"{'solo-fwd':>10}{'solo-bwd':>10}{'sign p':>9}")
    for name, grp in [("TUTTI", full.tail(N_GROUP)["ticker"].tolist()),
                      ("FRESCHI (poco stantii)", fresh),
                      ("STANTII (molto stantii)", stale)]:
        r = asymmetry(liquid, grp, wk)
        print(f"{name:<26}{r['npairs']:>7}{r['fwd_pct']:>8.1f}{r['bwd_pct']:>8.1f}"
              f"{r['only_fwd']:>10}{r['only_bwd']:>10}{r['sign_p']:>9.4f}")

    print(f"\n(atteso sotto H0: fwd% = bwd% = {100*ALPHA:.0f}%)")
