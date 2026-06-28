"""
Fase 3 — Test di Granger causality bidirezionale, robusto.

Differenze chiave rispetto al vecchio codice (che era sbagliato):
  - DIREZIONE CORRETTA. Testiamo esplicitamente entrambe le direzioni:
        liquida -> illiquida   e   illiquida -> liquida
    Il vecchio `grangercausalitytests([df[col], returns])` testava di fatto il contrario
    di quel che diceva di testare (in statsmodels la 2a colonna causa la 1a).
  - INFERENZA HAC (Newey-West). Stimiamo la regressione di Granger con OLS e covarianza
    robusta a eteroschedasticita' e autocorrelazione, perche' KPSS (Fase 2) ha mostrato
    varianza non costante su alcune serie -> il test F classico avrebbe p-value gonfiati.
  - NIENTE CHERRY-PICKING DEL LAG. Riportiamo uno sweep fisso di lag 1..4 (orizzonte
    economico del lead-lag settimanale) PIU' il lag scelto da AIC. Mostriamo tutti i lag,
    non solo quello "fortunato".
  - CORREZIONE PER TEST MULTIPLI. Sul set primario (lag AIC, 6 coppie x 2 direzioni = 12
    test) applichiamo Benjamini-Hochberg (FDR) per controllare i falsi positivi.

Test di Granger "X -> Y" come regressione a equazione singola:
    y_t = c + sum_{i=1..p} a_i y_{t-i} + sum_{i=1..p} b_i x_{t-i} + e_t
    H0: b_1 = ... = b_p = 0   (i lag di X non aiutano a prevedere Y)
    Wald F-test su {b_i} con covarianza HAC.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests

from data_foundation import download_daily, weekly_returns
from phase2_diagnostics import PAIRS, MAX_LAG


def _newey_west_lags(n: int) -> int:
    """Bandwidth automatica di Newey-West: floor(4*(n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def granger_hac(y: pd.Series, x: pd.Series, p: int) -> dict:
    """Test di Granger 'x -> y' a lag p, con covarianza HAC.
    Ritorna F, p-value, n osservazioni effettive."""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    cols = {"y": df["y"]}
    for i in range(1, p + 1):
        cols[f"y_l{i}"] = df["y"].shift(i)
        cols[f"x_l{i}"] = df["x"].shift(i)
    data = pd.DataFrame(cols).dropna()

    Y = data["y"]
    X = sm.add_constant(data.drop(columns=["y"]))
    nw = max(_newey_west_lags(len(data)), p)
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw})

    x_terms = [c for c in X.columns if c.startswith("x_l")]
    ftest = res.f_test(", ".join(f"{c}=0" for c in x_terms))
    return {"F": float(np.squeeze(ftest.fvalue)),
            "p": float(np.squeeze(ftest.pvalue)),
            "n": len(data)}


def aic_lag(pair_df: pd.DataFrame, maxlag=MAX_LAG) -> int:
    """Lag scelto da AIC sul VAR della coppia, con minimo 1 (per Granger serve p>=1)."""
    return max(int(VAR(pair_df).select_order(maxlags=maxlag).aic), 1)


if __name__ == "__main__":
    daily = download_daily()
    wk = weekly_returns(daily)

    SWEEP = [1, 2, 3, 4]
    star = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    primary = []  # (etichetta, p-value) al lag AIC, per la correzione multipla

    for liq, ill, settore in PAIRS:
        pair = wk[[liq, ill]].dropna()
        p_aic = aic_lag(pair)
        lags = sorted(set(SWEEP) | {p_aic})

        print(f"\n=== {liq}  <->  {ill}   ({settore})   n={len(pair)}, lag AIC={p_aic} ===")
        print(f"{'direzione':<26}{'lag':>4}{'F':>9}{'p-value':>11}")
        for caused_liq, (causing, caused) in [
            (False, (liq, ill)),   # liquida -> illiquida
            (True,  (ill, liq)),   # illiquida -> liquida
        ]:
            label = f"{causing} -> {caused}"
            for L in lags:
                r = granger_hac(wk[caused], wk[causing], L)
                tag = " (AIC)" if L == p_aic else ""
                print(f"{label:<26}{L:>4}{r['F']:>9.3f}{r['p']:>11.4f} {star(r['p'])}{tag}")
                if L == p_aic:
                    primary.append((f"{label} @lag{L}", r["p"]))
            print()

    # --- correzione per test multipli sul set primario (lag AIC) ---
    print("=== CORREZIONE TEST MULTIPLI (Benjamini-Hochberg, set primario lag=AIC) ===")
    labels = [a for a, _ in primary]
    pvals = [b for _, b in primary]
    rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    print(f"{'test':<34}{'p raw':>9}{'p FDR':>9}  signif@5%")
    for lab, p0, pa, rj in sorted(zip(labels, pvals, p_adj, rej), key=lambda z: z[2]):
        print(f"{lab:<34}{p0:>9.4f}{pa:>9.4f}  {'SI' if rj else 'no'}")
