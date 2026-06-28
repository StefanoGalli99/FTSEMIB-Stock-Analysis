"""
Fase 5c — Controllo del FATTORE COMUNE di mercato (Granger condizionale).

In Fase 5b abbiamo visto predittivita' bidirezionale simmetrica anche fra titoli freschi:
la firma di un fattore comune (l'indice) a cui i titoli reagiscono con dinamiche correlate.
Qui lo "togliamo" e guardiamo cosa resta.

Granger condizionale 'x -> y | mercato':
    r_y(t) = c + sum a_i r_y(t-i) + sum b_i r_x(t-i) + g0 r_mkt(t) + sum g_i r_mkt(t-i) + e
    H0: b_1..b_p = 0  (x non aiuta a prevedere y, una volta tolto il mercato)
Includiamo il mercato CONTEMPORANEO e i suoi ritardi come controlli; inferenza HAC.

Combinato col controllo staleness di 5b (sottogruppo FRESCHI), questo e' il test piu' pulito:
se dopo aver tolto SIA i prezzi stantii SIA il mercato non resta direzionalita',
la conclusione "nessun lead-lag informativo" e' blindata.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy.stats import binomtest

from data_foundation import weekly_returns, DATA_DIR, START_DATE, END_DATE
from phase3_granger import granger_hac, _newey_west_lags
from phase5_universe import load_universe, staleness

MKT_CACHE = os.path.join(DATA_DIR, "ftsemib_index.pkl")
N_GROUP = 20
LAGS = [1, 2, 3, 4]
ALPHA = 0.05


def market_weekly() -> pd.Series:
    """Rendimento settimanale dell'indice FTSE MIB (con cache)."""
    idx = None
    if os.path.exists(MKT_CACHE):
        try:
            idx = pd.read_pickle(MKT_CACHE)
        except Exception as e:
            print(f"[!] Errore nel caricamento della cache dell'indice ({e}). Scarico nuovamente...")
    if idx is None:
        idx = yf.download("FTSEMIB.MI", start=START_DATE, end=END_DATE,
                          interval="1d", auto_adjust=False, progress=False)
        idx.to_pickle(MKT_CACHE)
    close = idx["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.resample("W-FRI").last().pct_change().rename("mkt")


def granger_hac_cond(y, x, mkt, p) -> float:
    """Granger 'x -> y' condizionato al mercato (contemporaneo + p ritardi). Ritorna p-value."""
    df = pd.concat([y.rename("y"), x.rename("x"), mkt.rename("m")], axis=1,
                   sort=False).sort_index().dropna()
    cols = {"y": df["y"], "m0": df["m"]}            # mercato contemporaneo
    for i in range(1, p + 1):
        cols[f"y_l{i}"] = df["y"].shift(i)
        cols[f"x_l{i}"] = df["x"].shift(i)
        cols[f"m_l{i}"] = df["m"].shift(i)
    data = pd.DataFrame(cols).dropna()
    Y = data["y"]
    X = sm.add_constant(data.drop(columns=["y"]))
    nw = max(_newey_west_lags(len(data)), p)
    res = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw})
    x_terms = [c for c in X.columns if c.startswith("x_l")]
    return float(np.squeeze(res.f_test(", ".join(f"{c}=0" for c in x_terms)).pvalue))


def asym(liquid, illiquid, wk, mkt=None):
    """Conta forward/backward significativi. Se mkt e' dato, usa Granger condizionale."""
    fwd = bwd = total = only_f = only_b = 0
    for liq in liquid:
        for ill in illiquid:
            fa = ba = False
            for L in LAGS:
                if mkt is None:
                    pf = granger_hac(wk[ill], wk[liq], L)["p"]
                    pb = granger_hac(wk[liq], wk[ill], L)["p"]
                else:
                    pf = granger_hac_cond(wk[ill], wk[liq], mkt, L)
                    pb = granger_hac_cond(wk[liq], wk[ill], mkt, L)
                total += 1
                if pf < ALPHA: fwd += 1; fa = True
                if pb < ALPHA: bwd += 1; ba = True
            only_f += int(fa and not ba); only_b += int(ba and not fa)
    m = only_f + only_b
    sp = binomtest(only_f, m, 0.5).pvalue if m > 0 else np.nan
    return {"fwd_pct": 100*fwd/total, "bwd_pct": 100*bwd/total,
            "only_f": only_f, "only_b": only_b, "sign_p": sp}


if __name__ == "__main__":
    daily, full = load_universe()
    wk = weekly_returns(daily)
    mkt = market_weekly()
    print(f"Indice FTSE MIB: {mkt.dropna().shape[0]} settimane "
          f"({mkt.dropna().index.min().date()} -> {mkt.dropna().index.max().date()})")

    full = full.copy()
    full["staleness"] = staleness(daily, full["ticker"]).reindex(full["ticker"]).values
    liquid = full.head(N_GROUP)["ticker"].tolist()
    ill_df = full.tail(N_GROUP).copy()
    med = ill_df["staleness"].median()
    fresh = ill_df[ill_df["staleness"] <= med]["ticker"].tolist()
    all_ill = ill_df["ticker"].tolist()

    print(f"\n{'specifica':<40}{'fwd %':>8}{'bwd %':>8}{'solo-f':>8}{'solo-b':>8}{'sign p':>9}")
    print("-" * 81)
    specs = [
        ("ILL=tutti   | senza controlli",         all_ill, None),
        ("ILL=tutti   | + controllo mercato",      all_ill, mkt),
        ("ILL=freschi | senza controlli",          fresh,   None),
        ("ILL=freschi | + controllo mercato",      fresh,   mkt),
    ]
    for name, grp, m in specs:
        r = asym(liquid, grp, wk, m)
        print(f"{name:<40}{r['fwd_pct']:>8.1f}{r['bwd_pct']:>8.1f}"
              f"{r['only_f']:>8}{r['only_b']:>8}{r['sign_p']:>9.4f}")
    print("-" * 81)
    print(f"(atteso sotto H0: fwd% = bwd% = {100*ALPHA:.0f}%)")
