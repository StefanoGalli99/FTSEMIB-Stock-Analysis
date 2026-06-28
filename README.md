# Lead-lag di liquidità sul FTSE MIB — esiste davvero?

Studio empirico di **causalità di Granger tra azioni liquide e illiquide** della Borsa
Italiana. Domanda di ricerca: *i rendimenti delle azioni liquide anticipano (Granger-causano)
quelli delle azioni illiquide?* — l'effetto "lead-lag" di Lo & MacKinlay (1990) e
Chordia & Swaminathan (2000).

## Conclusione (in breve)

> **No.** Non troviamo evidenza di un lead-lag *informativo genuino* dalle liquide alle
> illiquide. L'asimmetria che a prima vista sembra esserci è interamente spiegata da due
> confondenti: (a) i **prezzi stantii** delle illiquide (non-synchronous trading) e (b) la
> reazione comune al **fattore di mercato**. Una volta rimossi entrambi, la predittività
> direzionale scende al livello del rumore (~5%, il tasso di falsi positivi atteso).

È un **risultato negativo difeso contro i confondenti**: ha resistito a ogni controllo che gli
abbiamo portato. Il valore dello studio sta proprio nei controlli — senza, avremmo riportato un
lead-lag spurio.

## Metodo e definizioni

- **Liquidità**: illiquidità di **Amihud (2002)** = media di `|rendimento giornaliero| /
  controvalore€ giornaliero`. Alto = illiquido. Calcolata su dati giornalieri.
- **Rendimenti**: **settimanali** (venerdì), da prezzi aggiustati. La frequenza settimanale
  attenua (non elimina) il rumore microstrutturale.
- **Test di Granger**: regressione a equazione singola con covarianza **HAC (Newey-West)**,
  robusta all'eteroschedasticità che KPSS rivela nei rendimenti. Test bidirezionale.
- **Scelta del lag**: niente cherry-picking — sweep fisso 1–4 settimane (+ lag AIC dove serve).
- **Controllo prezzi stantii**: misura di staleness di Lesmond-Ogden-Trzcinka (1999), frazione
  di giorni a rendimento zero; gli illiquidi si spaccano in "freschi" vs "stantii".
- **Controllo fattore comune**: Granger condizionale all'indice FTSE MIB (contemporaneo + lag).

## Pipeline (eseguire in ordine)

| File | Fase | Cosa fa |
|------|------|---------|
| `data_foundation.py`     | 1  | Download daily (cache), Amihud, ranking liquidità, rendimenti settimanali |
| `phase2_diagnostics.py`  | 2  | Stazionarietà (ADF + KPSS) e scelta lag (VAR + AIC/BIC) |
| `phase3_granger.py`      | 3  | Granger bidirezionale HAC sulle coppie, correzione FDR |
| `phase4_systematic.py`   | 4  | Test sistematico dell'asimmetria su molte coppie (MIB) |
| `phase5_universe.py`     | 5a | Universo ampliato (~70 titoli) + misura di staleness |
| `phase5b_controlled.py`  | 5b | Kill-test: asimmetria con controllo staleness (freschi vs stantii) |
| `phase5c_market_control.py` | 5c | Granger condizionale al fattore di mercato |

## Risultati chiave

**Validazione dati** (Fase 1/5a): il ranking Amihud è economicamente sensato (ISP/ENI/ENEL/UCG
i più liquidi; small-cap come Tesmec/Brunello Cucinelli i più illiquidi). Spread di liquidità
fra i due estremi ~163×. Staleness: illiquidi 5.2% vs liquidi 1.2% dei giorni (4.4×).

**Il kill-test** (Fase 5b) — l'asimmetria sparisce dove i prezzi non sono stantii:

| Sottogruppo illiquido | liquida→illiquida | illiquida→liquida | sign test |
|---|---|---|---|
| Freschi (scambiano spesso) | 11.5% | 11.5% | p=1.00 |
| Stantii (prezzi vecchi)    | 15.0% | 6.1%  | p=0.05 |

**Il controllo del mercato** (Fase 5c) — la direzione dell'ipotesi crolla al rumore:

| Specifica | liquida→illiquida | illiquida→liquida |
|---|---|---|
| Senza controlli            | 13.2% | 8.8% |
| + controllo mercato        | **4.9%** | 8.6% |

(atteso sotto l'ipotesi nulla: 5% in entrambe le direzioni)

## Limiti dichiarati

- **Survivorship bias**: l'universo è "di oggi", esclude i delistati (es. Tod's e Saras,
  spariti nel 2024). Non eliminabile con yfinance.
- **Staleness ed estrema illiquidità Amihud** sono parzialmente confusi: i titoli più stantii
  sono anche i più estremi per impatto di prezzo. Separarli richiederebbe più titoli.
- **Indice cap-weighted**: il FTSE MIB è dominato dalle large cap liquide, quindi il controllo
  di mercato sovrappone l'indice ai titoli liquidi — motivo per cui il residuo illiquida→liquida
  va letto con cautela (probabile artefatto, e comunque non significativo).
- **Qualità dati yfinance** su small cap: più buchi, storie più corte.

## Ambiente

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install statsmodels pandas numpy yfinance matplotlib scipy
python data_foundation.py        # e poi le fasi successive
```

I dati grezzi vengono messi in cache in `data/` (pickle) per non riscaricarli a ogni run.

## Riferimenti

- Amihud, Y. (2002). *Illiquidity and stock returns*. Journal of Financial Markets.
- Chordia, T., & Swaminathan, B. (2000). *Trading Volume and Cross-Autocorrelations in Stock
  Returns*. Journal of Finance.
- Lo, A., & MacKinlay, C. (1990). *When are contrarian profits due to stock market
  overreaction?*. Review of Financial Studies.
- Lesmond, Ogden, & Trzcinka (1999). *A new estimate of transaction costs*. RFS.
- Hong, H., & Stein, J. (1999). *A unified theory of underreaction, momentum trading and
  overreaction in asset markets*. Journal of Finance.
