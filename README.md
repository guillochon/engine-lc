# Light Curves and Demographics of Tidal Disruptions in Self-Sustaining Black Hole Engines

Manuscript source, bibliography, figures, and the population-synthesis / light-curve
pipeline behind Guillochon et al. Every number, table row, and figure in `engine-lc.tex`
is produced by the commands below.

## Layout

| Path | Contents |
| --- | --- |
| `engine-lc.tex`, `engine-lc.bib` | Manuscript (AASTeX 7.0.1, `pdftex` class option) and ADS-generated bibliography |
| `figures/`, `tables/` | Generated figures and table rows included by the manuscript |
| `products/results_<tag>.json` | Post-processed statistics for the fiducial model and each variant; the text quotes these |
| `products/popsynth_summary.json`, `products/engine_window_f022.json` | Population-synthesis summary and the engine equilibrium at the window midpoint (Table 1) |
| `scripts/engine_window_exponents.py` | Table 1 from the `tde-engine` equilibrium solver |
| `scripts/popsynth.py` | Population synthesis: engine and field Monte Carlo catalogs |
| `scripts/run_lcs.py` | MOSFiT `tde_shock` light curves for a catalog (f_rad log-uniform 0.02-0.07; epsilon_acc log-normal about 0.03, the disk term capped at 0.1 L_Edd (the thermal UV/optical share of an Eddington-limited disk; `--leddlim`) and prompt, the collision term capped at L_Edd; `--darkyear 1` restores the dark-year viscous delay with 0.5 dex scatter; photosphere kept inside the MOSFiT wind envelope) |
| `scripts/postprocess.py` | Disk screen, infrared echoes, luminosity functions, survey yields, figures, table rows |
| `scripts/variant_table.py` | Collects the variant results into `tables/variant_rows.tex` |
| `scripts/run_all.py` | Runs steps 3-4 below for every catalog, several catalogs at a time (`--workers N`); a tag `<catalog>_L<value>` reruns a catalog with the disk cap at `<value>` L_Edd (`run_lcs.py --leddlim`), and `_p<value>` with the super-Eddington exponent p (`--eddslope`) |
| `scripts/parse_yao2023.py` | Per-event ZTF sample of Yao et al. (2023) with host colours and green-valley probabilities, used in Figures 2 and 3 |
| `scripts/build_bib.py`, `scripts/ads_validate.py` | Bibliography from ADS bibcodes (`scripts/bibcodes.json`) |
| `scripts/check_abstract.py` | arXiv abstract length / macro check |
| `scripts/nicholl2022_posteriors.json` | MOSFiT posteriors of Nicholl et al. (2022) from which nuisance parameters are drawn |
| `modules/observables/filterrules.json` | MOSFiT filter rules adding Euclid VIS/NISP and SPHEREx top-hat bands |

## Environment

* Python 3.11+ with the packages in `requirements.txt`.
* MOSFiT 2.0 from the `tde-shock` branch of https://github.com/guillochon/MOSFiT (which adds the
  two-component `tde_shock` model used here: prompt collision-powered emission with
  epsilon = f_rad r_g/r_p plus capped accretion, prompt by default)
  (the viscous-delay recurrence of PR 250 is required), installed with `uv sync`; `numba` is
  needed for the viscous kernel. Below, `PY` is that environment's Python.
* The engine equilibrium solver from https://github.com/guillochon/tde-engine, checked out next
  to this repository; point `TDE_ENGINE_REPO` at it.
* An ADS API token in `.env` as `ADS_TOKEN=...` (bibliography only).
* A TeX distribution with `aastex` (7.0.1) and `latexmk`.

All commands are run from the repository root so that MOSFiT picks up
`modules/observables/filterrules.json`. MOSFiT downloads SVO filter curves into
`modules/observables/filters/` on first use; create that directory if it does not exist.

## Reproducing the paper

```bash
# 1. Engine equilibrium at the flattening-window midpoint (Table 1)
TDE_ENGINE_REPO=../tde-engine python scripts/engine_window_exponents.py

# 2. Population synthesis: fiducial (20000 events per population) and every variant (8000 each)
PY scripts/popsynth.py

# 3. MOSFiT light curves, one library per catalog (~0.02 s per event)
for c in fiducial prompt gaslimited burstrelation simple young1 young0 KH13 miller; do
  PY scripts/run_lcs.py catalog_$c.npz
done

# 4. Post-processing: screen, echoes, statistics, figures (fiducial only), table rows
for c in fiducial prompt gaslimited burstrelation simple young1 young0 KH13 miller; do
  PY scripts/postprocess.py $c
done
python scripts/variant_table.py
# (steps 3-4 in one go, catalogs in parallel, logging to products/run_all.log: PY scripts/run_all.py --workers 8)

# 5. Observed comparison sample (Figure 2); needs the Yao et al. (2023) text in research/y2023/
PY scripts/parse_yao2023.py

# 6. Bibliography from ADS (needs .env) and manuscript
export $(cat .env | tr -d '\r')
python scripts/build_bib.py
python scripts/check_abstract.py
latexmk -pdf engine-lc.tex
```

Steps 2–4 take roughly an hour on a laptop and write the catalogs and light-curve libraries
(`products/*.npz`, about 1.2 GB), which are not tracked here. Step 5 is optional: its output,
`scripts/yao2023_sample.json`, is included.

## Model definitions

The fiducial model uses slowed circularization (viscous times from the encounter geometry,
following Guillochon & Ramirez-Ruiz 2015), a resupplied disk (cusp-limited engine lifetime),
black holes that grow at 5% of Eddington through the post-starburst phase (the M_h–M_* relation
applies today), and soft window edges (per-nucleus scatter of 0.12 dex on the floor and 0.25 dex
on the ceiling). Each variant in `scripts/popsynth.py` changes one ingredient; `simple` drops the
first three.
