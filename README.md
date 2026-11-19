# Straight and Manifold Microchannel Cold Plate Crossover

A reduced-order model (ROM) that compares straight microchannel (SMC) and manifold microchannel (MMC) cold plates at matched pumping power, calibrated against CFD. It locates the crossover where the SMC becomes the cooler device, maps that crossover onto a dimensionless regime map, and reproduces the computed figures and Tables 2 to 6 of the thesis below.

> D. Curl, *Thermal Performance Crossover of Straight and Manifold Microchannel Cold Plates at Matched Pumping Power*, M.S. thesis, Mechanical Engineering, University of Arkansas, 2026.

## Quick start

Python 3.10 with the pinned packages in `requirements.txt`, tested on Windows. The figures use Times New Roman, which must be installed for pixel-identical output.

```
pip install -r requirements.txt
python reproduce.py      # generate every figure and table into results/ (about 5 minutes)
python verify.py         # rerun in a clean copy and check each output
```

Figures are not committed; `reproduce.py` generates them. `verify.py` checks each figure against `results/figure_checksums.csv` and each table against the committed CSV.

## Repository layout

| Path | Contents |
|---|---|
| `src/model.py` | Uncalibrated ROM: friction, Nusselt number, fin efficiency, and the SMC and MMC thermal-hydraulic models |
| `src/manifold.py` | End-fed tapered manifold pressure drop |
| `src/model_cal.py` | Calibrated closures and the measured transition bands |
| `src/refit_cal.py` | Refits the calibration constants from the CFD data; `--check` confirms they match `model_cal.py` |
| `src/helpers.py` | CFD data loading, transition band and crossover finder |
| `src/make_tables.py` | Thesis Tables 2 to 6 |
| `figures/` | One script per thesis figure; `style.py` holds the shared figure style |
| `data/cfd/` | CFD sweeps used for calibration and validation |
| `data/literature/` | Published device data digitized for Figs. 1 and 11 |
| `results/` | Tables 2 to 6 and the figure checksums; `reproduce.py` writes the figures here |

## Paper to repository

| Thesis | Content | Script | Output |
|---|---|---|---|
| Fig. 1 | Published SMC and MMC devices by channel Reynolds number and hydraulic diameter | `figures/device_map.py` | `results/fig01_device_map.png` |
| Figs. 2, 4, 5 | CAD renders and CFD unit cells | Not generated | |
| Fig. 3 | SMC transition band from CFD | `figures/transition_band.py` | `results/fig03_transition_band.png` |
| Fig. 6(a) | Baseline thermal resistance versus pumping power, CFD and ROM | `figures/baseline_crossover.py` | `results/fig06a_baseline_crossover.png` |
| Fig. 6(b) | CFD turbulent viscosity contours | Not generated | |
| Fig. 7 | Crossover pumping power across the five geometric sweeps | `figures/geometric_sensitivity.py` | `results/fig07_geometric_sensitivity.png` |
| Fig. 8 | Regime map, measured transition onset | `figures/regime_map.py` | `results/fig08_regime_map.png` |
| Fig. 9 | Regime map, classical transition onset | `figures/classical_onset.py` | `results/fig09_regime_map_classical_onset.png` |
| Fig. 10 | Baseline design swept on the regime map | `figures/regime_map.py` | `results/fig10_device_pair_sweeps.png` |
| Fig. 11 | Published devices on the regime map | `figures/published_devices.py` | `results/fig11_published_devices.png` |
| Fig. B.1 | Crossover with inlet and outlet fitting losses | `figures/fittings.py` | `results/figB1_fittings.png` |
| Fig. C.1 | Scored operating points and the fitted boundary | `figures/regime_map.py` | `results/figC1_operating_points.png` |
| Fig. D.1 | Sensitivity of the regime map to transition onset | `figures/regime_map.py` | `results/figD1_onset_sensitivity.png` |
| Table 1 | Transition band versus aspect ratio and length | Band values stored in `src/model_cal.py` (`MEASURED_BANDS`) | |
| Table 2 | SMC Nusselt number error | `src/make_tables.py` | `results/table02_smc_nusselt_error.csv` |
| Table 3 | MMC thermal calibration | `src/make_tables.py` | `results/table03_mmc_nusselt_calibration.csv` |
| Table 4 | SMC pressure drop error | `src/make_tables.py` | `results/table04_smc_pressure_drop_error.csv` |
| Table 5 | MMC hydraulic calibration | `src/make_tables.py` | `results/table05_mmc_pressure_drop_calibration.csv` |
| Table 6 | Closure error summary | `src/make_tables.py` | `results/table06_closure_error.csv` |
| Table 7 | CFD mesh study | Not generated | |
| Table 8 | Baseline parameters | `src/model.py` (`Geometry`, `Fluid`) | |
| Table 9 | Baseline crossover, CFD and ROM | `figures/baseline_crossover.py` (crossover pumping powers, annotated on Fig. 6(a)); other entries from `data/cfd/` | |
| Table 10 | Regime map sweep levels | `figures/regime_map.py` (`LS`, `WS`, `DHS`, `LSEG`, `ARMS_MM`) | |
| Table A.1 | Rectangular duct friction constant and Nusselt number | `src/model.py` (`_ALPHA`, `_FRE`, `_NU_H1`) | |
| Table A.2 | Developing laminar friction constants | `src/model.py` (`_PH_AR`, `_PH_C`) | |
| Section 3 | ROM and its calibration | `src/model.py`, `src/manifold.py`, `src/model_cal.py`, `src/refit_cal.py` | |

Fig. 1 regenerates with the "This work" box a few pixels from its position in the thesis; the published-device data are identical.

## Data

`data/cfd/` holds steady CFD results (ANSYS Fluent, transition SST, 5% inlet turbulence intensity) for single unit cells, scaled to the full 50 mm by 50 mm plate. Each row is one flow rate: device flow rate `Q_lpm` (L/min), thermal resistance `Rth_KW` (K/W), pressure drop `dP_Pa` (Pa), and the solver heat balance and wall y+.

| File | Device |
|---|---|
| `smc_transition_sst_medium.csv` | SMC baseline: D_h 1.037 mm, aspect ratio 0.35, L/D_h 48.2 |
| `mmc_endfed_sst_medium.csv` | MMC baseline: end-fed tapered manifold, 10 arms |
| `sweeps/alpha_smc_a020.csv`, `sweeps/alpha_smc_a070.csv` | SMC at aspect ratio 0.20 and 0.70 |
| `sweeps/alpha_smc_a070_highre.csv` | SMC at aspect ratio 0.70, higher Reynolds numbers |
| `sweeps/dh_smc_large.csv`, `sweeps/dh_smc_small.csv` | SMC at D_h 2.074 and 0.519 mm |
| `sweeps/dh_mmc_endfed_large.csv`, `sweeps/dh_mmc_endfed_small.csv` | MMC at D_h 2.074 and 0.519 mm |
| `sweeps/nseg_mmc_endfed_n20.csv`, `sweeps/nseg_mmc_endfed_n5.csv` | MMC with 20 and 5 arms |

`data/literature/` holds published data used in Figs. 1 and 11: the device list in `arch_devices.csv`, and data digitized from Escher et al. (2010), Collins et al. (2019), van Erp et al. (2020), and published equal-pumping-power comparisons.

## License

MIT. See `LICENSE`.
