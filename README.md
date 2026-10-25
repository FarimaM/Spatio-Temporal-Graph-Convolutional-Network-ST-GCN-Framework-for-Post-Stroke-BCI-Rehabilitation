# ST-GCN Framework for Motor-Imagery EEG in Post-Stroke BCI Rehabilitation

A PyTorch + MNE framework that models EEG as a **spatio-temporal graph**: electrodes are nodes, scalp geometry (or data-driven connectivity) defines edges, and temporal convolutions extract sensorimotor mu/beta dynamics. It decodes left- vs right-hand motor imagery, provides interpretability tools (learned edges, electrode attention, saliency, C3/C4 lateralization) and a simulated closed-loop decoder for robotic-orthosis / FES triggering.

> **Scope and honesty note.** The public dataset used here (PhysioNet EEG Motor Movement/Imagery) contains *healthy* participants. This repository is a methods and engineering framework motivated by post-stroke rehabilitation. It does **not** demonstrate clinical benefit or stroke biomarkers. Run the experiments yourself and report your own numbers.

## Features
- MNE preprocessing: name standardisation, 10-05 montage, CAR, 0.5-48 Hz band-pass, epoching, resampling, caching
- Graphs: spatial (Gaussian kernel + kNN), functional (training-set correlation), hybrid
- Model: ST-GCN with learnable edge importance, depthwise temporal convolutions, log band-power pooling, electrode-attention readout
- Protocols: within-subject k-fold and leave-one-subject-out, plus a CSP+LDA baseline
- Interpretation: edge importance, attention/saliency topomaps, mu/beta lateralization index
- Realtime simulation: streaming band-pass, sliding-window decoding, debounced triggering, mock actuator
- Synthetic data generator (no download) and a pytest suite

## Install
```bash
git clone https://github.com/FarimaM/Spatio-Temporal-Graph-Convolutional-Network-ST-GCN-Framework-for-Post-Stroke-BCI-Rehabilitation.git
cd Spatio-Temporal-Graph-Convolutional-Network-ST-GCN-Framework-for-Post-Stroke-BCI-Rehabilitation
pip install -e .[dev]
pytest -q
```

## Quick start
```bash
stgcn-bci demo --out runs/demo                      # synthetic data, no download
stgcn-bci download --subjects 1-10                  # PhysioNet EDF files (~MBs per subject)
stgcn-bci train --protocol within --subjects 1-10 --out runs/within
stgcn-bci train --protocol loso   --subjects 1-20 --out runs/loso --epochs 80
stgcn-bci train --graph hybrid --channels motor --subjects 1-10
stgcn-bci interpret --checkpoint runs/within/within_S001_fold0.pt --subjects 1 --out runs/interp
stgcn-bci realtime  --checkpoint runs/within/within_S001_fold0.pt --subjects 1
```
Configuration lives in `configs/default.yaml`. Subjects 88, 92, 100 and 104 are skipped (irregular recordings). Training on CPU is slow with the full 64-channel, 128 Hz setting; a GPU or `--channels motor` helps.

## Repository layout
```
stgcn_bci/
  config.py  graph.py  train.py  baselines.py  metrics.py  interpret.py  viz.py  realtime.py  cli.py
  data/    channels.py  physionet.py  synthetic.py  dataset.py
  models/  layers.py  stgcn.py
configs/default.yaml   tests/   docs/architecture.md   .github/workflows/ci.yml
```
See `docs/architecture.md` for design decisions and limitations.

## Extending to patient data
Provide epochs as `(n_trials, n_channels, n_samples)` with an `mne.Info` that has a montage, then call `run_protocol`. For lesioned cohorts, flip or align channels by lesioned hemisphere and compute lateralization over ipsilesional vs contralesional electrodes (`interpret.lateralization_index`).

## License
MIT
