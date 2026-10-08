# SCARLET Photometric Processing

Experimental framework for evaluating how image preprocessing affects astronomical photometric measurements across terrestrial and space-based imagery.

This repository accompanies a Master of Data Science thesis on **photometric processing enhancements for terrestrially acquired and space-based astronomical imagery**.

## Overview

The study compares classical aperture-photometry measurements from a controlled baseline workflow with measurements obtained after **SCARLET-based source-separation preprocessing**.

The central question is not simply whether preprocessing changes an astronomical image, but:

> **How do image-level changes propagate into photometric measurements?**

The experimental framework was applied to astronomical imagery from:

- **SDSS** — terrestrial/ground-based astronomical imaging
- **TESS** — space-based observations

The repository contains the experimental source code and exported results used in the analysis presented in the thesis.

## Experimental workflow

```text
                    Astronomical image
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Baseline workflow        SCARLET preprocessing
              │                         │
              ▼                         ▼
     Aperture photometry         Aperture photometry
              │                         │
              └────────────┬────────────┘
                           ▼
                  Comparative evaluation
```

The baseline and SCARLET branches are evaluated under controlled photometric conditions so that differences in the resulting measurements can be attributed primarily to the preprocessing path.

The analysis considers changes in photometric measurements, variability and correspondence between baseline and SCARLET measurements across different observational conditions.

## Repository contents

```text
scarlet-photometric-processing/
│
├── docs/                         # Project documentation
├── results/                      # Experimental result exports and outputs
├── scarlet-source/               # Local SCARLET 1.0.1 source used in the study
├── src/                          # Research and experimental source code
├── tests/                        # Project tests
├── validation/                   # Validation and compatibility checks
├── requirements.txt              # Python dependencies
├── setup.cfg                     # Project configuration
├── workflow-diagram.md           # Experimental workflow reference
└── README.md
```

### Experimental results

The `results/` directory contains the exported experimental outputs that form the basis of the analysis reported in the thesis.

These outputs are included so that the reported experimental results can be inspected independently of the original local execution environment.

Raw SDSS and TESS FITS data are **not included** in this repository. The local experiment environment used cached astronomical data separately from the published source and results.

## Software environment

The experimental environment was developed and tested on:

- Windows x64
- Python 3.14
- NumPy
- SciPy
- Astropy
- Astroquery
- Photutils
- Lightkurve
- Matplotlib
- Pandas
- SCARLET 1.0.1

The Python dependency versions used for the experiment are recorded in `requirements.txt`.

## SCARLET

The experiment uses **SCARLET 1.0.1** as the source-separation preprocessing stage.

SCARLET is retained as a local source checkout under:

```text
scarlet-source/
```

The local copy contains targeted compatibility modifications required to execute SCARLET 1.0.1 in the experimental Python/NumPy/SciPy environment. The SCARLET source itself has not been broadly rewritten.

The native SCARLET extension requires a working Microsoft C++ build environment on Windows.

## Getting started

Create a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

### SCARLET native extension

SCARLET 1.0.1 includes a native extension. Building it requires the Microsoft C++ Build Tools with the **Desktop development with C++** workload.

From a Visual Studio Developer Command Prompt with the project virtual environment activated:

```cmd
cd scarlet-source
python -m pip install pybind11
python setup.py build_ext --inplace
```

Then verify:

```cmd
python -c "import scarlet; print(scarlet.__file__); print(getattr(scarlet, '__version__', 'NO VERSION'))"
```

Expected version:

```text
1.0.1
```

## Running tests

Project tests can be run from the repository root:

```powershell
python -m pytest tests -v
```

SCARLET-specific preprocessing tests:

```powershell
python -m pytest tests/preprocessing/test_scarlet.py -v
```

The tests provide software and workflow validation; they are not a substitute for the scientific comparison performed on the astronomical datasets.

## Reproducibility

The published repository provides the source code and experimental outputs used in the thesis. Reproducing the full experiment also depends on the execution environment and the underlying astronomical input data.

For reproducible execution, retain:

- the Python version and architecture;
- the dependency versions in `requirements.txt`;
- the SCARLET 1.0.1 source and compatibility modifications;
- the experiment configuration;
- the relevant astronomical input data and their provenance;
- the exported experimental results.

Raw astronomical datasets are intentionally excluded because of their size and because the experiment obtained them through the relevant astronomical data services.

## Scope of this repository

This repository is a **research snapshot**, published to accompany the thesis and provide access to the experimental setup and results underlying the reported analysis.

It is not intended to represent a finished general-purpose photometry package.

The current code reflects the experimental implementation used during the study. A later development stage may simplify the source tree, consolidate experiment configuration, improve documentation and provide a more streamlined workflow for reuse.

## Scientific focus

The study investigates the **photometric consequences of image preprocessing**, with particular emphasis on whether changes introduced by source separation are reflected in:

- measured photometric flux;
- measurement variability;
- correspondence between baseline and SCARLET measurements;
- preservation of temporal signal structure.

The results therefore distinguish between **image-level change** and **photometric improvement**. A change in measured flux does not, by itself, establish improved photometric accuracy.

## Thesis context

This repository supports a Master of Data Science thesis investigating:

**Photometric processing enhancements to terrestrially acquired and space-based astronomical imagery**

The experimental framework provides a controlled comparison between baseline aperture photometry and SCARLET-enhanced preprocessing across SDSS and TESS observations.

## Status

**Current status:** Experimental thesis repository / research snapshot

The repository currently prioritises preservation of the experimental setup and results used in the thesis. Code cleanup and broader packaging are planned as a separate development stage.
