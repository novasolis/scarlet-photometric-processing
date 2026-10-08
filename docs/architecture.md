# Photometric Analysis Pipeline

## Software Architecture

### 1. Overview

This project implements a comparative photometric analysis pipeline for astronomical imagery. It is designed to evaluate the effect of SCARLET source deblending on downstream photometric measurements by applying a consistent analysis workflow to astronomical image data.

The architecture separates data acquisition, image preparation, source handling, photometric measurement, workflow orchestration, evaluation, and result export. This keeps the scientific processing stages independently testable while ensuring that Baseline and SCARLET workflows can be compared using the same experimental configuration and evaluation framework.

The implementation is organised around explicit data objects and one-directional data flow:

```text
Experiment Configuration
          │
          ▼
    Data Acquisition
          │
          ▼
    Image Preparation
          │
          ▼
     Source / Target
       Preparation
          │
          ▼
   ┌──────┴──────┐
   │             │
   ▼             ▼
Baseline      SCARLET
Workflow      Workflow
   │             │
   └──────┬──────┘
          │
          ▼
 Comparative Evaluation
          │
          ▼
      Results Export
```

The architecture is implementation-oriented rather than tied to a particular dataset. Dataset-specific acquisition and metadata handling are isolated from the processing and evaluation layers.

---

## 2. Architectural Principles

The project follows these principles:

- **Separation of responsibilities** — each module owns a distinct stage or concern.
- **Explicit data flow** — processing stages receive defined inputs and produce defined outputs.
- **Configuration-driven experiments** — experimental parameters are represented centrally rather than embedded throughout the pipeline.
- **Comparable workflows** — Baseline and SCARLET processing operate on equivalent inputs and produce results suitable for direct comparison.
- **Reproducibility** — dataset provenance, configuration, processing choices, and generated results should be traceable.
- **Testability** — scientific logic and workflow behaviour should be testable independently of external data acquisition where practical.
- **Minimal coupling** — higher-level workflow and evaluation code should depend on stable project abstractions rather than dataset-specific implementation details.

---

## 3. Pipeline Layers

### 3.1 Configuration

The `config` package defines the parameters that control an experiment, including the observational scenario, source scope, processing pipeline, dataset information, and relevant photometric parameters.

Configuration should be treated as input to the experiment rather than being modified by processing stages.

### 3.2 Data Acquisition

The `acquisition` package obtains and loads astronomical observations from supported sources such as TESS and SDSS.

Its responsibility ends with producing project-compatible image and metadata representations. Dataset-specific download logic should not be required by downstream photometric processing.

### 3.3 Data and Domain Model

The `data`, `domain`, `core`, and `target` packages provide the internal representations and scientific abstractions used throughout the pipeline.

These layers separate:

- astronomical image and metadata representation;
- observational and experimental concepts;
- target and reference-source information;
- shared application-level data structures.

The domain model should remain independent of a particular acquisition mechanism or processing implementation wherever possible.

### 3.4 Detection and Preparation

The `detection` and `preprocessing` packages prepare observations for photometric processing.

Typical responsibilities include:

- image validation;
- dimensionality and data-quality checks;
- background handling;
- source detection or source-related preparation;
- conversion between external image representations and project representations;
- preparation of inputs required by downstream processing.

Preprocessing must preserve the information required to perform a meaningful comparison between workflows.

### 3.5 Photometry

The `photometry` package contains the photometric measurement layer.

It is responsible for converting processed astronomical images and source information into photometric measurements. The measurement logic should be independent of whether the image originated from the Baseline or SCARLET branch whenever the same measurement method is intended for both.

### 3.6 Workflows

The `workflows` package orchestrates complete processing paths.

The principal comparison is:

```text
Prepared Observation
       │
       ├──────────────► Baseline Workflow
       │                       │
       │                       ▼
       │                 Photometry Result
       │
       └──────────────► SCARLET Workflow
                               │
                               ▼
                         Photometry Result
```

The workflow layer coordinates components but should not duplicate low-level image-processing or photometric algorithms.

The SCARLET workflow treats SCARLET as a processing component rather than as the overall application architecture. The project therefore remains responsible for input preparation, target/source handling, photometric measurement, comparison, and result management.

### 3.7 Evaluation

The `evaluation` package compares outputs from the processing workflows.

Evaluation is deliberately downstream of photometric measurement so that the comparison is based on equivalent result objects rather than implementation-specific intermediate representations.

Where applicable, evaluation may include measurements such as:

- flux differences;
- relative differences;
- uncertainty-related comparisons;
- consistency or agreement between workflows;
- experiment-level summary statistics.

The exact metrics are determined by the experimental configuration and study methodology.

### 3.8 Export

The `export` package converts experiment outputs into persistent artefacts such as:

- tabular results;
- figures;
- summaries;
- machine-readable result files.

Export code should not alter scientific results. It is responsible for presentation and persistence only.

### 3.9 Storage

The `storage` package provides persistence-related abstractions used by the application. Storage concerns should remain separate from scientific processing so that processing code does not depend directly on a particular filesystem or persistence implementation.

### 3.10 Utilities and Shared Services

The `utils` package contains small, reusable helper functionality that does not belong to a specific scientific or workflow layer.

Utilities should remain general enough to avoid becoming an unstructured dependency layer.

---

## 4. Core Data Objects

The architecture uses explicit objects to move information between stages. The principal conceptual objects are:

| Object             | Purpose                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| `ExperimentConfig` | Defines experimental conditions, dataset selection, processing options, and analysis parameters. |
| `AstroImage`       | Represents an astronomical image together with the metadata required by downstream processing.   |
| `PhotometryResult` | Represents photometric measurements produced by a processing workflow.                           |
| `ComparisonResult` | Represents evaluation metrics comparing workflow outputs.                                        |

Additional domain objects may represent observations, targets, detected sources, provenance, and experiment-specific configuration.

These objects should provide stable boundaries between modules and prevent individual stages from passing loosely structured collections of unrelated values.

---

## 5. Workflow Comparison Model

The central architectural requirement is that Baseline and SCARLET results remain scientifically comparable.

The intended structure is:

```text
                 Same ExperimentConfig
                         │
                         ▼
                  Prepared Observation
                    /                                /                                 ▼                 ▼
             Baseline            SCARLET
             Processing          Processing
                  │                 │
                  ▼                 ▼
          Photometric Result  Photometric Result
                  \                 /
                   \               /
                    ▼             ▼
                    Comparative Evaluation
                             │
                             ▼
                       ComparisonResult
```

The comparison should therefore change the **processing path**, not unrelated experimental conditions.

Dataset, target definition, coordinate information, photometric measurement method, and evaluation criteria should remain controlled according to the experiment configuration.

---

## 6. SCARLET Integration

SCARLET is integrated as an optional source-separation/deblending component within the preprocessing/workflow layer.

The project uses the local SCARLET 1.0.1 source checkout because the native `operators_pybind11` extension is required and the available source was adapted for compatibility with the installed SciPy release.

The integration has the following boundary:

```text
Project Observation
        │
        ▼
SCARLET input preparation
        │
        ▼
SCARLET source separation
        │
        ▼
Recovered/deblended image or sources
        │
        ▼
Project photometric measurement
        │
        ▼
SCARLET PhotometryResult
```

SCARLET-specific implementation details should remain isolated from the rest of the application. This allows the Baseline workflow and the common photometric/evaluation layers to remain independent of the SCARLET implementation.

The SCARLET installation and local native-extension build procedure are documented in the repository `README.md`.

---

## 7. Project Structure

The main application structure is:

```text
photometry/
├── data/
├── docs/
│   ├── architecture.md
│   └── development_log.md
├── logs/
├── results/
├── scarlet-source/
├── src/
│   ├── acquisition/
│   ├── analysis/
│   ├── config/
│   ├── core/
│   ├── data/
│   ├── demos/
│   ├── detection/
│   ├── domain/
│   ├── evaluation/
│   ├── export/
│   ├── photometry/
│   ├── preprocessing/
│   ├── storage/
│   ├── target/
│   ├── utils/
│   └── workflows/
├── README.md
├── requirements.txt
└── workflow_diagram.md
```

### Directory responsibilities

| Directory            | Responsibility                                                         |
| -------------------- | ---------------------------------------------------------------------- |
| `data/`              | Project data and data-related artefacts.                               |
| `docs/`              | Architecture and development documentation.                            |
| `logs/`              | Development and execution logs where retained.                         |
| `results/`           | Generated experiment and analysis results.                             |
| `scarlet-source/`    | Local SCARLET source checkout and native extension.                    |
| `src/acquisition/`   | Astronomical dataset acquisition and loading.                          |
| `src/analysis/`      | Higher-level analysis functionality.                                   |
| `src/config/`        | Experiment configuration and configuration handling.                   |
| `src/core/`          | Shared application/core abstractions.                                  |
| `src/data/`          | Internal data representations and data handling.                       |
| `src/demos/`         | Demonstration and exploratory entry points.                            |
| `src/detection/`     | Source detection and related processing.                               |
| `src/domain/`        | Scientific/domain model objects.                                       |
| `src/evaluation/`    | Comparative evaluation and metrics.                                    |
| `src/export/`        | Result, table, figure, and report export.                              |
| `src/photometry/`    | Photometric measurement functionality.                                 |
| `src/preprocessing/` | Image validation and preparation, including SCARLET input preparation. |
| `src/storage/`       | Persistence and storage abstractions.                                  |
| `src/target/`        | Target-source and reference-source handling.                           |
| `src/utils/`         | General-purpose shared utilities.                                      |
| `src/workflows/`     | End-to-end experiment workflow orchestration.                          |

The repository also contains development and test artefacts that are intentionally not part of the architectural runtime structure.

---

## 8. Dependency Direction

The preferred dependency direction is from higher-level orchestration towards lower-level implementation:

```text
Workflows
    │
    ├── Evaluation
    ├── Photometry
    ├── Preprocessing
    ├── Detection
    ├── Target
    └── Acquisition
             │
             ▼
        Data / Domain / Core
             │
             ▼
           Utils
```

Cross-layer dependencies should be kept deliberate. In particular:

- acquisition code should not depend on evaluation;
- preprocessing should not depend on export;
- photometry should not depend on workflow orchestration;
- evaluation should consume results rather than reach back into processing internals;
- export should consume completed results rather than perform scientific processing.

---

## 9. Reproducibility and Provenance

Each experiment should be reproducible from its configuration, source data, processing implementation, and recorded outputs.

Where applicable, provenance should capture:

- observation/source information;
- survey and dataset identifier;
- target coordinates;
- processing pipeline;
- experiment type and source scope;
- relevant processing parameters;
- software/dependency versions;
- generated result location.

The repository should distinguish between source data, intermediate processing products, and final exported results.

---

## 10. Error Handling

Errors crossing module boundaries should be expressed using project-level exceptions where appropriate.

Low-level implementation errors may be caught and wrapped when doing so adds meaningful context for the caller. Error messages should identify:

1. the operation that failed;
2. the relevant object or processing stage;
3. the underlying cause where useful.

Validation failures should occur as early as practical, particularly for missing metadata, invalid image dimensions, incompatible inputs, and unsupported processing configurations.

---

## 11. Testing Architecture

Tests are organised separately from the runtime source tree and are intended to verify the system at multiple levels:

- configuration validation;
- acquisition and data handling;
- preprocessing;
- workflow behaviour;
- photometric calculations;
- evaluation;
- export;
- SCARLET integration boundaries.

Unit tests should prefer deterministic synthetic inputs where practical. Integration tests may exercise real package integrations or representative astronomical data where required.

The architecture should support testing individual stages without requiring a complete end-to-end experiment for every change.

---

## 12. Scientific Reproducibility Boundary

The architecture distinguishes **scientific computation** from **experiment orchestration**.

Scientific computation includes image preparation, source handling, photometric measurement, deblending, and metric calculation.

Experiment orchestration includes selecting configuration, acquiring observations, invoking workflows, collecting outputs, and exporting results.

This separation allows the experimental conditions to remain controlled while individual processing components can be changed, tested, and evaluated independently.
