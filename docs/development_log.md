**# Development Log**

**## Purpose**

This document records significant implementation decisions, compatibility issues, troubleshooting steps, and development milestones for the Photometric Analysis Pipeline.

It is intended to preserve the reasoning behind the current implementation without duplicating the formal architecture described in [\`architecture.md\`]\(architecture.md).

\---

**## 1. Project Foundation**

The project was established as a modular Python pipeline for comparative astronomical photometry.

The primary scientific objective is to evaluate the effect of SCARLET source deblending on photometric measurements by comparing:

\- a **\*\*Baseline\*\*** photometric workflow; and

\- a **\*\*SCARLET\*\***-processed workflow.

The architecture was deliberately separated into acquisition, preprocessing, detection, target handling, photometry, workflow orchestration, evaluation, and export layers.

The implementation uses explicit domain/configuration concepts rather than passing loosely structured parameters between stages.

\---

**## 2. Experimental Configuration Model**

The configuration model was expanded to represent the scientific scenarios used by the study.

The following enumerations were established:

\`\`\`python

**class** ObservationSource(Enum):

    SPACE_BASED = "Space-based"

    GROUND_BASED = "Ground-based"

**class** Survey(Enum):

    TESS = "TESS"

    SDSS = "SDSS"

**class** ProcessingPipeline(Enum):

    BASELINE = "Baseline"

    SCARLET = "SCARLET"

**class** ExperimentType(Enum):

    TARGET_SOURCE = "Target-source photometry"

    SOURCE_FIELD = "Source-field photometry"

    CROWDED_FIELD = "Crowded-field photometry"

    DIFFERENTIAL_PHOTOMETRY = "Differential photometry"

**class** SourceScope(Enum):

    TARGET = "Target source"

    TARGET_AND_REFERENCE = "Target and reference stars"

    FIELD = "Entire detected field"

\`\`\`

These values provide controlled vocabulary for experiment configuration and help prevent inconsistent string values from propagating through the pipeline.

\---

**## 3. Observation Model**

The observation representation was designed to keep astronomical image data together with the metadata required for processing and provenance.

The observation metadata includes, where applicable:

\- target name;

\- right ascension;

\- declination;

\- World Coordinate System (WCS);

\- observational source;

\- survey;

\- dataset identifier;

\- source filename;

\- download timestamp.

An observation also records provenance information separately from the image itself.

This distinction allows the processing pipeline to retain information about where an observation originated without coupling scientific processing code to a particular acquisition mechanism.

\---

**## 4. Initial SCARLET Integration**

SCARLET was selected as the source-separation/deblending component for the comparative workflow.

The project uses **\*\*SCARLET 1.0.1\*\*** from a local source checkout rather than relying solely on a pre-built package installation.

The local checkout is retained under:

\`\`\`text

scarlet-source/

\`\`\`

The native extension required by SCARLET is:

\`\`\`text

scarlet.operators_pybind11

\`\`\`

The integration was designed so that SCARLET remains a processing component within the project rather than becoming the application's architectural dependency.

\---

**## 5. SCARLET Compatibility Issue: SciPy**

The initial SCARLET import failed because the source used:

\`\`\`python

from scipy.stats import median_absolute_deviation

\`\`\`

The installed SciPy release provides:

\`\`\`python

scipy.stats.median_abs_deviation

\`\`\`

but not \`median_absolute_deviation\`.

The SCARLET source was therefore adapted by replacing the deprecated/removed import with:

\`\`\`python

from scipy.stats import median_abs_deviation as mad

\`\`\`

After this change, the SCARLET package imported successfully.

The modification is intentionally retained as part of the local SCARLET source checkout and is documented in the repository README for reproducibility.

\---

**## 6. SCARLET Native Extension Build**

SCARLET requires compilation of its \`operators_pybind11\` native extension.

The first installation attempt failed with:

\`\`\`text

Microsoft Visual C++ 14.0 or greater is required.

\`\`\`

The Windows environment did not initially have the Microsoft C++ compiler available from the active shell.

Microsoft Visual Studio Build Tools were subsequently installed with the **\*\*Desktop development with C++\*\*** workload.

The relevant x64 compiler environment was verified using:

\`\`\`text

vcvars64.bat

\`\`\`

and:

\`\`\`text

where cl

\`\`\`

The x64 compiler was then confirmed with:

\`\`\`text

cl

\`\`\`

which reported the Microsoft C/C++ compiler for x64.

The native SCARLET extension was subsequently built successfully.

Verification:

\`\`\`text

python -c "import scarlet.operators_pybind11; print('SCARLET native extension: OK')"

\`\`\`

returned:

\`\`\`text

SCARLET native extension: OK

\`\`\`

\---

**## 7. Python Environment**

The project is currently developed in a Windows 64-bit Python virtual environment using:

\`\`\`text

Python 3.14.5

\`\`\`

The environment is:

\`\`\`text

.venv/

\`\`\`

The interpreter architecture was verified as:

\`\`\`text

AMD64

64-bit Windows

\`\`\`

The SCARLET native extension was successfully compiled for this environment.

Because SCARLET contains native code and is based on an older codebase, its compatibility with future Python, NumPy, and SciPy releases should be treated as an explicit reproducibility concern.

\---

**## 8. SCARLET Installation Troubleshooting**

An initial editable installation attempt was made with:

\`\`\`text

python -m pip install -e .

\`\`\`

The build reached SCARLET's native extension compilation stage but failed because the Microsoft C++ compiler was unavailable.

After installing the required Build Tools and configuring the x64 compiler environment, installation was retried.

A subsequent attempt using:

\`\`\`text

python -m pip install -e . --no-build-isolation

\`\`\`

encountered:

\`\`\`text

BackendUnavailable: Cannot import 'setuptools.build_meta'

\`\`\`

This was a build-environment issue rather than a SCARLET source-code failure.

The environment was then corrected and the installation/build process completed sufficiently for SCARLET to become importable and for the native extension to load.

The important reproducibility requirements are therefore:

1\. use the project's virtual environment;

2\. use a compatible Python architecture;

3\. have the Microsoft C++ Build Tools installed;

4\. initialise the x64 Visual C++ environment before building when required;

5\. ensure the SCARLET SciPy compatibility modification is present;

6\. verify both \`import scarlet\` and \`import scarlet.operators_pybind11\`.

The detailed installation procedure is maintained in the repository README.

\---

**## 9. SCARLET Version Locked**

The SCARLET implementation used for the study is currently locked to:

\`\`\`text

SCARLET 1.0.1

\`\`\`

The working installation was verified with:

\`\`\`text

python -c "import scarlet; print(scarlet.\_\_file\_\_); print(getattr(scarlet, '\_\_version\_\_', 'NO VERSION'))"

\`\`\`

The package reported:

\`\`\`text

1.0.1

\`\`\`

A Python syntax warning is emitted by the original SCARLET source for an existing expression using \`is\` with a string literal:

\`\`\`text

SyntaxWarning: "is" with 'str' literal. Did you mean "=="?

\`\`\`

This warning is not currently treated as a blocking issue because SCARLET imports and its native extension loads successfully.

The warning should not be silently interpreted as a project defect; it originates from the legacy SCARLET source.

\---

**## 10. SCARLET Integration Tests**

Dedicated SCARLET preprocessing tests were added to validate the project's integration boundary.

The test suite covers:

\- requirement for background subtraction;

\- rejection of missing target right ascension;

\- rejection of missing target declination;

\- requirement for a two-dimensional image;

\- rejection of an empty image;

\- acceptance of a single-channel image cube;

\- rejection of multiple channels;

\- rejection of invalid image shapes.

The resulting test run:

\`\`\`text

python -m pytest tests/preprocessing/test_scarlet.py -v

\`\`\`

completed successfully:

\`\`\`text

8 passed

\`\`\`

This established that the SCARLET preprocessing boundary behaves as intended independently of a full scientific SCARLET run.

\---

**## 11. Workflow Error Handling**

Workflow-level error handling was also tested.

The test:

\`\`\`text

tests/workflows/test_photometric_workflow\.py::test_unexpected_workflow_error_is_wrapped

\`\`\`

passed successfully.

This confirms the intended behaviour of wrapping unexpected lower-level failures in the project's workflow-level error abstraction while preserving useful failure context.

\---

**## 12. Full Test Suite and Repository Layout**

A full repository test run was attempted with:

\`\`\`text

python -m pytest -v

\`\`\`

The project tests themselves were not the only tests discovered. Pytest also collected the upstream SCARLET tests located under:

\`\`\`text

scarlet-source/tests/

\`\`\`

This produced nine collection errors.

The errors fell into two categories:

**### SCARLET test import errors**

The upstream SCARLET tests attempted to import \`scarlet\` but were not resolving the local editable package correctly from the root-project test invocation.

**### Missing upstream test dependency**

The SCARLET documentation test required:

\`\`\`text

nbformat

\`\`\`

which was not installed in the project environment.

These errors are distinct from failures in the project's own test suite.

The dedicated project SCARLET tests continued to pass, and the SCARLET package itself was independently verified as importable with its native extension.

The repository therefore needs to keep the upstream SCARLET test tree conceptually separate from the project's own test suite when running validation.

\---

**## 13. Test Execution Convention**

During development, an important distinction was established between:

\- the **\*\*project virtual environment\*\***;

\- the **\*\*Visual Studio Developer Command Prompt\*\*** used to initialise the C++ compiler environment; and

\- the **\*\*project's VS Code terminal\*\*** used for normal Python development and testing.

The normal project test command is run from the project root with the project's \`.venv\` active:

\`\`\`text

python -m pytest -v

\`\`\`

For SCARLET native compilation on Windows, the x64 Visual C++ environment must be initialised before the build when the compiler environment is not already available.

The Python test environment and the C++ build environment are related but are not the same thing.

\---

**## 14. Dependency Baseline**

The project's current direct dependency baseline is:

\`\`\`text

numpy==2.5.1

scipy==1.18.0

matplotlib==3.11.1

pandas==2.3.3

astropy==8.0.1

astroquery==0.4.11

photutils==3.0.0

lightkurve==2.6.0

\`\`\`

These versions are retained as the project's reproducibility baseline.

SCARLET's own dependencies are installed as part of the local SCARLET setup and are not treated as a replacement for the project's primary \`requirements.txt\`.

The environment should not be casually upgraded while the current experimental implementation is being validated. Dependency changes should be tested and recorded as deliberate changes.

\---

**## 15. Current Stable Milestone**

The following milestone has been reached:

\- project architecture established;

\- configuration enums established;

\- observation/provenance model established;

\- SCARLET 1.0.1 source integrated locally;

\- SciPy compatibility modification applied;

\- Microsoft C++ Build Tools configured;

\- SCARLET native extension compiled successfully;

\- SCARLET package import verified;

\- SCARLET preprocessing tests implemented;

\- SCARLET preprocessing tests passing (\`8 passed\`);

\- workflow error-wrapping test passing;

\- project dependency baseline recorded;

\- README installation documentation established;

\- architecture documentation established.

At this point, the SCARLET version and integration environment should be treated as a **\*\*locked experimental baseline\*\***.

\---

**## 16. Next Development Phase**

Further development should focus on the scientific execution of the SCARLET workflow rather than additional infrastructure changes unless a concrete requirement emerges.

The next major work should include:

1\. constructing representative astronomical inputs;

2\. completing the SCARLET processing workflow;

3\. obtaining and validating SCARLET source/deblended outputs;

4\. applying the common photometric measurement stage;

5\. running equivalent Baseline and SCARLET experiments;

6\. calculating the defined comparison metrics;

7\. validating results on the selected observational datasets;

8\. exporting reproducible tables and figures;

9\. documenting experimental results and observations.

Changes to the locked SCARLET version or dependency baseline should only be made when technically necessary and should be recorded here with the reason and validation performed.

\---

**## 17. Documentation Boundary**

The repository documentation is intentionally divided by purpose:

\| Document                  | Purpose                                                                                               |

\| ------------------------- | ----------------------------------------------------------------------------------------------------- |

\| \`README.md\`               | Project overview, setup, installation, execution, and essential reproducibility instructions.         |

\| \`docs/architecture.md\`    | Current software architecture, module responsibilities, data flow, and system boundaries.             |

\| \`docs/development_log.md\` | Historical implementation decisions, troubleshooting, compatibility work, and development milestones. |

\| \`workflow_diagram.md\`     | High-level workflow representation.                                                                   |

This separation keeps the README lean while preserving enough development history to explain why the current implementation and environment look the way they do.

\---

**## 18. Change Log Convention**

Future significant development entries should record:

\- **\*\*Date\*\***

\- **\*\*Change\*\***

\- **\*\*Reason\*\***

\- **\*\*Implementation\*\***

\- **\*\*Validation\*\***

\- **\*\*Impact on reproducibility\*\***, where applicable.

Minor refactoring does not require a detailed entry unless it changes architecture, scientific behaviour, dependencies, or reproducibility.

**## SCARLET NumPy 2.x Compatibility Patch**

**### Date**

2026-09-10

**### Issue**

The real SDSS SCARLET smoke test reached the SCARLET source initialisation

stage but failed because SCARLET 1.0.1 uses deprecated NumPy aliases

\`np.float\` and \`np.int\`, which are unavailable in the project's NumPy 2.x

environment.

**### Resolution**

Applied a minimal compatibility patch to the local SCARLET source:

\- \`np.float\` → \`float\`

\- \`np.int\` → \`int\`

Affected SCARLET modules:

\- \`constraint.py\`

\- \`frame.py\`

\- \`initialization.py\`

\- \`observation.py\`

\- \`wavelet.py\`

Valid NumPy-specific types such as \`np.float32\` and \`np.float64\` were

left unchanged.

**### Rationale**

The patch does not alter the SCARLET algorithm or the project's

photometric workflow. It replaces removed NumPy aliases with their

equivalent built-in Python types so that the existing SCARLET 1.0.1

implementation can execute in the modern project environment.

**### Validation**

The SCARLET preprocessing unit tests passed before the integration test

was rerun.

The real SDSS integration smoke test is used to validate whether the

patched SCARLET implementation can proceed through source modelling.

---

## 19. SCARLET 1.0.1 Modern-Environment Compatibility Audit

### Date

2026-09-11

### Scope

A broader compatibility audit was performed against the locally integrated SCARLET 1.0.1 source under:

```text
scarlet-source/scarlet/
```

The objective was to make the original SCARLET 1.0.1 implementation executable in the project's modern Python/NumPy environment while preserving its scientific implementation and keeping changes as small as possible.

The audit was performed incrementally rather than by applying a broad modernisation of the codebase.

### Environment

The working project environment is:

```text
Python 3.14.5
NumPy 2.5.1
SciPy 1.18.0
```

The SCARLET native extension is compiled for:

```text
Windows x64
Python 3.14
```

and is available as:

```text
scarlet/operators_pybind11.cp314-win_amd64.pyd
```

### Compatibility Changes Applied

#### 19.1 SciPy API compatibility

The original SCARLET source imported:

```python
from scipy.stats import median_absolute_deviation
```

The modern SciPy environment provides the renamed API:

```python
from scipy.stats import median_abs_deviation as mad
```

The SCARLET source was updated accordingly.

This is an API-compatibility change only; the intended median absolute deviation calculation remains unchanged.

#### 19.2 NumPy 2.x removed aliases

The SCARLET source contained removed NumPy aliases including:

```python
np.float
np.int
```

These were replaced with the corresponding built-in Python types:

```python
float
int
```

The affected modules are:

```text
constraint.py
frame.py
initialization.py
observation.py
wavelet.py
```

Explicit NumPy precision types such as:

```python
np.float32
np.float64
```

were intentionally retained.

#### 19.3 Autograd compatibility in component model construction

During execution under the modern NumPy/Autograd stack, an in-place array update in `CombinedComponent.get_model()` caused an incompatibility.

The original operation:

```python
model += model_
```

was changed to the non-in-place equivalent:

```python
model = model + model_
```

This is important because the two expressions are mathematically equivalent for this operation, but the latter does not require an in-place mutation of the array being tracked through the Autograd execution path.

The change is therefore considered a compatibility-preserving implementation change rather than a scientific algorithm change.

An analogous in-place multiplication was reviewed in the same context and was only changed where required by the actual Autograd execution path.

No blanket replacement of all `+=`, `*=`, or other in-place operations was performed.

#### 19.4 NumPy byte-order API compatibility in documentation

The SCARLET documentation test exposed an additional NumPy 2.x compatibility issue in the `multiresolution.ipynb` notebook.

The older form:

```python
data_hst.byteswap().newbyteorder()
```

is no longer supported because `ndarray.newbyteorder()` was removed in NumPy 2.x.

The operation was updated using the modern dtype-based form:

```python
data_hst.byteswap().view(data_hst.dtype.newbyteorder())
```

This change is confined to the documentation/tutorial example and does not modify the SCARLET scientific implementation.

#### 19.5 Documentation test dependency

The upstream documentation tests require Jupyter notebook execution dependencies that were not part of the original project environment.

The required packages were installed as needed, including:

```text
nbformat
nbconvert
```

The `multiresolution.ipynb` documentation test subsequently completed successfully.

### Validation Performed

Compatibility was validated incrementally using the upstream SCARLET tests and targeted tests.

A previously failing multiresolution test:

```text
tests/test_multiresolution.py::TestLowResObservation::test_surveys
```

was repaired and subsequently passed:

```text
1 passed
```

The companion multiresolution test also passed:

```text
tests/test_multiresolution.py::TestLowResObservation::test_padded_frame
```

The documentation test:

```text
tests/test_docs.py::TestDocs::test_docs
```

subsequently passed:

```text
1 passed, 1 warning
```

The remaining warning came from the Windows ZeroMQ/Tornado event-loop integration used by notebook execution:

```text
RuntimeWarning: Proactor event loop does not implement add_reader...
```

This warning is infrastructure-related and is not a SCARLET scientific failure.

The targeted SCARLET constraint test was also verified after the compatibility work:

```text
tests/test_constraint.py::TestUpdate::test_threshold PASSED
```

### `sep` Dependency Decision

The documentation notebook required the external `sep` package for one of its examples.

`sep` was installed so that the upstream documentation test could be executed completely.

It is not currently considered a required dependency of the project's own photometric workflow. The project's planned baseline photometry uses the project's selected detection/photometry components rather than depending on the SCARLET documentation example.

Therefore, `sep` should not be promoted to a core project dependency solely because the upstream SCARLET documentation test imports it.

### Scientific-Safety Assessment

The compatibility work deliberately follows a minimal-change principle.

The changes made so far are limited to:

1. replacing removed dependency APIs;
2. replacing removed NumPy aliases;
3. avoiding an Autograd-incompatible in-place model accumulation;
4. updating an obsolete NumPy byte-order expression in a documentation notebook;
5. installing packages required to execute upstream documentation tests.

No SCARLET source-separation algorithm, optimisation objective, component model, PSF model, morphology model, or photometric calculation has intentionally been redesigned.

The purpose is to preserve the original SCARLET 1.0.1 behaviour while making it executable in the current experimental environment.

### Important Limitation

Passing the upstream unit tests does not by itself establish scientific equivalence with the original historical SCARLET environment.

The definitive project-level validation remains execution on representative astronomical data and comparison of the resulting SCARLET/deblended photometry against the baseline pipeline.

Consequently, SCARLET 1.0.1 remains an experimental, locally patched dependency rather than a generally supported modern release.

---

## 20. Current SCARLET 1.0.1 Compatibility Status

At this stage, the SCARLET 1.0.1 source has passed the compatibility issues encountered during the incremental audit sufficiently to proceed to the project's scientific integration work.

The current approach is:

- retain the original SCARLET 1.0.1 source;
- retain the local compatibility patches;
- avoid unnecessary broad refactoring;
- use targeted upstream tests to identify genuine runtime incompatibilities;
- validate the patched implementation with the project's actual astronomical data;
- record any further compatibility issue only when it is encountered during execution.

The next step is therefore not another broad SCARLET rewrite. It is to continue with the scientific SCARLET workflow and use real project inputs to identify any remaining execution or data-interface issues.

Any additional patch should be documented with the same convention:

- **Date**
- **Issue**
- **Resolution**
- **Rationale**
- **Validation**
- **Scientific/reproducibility impact**
