Photometric validation blueprint

The analysis should compare classical aperture photometry on the original image against aperture photometry after SCARLET preprocessing, using the same targets, apertures, and measurement procedure.

1. Photometric precision — primary metric

Goal: Does SCARLET make repeated/related measurements more stable?

Use:

Standard deviation of flux
Coefficient of variation (CV)
$$ CV = \frac{\sigma_F}{\bar F} $$

Lower CV → greater relative photometric stability.

For each target:

Metric Baseline SCARLET
Mean flux \(\bar F\) \(\bar F*S\)
Flux SD \(\sigma_F\) \(\sigma*{F,S}\)
CV \(CV\) \(CV_S\)

Primary comparison:

$$ \Delta CV = CV_S-CV $$

and percentage change:

$$ \%\Delta CV = \frac{CV_S-CV}{CV}\times100 $$

Interpretation: negative values indicate reduced relative variability after preprocessing.

2. Background contamination / local background stability

Goal: Determine whether SCARLET reduces contamination from neighbouring sources and background structure.

Measure the local background using the same annulus before and after preprocessing.

Record:

median background
background standard deviation
background RMS
background-subtracted flux

A useful metric is:

$$ \Delta B = \sigma\_{B,S}-\sigma_B $$

Lower background scatter indicates a more stable local background.

This is particularly important for crowded SDSS fields such as M67 and NGC 6791.

3. Aperture-radius stability

This should be one of our strongest diagnostics.

Measure the target using several apertures, for example:

$$ r = 2,3,4,5,6,7,8\ {\rm pixels} $$

Then examine how rapidly the measured flux changes with aperture radius.

For a stable isolated source, the curve should approach a plateau.

Calculate a normalized aperture sensitivity such as:

$$ S_r = \frac{|F(r+\Delta r)-F(r)|}{F(r)} $$

Lower sensitivity means the photometric measurement is less dependent on the exact aperture choice.

Compare:

Original → SCARLET

This directly tests whether SCARLET makes aperture photometry less sensitive to contaminating neighbouring flux.

4. Crowding sensitivity

For targets classified as valid vs crowded, compare the SCARLET effect separately.

For example:

Population Baseline CV SCARLET CV ΔCV
Valid ... ... ...
Crowded ... ... ...

This is important because SCARLET's scientific motivation is particularly relevant to source blending/deblending.

We should not simply pool everything together.

5. Flux preservation

This is a diagnostic, not an accuracy test.

Calculate:

$$ R_F = \frac{F_S}{F} $$

and

$$ \Delta F\_\% = \frac{F_S-F}{F}\times100 $$

We can examine whether SCARLET systematically changes measured flux.

For example:

median flux ratio
MAD/IQR of flux ratio
distribution of percentage changes

This tells us whether SCARLET introduces a systematic photometric shift.

Importantly, we should not interpret \(F_S/F\neq1\) as SCARLET being inaccurate, because we don't have ground truth.

6. S/N or measurement uncertainty

Where the pipeline provides uncertainty estimates, compare:

$$ S/N = \frac{F}{\sigma_F} $$

or equivalent photometric uncertainty.

Potential outcomes:

higher S/N
lower uncertainty
unchanged uncertainty

Again, this is evidence of measurement quality/precision, not absolute accuracy.

7. Inter-target consistency

For clusters such as M67 and NGC 6791, examine whether SCARLET produces more consistent photometry across stars with similar observational characteristics.

Useful plots:

baseline flux vs SCARLET flux
percentage flux change vs baseline brightness
CV vs magnitude
aperture sensitivity vs magnitude
flux ratio vs crowding classification

This can reveal whether any SCARLET benefit is concentrated among faint or crowded sources.

Recommended analysis hierarchy

I would structure the eventual analysis layer like this:

Tier 1 — Primary evidence

Photometric precision

CV
flux scatter
uncertainty
repeatability where repeated observations exist
Tier 2 — Mechanistic evidence

Why might precision change?

background RMS
aperture-radius sensitivity
crowding/deblending
local contamination
source isolation
Tier 3 — Integrity checks

Did preprocessing substantially alter the measurement?

flux ratio
percentage flux change
flux distributions
systematic trends with brightness
Tier 4 — Visual diagnostics

Figures showing:

original image
SCARLET-preprocessed image
target/deblended component
aperture
neighbouring sources
residual/background structure

These support the numerical findings but aren't themselves statistical evidence.

The thesis-level conclusion we can legitimately make

The central question should therefore become something like:

Does SCARLET preprocessing improve the precision and robustness of classical aperture photometry under crowded and contaminated observational conditions?

Rather than:

Does SCARLET produce more accurate photometry?

Without ground truth, we cannot establish absolute accuracy. But we can establish evidence for improvements in precision, stability, aperture robustness, background stability, and resistance to crowding.

That is actually a coherent experimental design for your thesis because your baseline is well defined:

Raw SDSS/TESS image → classical aperture photometry

versus

SCARLET-preprocessed image → same classical aperture photometry

with the photometry algorithm held constant.

That makes the effect being tested specifically the SCARLET preprocessing step, rather than changing the entire photometric methodology.

Analysis-layer blueprint
PRODUCTION PHOTOMETRY
│
┌─────────────┴─────────────┐
│ │
BASELINE SCARLET
photometry photometry
│ │
└─────────────┬─────────────┘
│
COMPARISON LAYER
│
┌──────────────────┼──────────────────┐
│ │ │
Precision Robustness Integrity
│ │ │
CV Aperture Flux
Scatter sensitivity ratio
Uncertainty Background Δflux
stability
Crowding
│
▼
STATISTICAL SUMMARY
│
┌─────────────┴─────────────┐
│ │
Overall effect Subgroups
│ │
median ΔCV crowded/valid
median Δflux bright/faint
effect distribution SDSS/TESS
│
▼
THESIS CONCLUSION

Core principle: we demonstrate improved measurement properties, not absolute truth. This should be the guiding principle when we build the final analysis and figure-generation layer.
