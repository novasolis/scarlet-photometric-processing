from enum import Enum


class ObservationSource(Enum):
    """Physical origin of an astronomical observation."""

    SPACE_BASED = "Space-based"
    GROUND_BASED = "Ground-based"


class Survey(Enum):
    """Supported astronomical surveys."""

    TESS = "TESS"
    SDSS = "SDSS"


class ProcessingPipeline(Enum):
    """Photometric processing pipeline."""

    BASELINE = "Baseline"
    SCARLET = "SCARLET"


class ExperimentType(Enum):
    """Scientific analysis performed for a target-centred experiment."""

    TARGET_PHOTOMETRY = "Target photometry"
    FIELD_ANALYSIS = "field_analysis"
    DIFFERENTIAL_PHOTOMETRY = "Differential photometry"