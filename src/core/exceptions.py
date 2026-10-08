"""
Project-wide exception hierarchy for the Photometric Analysis Pipeline.

All custom exceptions should inherit from PhotometricAnalysisError to
provide a consistent error handling strategy throughout the application.
"""


class PhotometricAnalysisError(Exception):
    """Base class for all project-specific exceptions."""


class ConfigurationError(PhotometricAnalysisError):
    """Raised when experiment or runtime configuration is invalid."""


class AcquisitionError(PhotometricAnalysisError):
    """Raised when astronomical data acquisition fails."""


class ObservationError(PhotometricAnalysisError):
    """Raised when constructing or manipulating Observation objects fails."""


class PreprocessingError(PhotometricAnalysisError):
    """Raised when image preprocessing fails."""


class DetectionError(PhotometricAnalysisError):
    """Raised when source detection fails."""


class PhotometryError(PhotometricAnalysisError):
    """Raised when photometric measurements fail."""


class EvaluationError(PhotometricAnalysisError):
    """Raised when evaluation or comparison of results fails."""


class ExportError(PhotometricAnalysisError):
    """Raised when exporting results or figures fails."""