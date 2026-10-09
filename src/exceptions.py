"""Custom exception hierarchy for fraud detection framework."""

import sys
from typing import Optional


class FraudDetectionException(Exception):
    """Base exception class for all fraud detection module errors."""

    def __init__(self, message: str, error_detail: Optional[Exception] = None):
        """Initialize base fraud detection exception with formatted traceback details.

        Args:
            message: Human-readable error message.
            error_detail: Original caught exception instance if available.
        """
        self.original_message = message
        self.error_detail = error_detail

        if error_detail is not None:
            _, _, exc_tb = sys.exc_info()
            if exc_tb is not None:
                line_no = exc_tb.tb_lineno
                file_name = exc_tb.tb_frame.f_code.co_filename
                formatted_message = (
                    f"{message} | Exception: {type(error_detail).__name__}: {str(error_detail)} "
                    f"at [{file_name}:{line_no}]"
                )
            else:
                formatted_message = f"{message} | Exception: {type(error_detail).__name__}: {str(error_detail)}"
        else:
            formatted_message = message

        super().__init__(formatted_message)


class DataLoadError(FraudDetectionException):
    """Raised when raw or processed dataset fails to load from disk."""
    pass


class DataPreprocessingError(FraudDetectionException):
    """Raised when data transformation, imputation, or scaling fails."""
    pass


class DataSplitError(FraudDetectionException):
    """Raised when train/validation/test partitioning configuration is invalid."""
    pass


class ModelTrainingError(FraudDetectionException):
    """Raised when model fitting, tuning, or ensembling encounters an error."""
    pass


class ModelEvaluationError(FraudDetectionException):
    """Raised when scoring, metrics calculation, or diagnostic generation fails."""
    pass


class ModelInferenceError(FraudDetectionException):
    """Raised when transaction scoring or probability estimation fails during inference."""
    pass
