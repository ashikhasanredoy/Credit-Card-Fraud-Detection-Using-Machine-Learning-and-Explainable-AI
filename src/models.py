"""End-to-end model training, tuning, ensembling, evaluation, and explainability pipeline."""

from src.models.pipeline import run_modeling_pipeline

__all__ = ["run_modeling_pipeline"]

if __name__ == "__main__":
    run_modeling_pipeline()
