import os
import sys

# Add repo root into sys.path so local modules can be imported when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline import run_bottom_up_synthesis  # noqa: E402


def main() -> None:
    """Run the full pipeline with horizon=20m."""

    # Run the bottom-up pipeline under the standard AGENTS.md configuration.
    run_bottom_up_synthesis(
        seed=42,
        etf_code_int=510500,
        label_horizon_minutes=20,
        train_start=20210101,
        train_end=20231231,
        test_start=20240201,
        test_end=20251231,
        factor_set_name="stock_all",
        etf_factor_set_name="etf_all",
        basket_variants=[],
    )


if __name__ == "__main__":
    main()

