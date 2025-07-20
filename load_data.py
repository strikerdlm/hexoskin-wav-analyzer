"""
Top-level wrapper for Valquiria data-loading utilities.

Having this thin wrapper in the project root means that scripts and notebooks can simply do

    from load_data import load_database_data, load_csv_data

without worrying about the current working directory or the full package path.  All actual
logic lives in `Data.joined_data.scripts.load_data` – we just re-export the public functions
here for convenience.
"""

from joined_data.scripts.load_data import (
    load_csv_data,
    load_database_data,
    get_data_summary,
    print_data_summary,
    load_sample_data,
    check_data_quality,
    get_subject_data,
)

__all__ = [
    "load_csv_data",
    "load_database_data",
    "get_data_summary",
    "print_data_summary",
    "load_sample_data",
    "check_data_quality",
    "get_subject_data",
] 