from datetime import datetime
from typing import Optional, Union


def calculate_posting_age_days(
    date_acquired: Union[str, datetime, None], date_expired: Union[str, datetime, None]
) -> Optional[int]:
    """
    Calculate the posting age in days as the difference between expired date and acquired date.

    Args:
        date_acquired (Union[str, datetime, None]): The date the job posting was acquired.
        date_expired (Union[str, datetime, None]): The date the job posting expired.

    Returns:
        Optional[int]: Number of days between acquired and expired dates, or None if inputs are invalid.
    """
    if date_acquired is None or date_expired is None:
        return None

    if isinstance(date_acquired, str):
        try:
            date_acquired = datetime.fromisoformat(date_acquired)
        except ValueError:
            return None

    if isinstance(date_expired, str):
        try:
            date_expired = datetime.fromisoformat(date_expired)
        except ValueError:
            return None

    delta = date_expired - date_acquired
    return max(delta.days, 0)


def normalize_pay_to_hourly(
    salary_min: Optional[float],
    salary_max: Optional[float],
    salary_unit: Optional[str],
) -> Optional[float]:
    """
    Normalize salary min/max and unit to an hourly pay rate.
    If both min and max are provided, returns the average hourly rate.

    Args:
        salary_min (Optional[float]): Minimum salary value.
        salary_max (Optional[float]): Maximum salary value.
        salary_unit (Optional[str]): Unit of salary (e.g., 'hour', 'year', 'month', 'week', 'day').

    Returns:
        Optional[float]: Normalized hourly pay rate or None if inputs are invalid.
    """
    if salary_min is None and salary_max is None:
        return None

    # Use average if both min and max are present
    if salary_min is not None and salary_max is not None:
        salary = (salary_min + salary_max) / 2
    else:
        salary = salary_min if salary_min is not None else salary_max

    if salary_unit is None:
        return None

    unit = salary_unit.strip().lower()

    # Conversion factors to hourly pay
    # Assumptions: 40 hours/week, 52 weeks/year, 8 hours/day
    if unit in {"hour", "hourly", "hr", "h"}:
        hourly_pay = salary
    elif unit in {"year", "annual", "annually", "yr", "y"}:
        hourly_pay = salary / (52 * 40)
    elif unit in {"month", "monthly", "mo", "m"}:
        hourly_pay = salary / (4.33 * 40)
    elif unit in {"week", "weekly", "wk", "w"}:
        hourly_pay = salary / 40
    elif unit in {"day", "daily", "d"}:
        hourly_pay = salary / 8
    else:
        # Unknown unit
        return None

    return max(hourly_pay, 0)


def normalize_onet_code(onet_code: Optional[str]) -> Optional[str]:
    """
    Normalize O*NET code by cleaning and standardizing format.
    Expected format: 6-digit numeric string, optionally with decimal point.

    Args:
        onet_code (Optional[str]): Raw O*NET code string.

    Returns:
        Optional[str]: Normalized O*NET code string or None if invalid.
    """
    if onet_code is None:
        return None

    code = onet_code.strip()

    # Remove any non-numeric and non-dot characters
    allowed_chars = set("0123456789.")
    code = "".join(c for c in code if c in allowed_chars)

    # Remove trailing dots
    code = code.rstrip(".")

    # Validate length and format
    # O*NET codes are typically 6 digits, sometimes with a decimal after 2 digits (e.g., 15-1121 or 15-1121.00)
    # Here we simplify to digits only, no dash
    if len(code) == 6 and code.isdigit():
        return code
    elif len(code) > 6 and code.replace(".", "").isdigit():
        return code
    else:
        return None


def is_ghost_job(ghostjob_flag: Optional[Union[str, int, bool]]) -> bool:
    """
    Determine if a job posting is a ghost job based on the ghostjob flag.

    Args:
        ghostjob_flag (Optional[Union[str, int, bool]]): Raw ghost job indicator.

    Returns:
        bool: True if ghost job, False otherwise.
    """
    if ghostjob_flag is None:
        return False

    if isinstance(ghostjob_flag, bool):
        return ghostjob_flag

    if isinstance(ghostjob_flag, int):
        return ghostjob_flag == 1

    if isinstance(ghostjob_flag, str):
        val = ghostjob_flag.strip().lower()
        return val in {"1", "true", "yes", "y"}

    return False


def is_remote_job(remote_flag: Optional[Union[str, int, bool]]) -> bool:
    """
    Determine if a job posting is remote based on a remote flag or description.

    Args:
        remote_flag (Optional[Union[str, int, bool]]): Raw remote job indicator.

    Returns:
        bool: True if remote job, False otherwise.
    """
    if remote_flag is None:
        return False

    if isinstance(remote_flag, bool):
        return remote_flag

    if isinstance(remote_flag, int):
        return remote_flag == 1

    if isinstance(remote_flag, str):
        val = remote_flag.strip().lower()
        return val in {"1", "true", "yes", "remote"}

    return False


def clean_education_field(education: Optional[str]) -> Optional[str]:
    """
    Clean and normalize education requirement field.

    Args:
        education (Optional[str]): Raw education requirement string.

    Returns:
        Optional[str]: Cleaned education string or None if empty.
    """
    if education is None:
        return None

    cleaned = education.strip().lower()
    if cleaned == "":
        return None

    # Additional normalization can be added here if needed
    return cleaned


def clean_experience_field(experience: Optional[str]) -> Optional[str]:
    """
    Clean and normalize experience requirement field.

    Args:
        experience (Optional[str]): Raw experience requirement string.

    Returns:
        Optional[str]: Cleaned experience string or None if empty.
    """
    if experience is None:
        return None

    cleaned = experience.strip().lower()
    if cleaned == "":
        return None

    # Additional normalization can be added here if needed
    return cleaned


def clean_license_field(license_field: Optional[str]) -> Optional[str]:
    """
    Clean and normalize license requirement field.

    Args:
        license_field (Optional[str]): Raw license requirement string.

    Returns:
        Optional[str]: Cleaned license string or None if empty.
    """
    if license_field is None:
        return None

    cleaned = license_field.strip().lower()
    if cleaned == "":
        return None

    # Additional normalization can be added here if needed
    return cleaned