import pytest
from datetime import datetime
from src.reader.transforms import transformations as tf

def test_calculate_posting_age_days():
    assert tf.calculate_posting_age_days("2022-01-01", "2022-01-10") == 9
    assert tf.calculate_posting_age_days(datetime(2022, 1, 1), datetime(2022, 1, 10)) == 9
    assert tf.calculate_posting_age_days(None, "2022-01-10") is None
    assert tf.calculate_posting_age_days("invalid", "2022-01-10") is None
    assert tf.calculate_posting_age_days("2022-01-10", "2022-01-01") == 0  # no negative days

def test_normalize_pay_to_hourly():
    assert pytest.approx(tf.normalize_pay_to_hourly(52000, 78000, "year"), 0.01) == 65000 / (52 * 40)
    assert tf.normalize_pay_to_hourly(None, None, "year") is None
    assert tf.normalize_pay_to_hourly(20, None, "hour") == 20
    assert tf.normalize_pay_to_hourly(1000, 2000, "month") == (1500 / (4.33 * 40))
    assert tf.normalize_pay_to_hourly(500, 500, "week") == 500 / 40
    assert tf.normalize_pay_to_hourly(100, 100, "day") == 100 / 8
    assert tf.normalize_pay_to_hourly(100, 100, "unknown") is None

def test_normalize_onet_code():
    assert tf.normalize_onet_code("15-1121") == "151121"
    assert tf.normalize_onet_code("15-1121.00") == "151121.00"
    assert tf.normalize_onet_code("abc") is None
    assert tf.normalize_onet_code(None) is None

def test_is_ghost_job():
    assert tf.is_ghost_job(1) is True
    assert tf.is_ghost_job("true") is True
    assert tf.is_ghost_job("yes") is True
    assert tf.is_ghost_job("no") is False
    assert tf.is_ghost_job(False) is False
    assert tf.is_ghost_job(None) is False

def test_is_remote_job():
    assert tf.is_remote_job(1) is True
    assert tf.is_remote_job("remote") is True
    assert tf.is_remote_job("yes") is True
    assert tf.is_remote_job("no") is False
    assert tf.is_remote_job(False) is False
    assert tf.is_remote_job(None) is False

def test_clean_education_field():
    assert tf.clean_education_field(" Bachelor'S ") == "bachelor's"
    assert tf.clean_education_field(" ") is None
    assert tf.clean_education_field(None) is None

    # Additional tests for Unicode normalization and apostrophe replacement
    assert tf.clean_education_field("Bachelor\u2019s Degree") == "bachelor's degree"  # right single quotation mark
    assert tf.clean_education_field("Bachelor\u2018s Degree") == "bachelor's degree"  # left single quotation mark
    assert tf.clean_education_field("Bachelor\u201Bs Degree") == "bachelor's degree"  # single high-reversed-9 quotation mark
    assert tf.clean_education_field("Bachelor\u2032s Degree") == "bachelor's degree"  # prime
    assert tf.clean_education_field("Bachelor\u02BCs Degree") == "bachelor's degree"  # modifier letter apostrophe
    assert tf.clean_education_field("Bachelor\uFF07s Degree") == "bachelor's degree"  # fullwidth apostrophe

def test_clean_experience_field():
    assert tf.clean_experience_field(" 3 Years ") == "3 years"
    assert tf.clean_experience_field("") is None
    assert tf.clean_experience_field(None) is None

def test_clean_license_field():
    assert tf.clean_license_field(" None ") == "none"
    assert tf.clean_license_field(" ") is None
    assert tf.clean_license_field(None) is None