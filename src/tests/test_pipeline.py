import pytest
from pyspark.sql import SparkSession
from main.job.pipeline import PySparkJob
from main.base import schema

job = PySparkJob()

elg_sample = [
    ("101", "Fizz", "Buzz"),
    ("103", "John", "Sena"),
]

med_sample = [
    ("101", None, 20),
    ("105", None, 60),
    ("103", None, 90)
]


def create_sample(sample, data_schema):
    return job.spark.createDataFrame(data=sample, schema=data_schema)


@pytest.mark.filterwarnings("ignore")
def test_init_spark_session():
    assert isinstance(job.spark, SparkSession), "-- spark session not implemented"


@pytest.mark.filterwarnings("ignore")
def test_filter_medical():
    eligibility = create_sample(elg_sample, schema.eligibility)
    medicals = create_sample(med_sample, schema.medical)

    filtered_medical = job.filter_medical(eligibility, medicals)
    medical_list = sorted(filtered_medical.collect(), key=lambda x: x.memberId)

    assert (2 == len(medical_list))
    assert ("101" == medical_list[0].memberId)
    assert ("103" == medical_list[1].memberId)


@pytest.mark.filterwarnings("ignore")
def test_generate_full_name():
    eligibility = create_sample(elg_sample, schema.eligibility)
    medicals = create_sample(med_sample, schema.medical)

    filtered_medical = job.filter_medical(eligibility, medicals)
    full_name = job.generate_full_name(eligibility, filtered_medical)

    medical_list = sorted(full_name.collect(), key=lambda x: x.memberId)

    assert (2 == len(medical_list))
    assert ("101" == medical_list[0].memberId and "Fizz Buzz" == medical_list[0].fullName)
    assert ("103" == medical_list[1].memberId and "John Sena" == medical_list[1].fullName)


@pytest.mark.filterwarnings("ignore")
def test_find_max_paid_member():
    medicals = create_sample(med_sample, schema.medical)

    max_paid_member = job.find_max_paid_member(medicals)
    assert max_paid_member == "103"


@pytest.mark.filterwarnings("ignore")
def test_find_total_paid_amount():
    medicals = create_sample(med_sample, schema.medical)

    total_paid_amount = job.find_total_paid_amount(medicals)
    assert total_paid_amount == 170
