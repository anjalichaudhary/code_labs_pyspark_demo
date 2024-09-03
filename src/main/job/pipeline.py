from pyspark.sql import SparkSession, DataFrame, Window
from main.base import PySparkJobInterface
import pyspark.sql.functions as F


class PySparkJob(PySparkJobInterface):

    def init_spark_session(self) -> SparkSession:
        spark = SparkSession.builder \
            .master("local") \
            .appName("Data Cleaning") \
            .getOrCreate()
        return spark

    def filter_medical(self, eligibility: DataFrame, medicals: DataFrame) -> DataFrame:
        filtered_medical = medicals.join(eligibility.select("memberId"), on="memberId", how="inner")
        return filtered_medical

    def generate_full_name(self, eligibility: DataFrame, medical: DataFrame) -> DataFrame:
        eligibility_with_full_name = eligibility.withColumn(
            "generatedFullName",
            F.concat_ws(" ", F.col("firstName"), F.col("lastName"))
        )
        
        # Select only the necessary columns to avoid ambiguity
        eligibility_reduced = eligibility_with_full_name.select("memberId", "generatedFullName")
        
        # Join medical DataFrame with the reduced eligibility DataFrame
        updated_medical = medical.join(
            eligibility_reduced,
            on="memberId",
            how="left"
        )
        
        # Rename 'generatedFullName' to 'fullName'
        updated_medical = updated_medical.withColumnRenamed("generatedFullName", "fullName")
        
        return updated_medical

    def find_max_paid_member(self, medicals: DataFrame) -> str:
        max_paid_row = medicals.orderBy(F.col("paidAmount").desc()).first()
        return max_paid_row["memberId"]

    def find_total_paid_amount(self, medicals: DataFrame) -> int:
        total_amount = medicals.agg(F.sum("paidAmount").alias("totalPaid")).first()
        return total_amount["totalPaid"]
