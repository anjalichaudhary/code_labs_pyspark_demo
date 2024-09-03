from pyspark.sql.types import StructType, StringType, StructField, IntegerType

eligibility = StructType([
    StructField("memberId", StringType()),
    StructField("firstName", StringType()),
    StructField("lastName", StringType())
])

medical = StructType([
    StructField("memberId", StringType()),
    StructField("fullName", StringType()),
    StructField("paidAmount", IntegerType())
])
