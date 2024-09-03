import abc

from pyspark.sql.types import StructType

from pyspark.sql import SparkSession, DataFrame


class PySparkJobInterface(abc.ABC):

    def __init__(self):
        self.spark = self.init_spark_session()

    @abc.abstractmethod
    def init_spark_session(self) -> SparkSession:
        """Create spark session"""
        raise NotImplementedError

    def read_csv(self, input_path: str, schema: StructType = None) -> DataFrame:
        reader = self.spark.read
        if schema is None:
            return reader.options(header=True, inferSchema=True).csv(input_path)
        else:
            return reader.options(header=True).schema(schema).csv(input_path)

    @abc.abstractmethod
    def filter_medical(self, eligibility: DataFrame, medical: DataFrame) -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def generate_full_name(self, eligibility: DataFrame, medical: DataFrame) -> DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def find_max_paid_member(self, medicals: DataFrame) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def find_total_paid_amount(self, medicals: DataFrame) -> int:
        raise NotImplementedError

    def stop(self) -> None:
        self.spark.stop()
