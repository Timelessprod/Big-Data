"""
load dataframe from a csv
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import TimestampType, DoubleType, LongType, \
        StringType, StructField, StructType


def create_spark_session(name: str) -> SparkSession:
    """
    Create a spark session using the <name>
    """
    return (SparkSession.builder.appName(name).getOrCreate())


def load_data(path: str) -> DataFrame:
    """
    Create spark dataframe from the csv file at <path>
    """
    dev_schema = StructType([
        StructField("Date", TimestampType()),
        StructField("High", DoubleType()),
        StructField("Low", DoubleType()),
        StructField("Open", DoubleType()),
        StructField("Close", DoubleType()),
        StructField("Volume", LongType()),
        StructField("Adj Close", DoubleType()),
        StructField("company_name", StringType()),
    ])
    return spark.read.csv(path, dev_schema, header=True)


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    df = load_data('stocks_data/AMAZON.csv')

    df.printSchema()
    df.show(40)

    df.describe().show()
