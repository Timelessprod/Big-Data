from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *


def create_spark_session(name: str) -> SparkSession:
    return (SparkSession.builder.appName(name).getOrCreate())


def load_data(path: str) -> DataFrame:
    devSchema = StructType([
        StructField("Date", TimestampType()),
        StructField("High", DoubleType()),
        StructField("Low", DoubleType()),
        StructField("Open", DoubleType()),
        StructField("Close", DoubleType()),
        StructField("Volume", LongType()),
        StructField("Adj Close", DoubleType()),
        StructField("company_name", StringType()),
    ])
    return spark.read.csv(path, devSchema, header = True)


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    df = load_data('stocks_data/AMAZON.csv')

    df.printSchema()
    df.show(40)
