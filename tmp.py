"""
load dataframe from a csv
"""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import TimestampType, DoubleType, LongType, \
        StringType, StructField, StructType
from pyspark.sql.functions import isnan, when, count


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


def count_nan(data_frame: DataFrame) -> DataFrame:
    """
    Count nan in the <df> DataFrame
    """
    return data_frame.select([
                count(when(isnan(c), c)).alias(c)
                for c
                in ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
              ])


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    df = load_data('stocks_data/AMAZON.csv')

    df.printSchema()
    df.show(40)

    # Descriptive statistics for each dataframe and each column (min, max,
    # standard deviation)
    df.describe().show()

    # Number of missing values for each dataframe and column
    count_nan(df).show()
