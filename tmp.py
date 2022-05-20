import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window


def create_spark_session(name: str) -> SparkSession:
    return SparkSession.builder.appName(name).getOrCreate()


def load_data(path: str) -> DataFrame:
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


def duration_between_rows(df: DataFrame):
    df = df.orderBy('Date')
    df = df.withColumn('duration',
                       datediff(df['Date'],
                                lag(df['Date'], 1).over(Window.partitionBy("company_name").orderBy('Date')))
                       )
    return df.select(mean('duration')).collect()[0][0]


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    df = load_data('stocks_data/AMAZON.csv')

    df.printSchema()
    df.show(40)
    df.count()
    print(duration_between_rows(df))
