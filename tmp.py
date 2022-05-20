"""
load dataframe from a csv
"""
from pyspark.ml.linalg import DenseMatrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import TimestampType, DoubleType, LongType, \
        StringType, StructField, StructType
from pyspark.sql.functions import isnan, when, count, datediff, mean, lag
from pyspark.sql.window import Window


def create_spark_session(name: str) -> SparkSession:
    """
    Create a spark session using the <name>
    """
    return SparkSession.builder.appName(name).getOrCreate()


def load_data(spark_session: SparkSession, path: str) -> DataFrame:
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
    return spark_session.read.csv(path, dev_schema, header=True)


def count_nan(data_frame: DataFrame) -> DataFrame:
    """
    Count nan in the <df> DataFrame
    """
    return data_frame.select([
                count(when(isnan(c), c)).alias(c)
                for c
                in ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
              ])


def duration_between_rows(df: DataFrame):
    df = df.orderBy('Date')
    df = df.withColumn('duration',
                       datediff(df['Date'],
                                lag(df['Date'], 1).over(Window.partitionBy("company_name").orderBy('Date')))
                       )
    return df.select(mean('duration')).collect()[0][0]


def corr_two_columns(data_frame: DataFrame, col1: str, col2: str) -> float:
    """
    Return the correlation between two columns of the dataframe
    """
    return data_frame.stat.corr(col1, col2)


def corr_matrix(data_frame: DataFrame) -> DenseMatrix:
    """
    Return the correlation matrix of a dataframe
    """

    vector_col = "corr_features"
    assembler = VectorAssembler(
            inputCols=["High", "Low", "Open", "Close", "Volume", "Adj Close"],
            outputCol=vector_col
    )
    df_vector = assembler.transform(data_frame).select(vector_col)

    matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]

    return matrix


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    df = load_data(spark, 'stocks_data/AMAZON.csv')

    df.printSchema()
    df.show(40)
    df.count()
    print(duration_between_rows(df))

    # Descriptive statistics for each dataframe and each column (min, max,
    # standard deviation)
    df.describe().show()

    # Number of missing values for each dataframe and column
    count_nan(df).show()

    # Correlation between values
    pearsonCorr = corr_two_columns(df, "High", "Low")
    corr = corr_matrix(df)
    print(corr)
