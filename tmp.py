"""
load dataframe from a csv
"""
from pyspark.ml.linalg import DenseMatrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import TimestampType, DoubleType, StringType, \
    StructField, StructType
from pyspark.sql.functions import isnan, when, count, datediff, mean, lag, \
    col, month, year, weekofyear, date_format, dayofmonth
from pyspark.sql.window import Window
import matplotlib as plt


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
        StructField("Volume", DoubleType()),
        StructField("Adj Close", DoubleType()),
        StructField("company_name", StringType()),
    ])
    return spark_session.read.csv(path, dev_schema, header=True)


def count_nan(data_frame: DataFrame) -> DataFrame:
    """
    Count nan in the <df> DataFrame
    """
    return data_frame.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c)
        for c
        in ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
    ])


def duration_between_rows(data_frame: DataFrame):
    """
    Calculate the mean duration between each rows
    """
    data_frame = data_frame.orderBy('Date')
    data_frame = data_frame.withColumn('duration',
                                       datediff(data_frame['Date'],
                                                lag(data_frame['Date'], 1).over(
                                                    Window.partitionBy("company_name").orderBy('Date')))
                                       )
    return data_frame.select(mean('duration')).collect()[0][0]


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


def describe_data_frame(data_frame: DataFrame):
    """
    Describe the dataframe
    - print the first and last 40 lines
    - print the number of observations
    - print the period between the data points
    - print the min, max, mean and standard deviation
    - print the number of missing values for each dataframe and column
    - print correlation matrix
    """
    print("Dataframe schema:")
    data_frame.printSchema()

    print("First 40 lines:")
    data_frame.show(40)

    print(f"Number of observations: {data_frame.count()}\n")

    print("Period between data points:")
    print(duration_between_rows(data_frame))

    # Descriptive statistics for each dataframe and each column (min, max,
    # standard deviation)
    data_frame.describe().show()

    print("Number of missing values for each dataframe and column:")
    count_nan(data_frame).show()

    print("Correlation between 'High' and 'Low':")
    pearson_corr = corr_two_columns(data_frame, "High", "Low")
    print(pearson_corr)
    print("Correlation matrix:")
    corr = corr_matrix(data_frame)
    print(corr, '\n')


def week_mean(data_frame: DataFrame) -> DataFrame:
    """
    Return average of each week for columns Open and Close
    """
    return data_frame.withColumn("Year", year("Date")) \
        .withColumn("Week", weekofyear("Date")) \
        .groupBy("Year", "Week") \
        .avg("Open", "Close") \
        .orderBy(["Year", "Week"])


def month_mean(data_frame: DataFrame) -> DataFrame:
    """
    Return average of each month for columns Open and Close
    """
    return data_frame.withColumn("Year", year("Date")) \
        .withColumn("Month", month("Date")) \
        .groupBy("Year", "Month") \
        .avg("Open", "Close") \
        .orderBy(["Year", "Month"])


def year_mean(data_frame: DataFrame) -> DataFrame:
    """
    Return average of each year for columns Open and Close
    """
    return data_frame.withColumn("Year", year("Date")) \
        .groupBy("Year") \
        .avg("Open", "Close") \
        .orderBy("Year")


def change_day_to_day(df: DataFrame, col: str) -> DataFrame:
    """
    Add new column with comparaison between previous and next row value in column col
    """
    df = df.orderBy('Date')
    return df.withColumn(col + "_change",
                         lag(df[col], 1).over(Window.partitionBy("company_name").orderBy('Date')) - df[col]
                         )


def change_month_to_month(df: DataFrame, col_name: str) -> DataFrame:
    """
    Add new column with comparaison between previous and next month in column <col>
    """
    return df.withColumn("Year", year("Date")) \
        .withColumn("Month", month("Date")) \
        .groupBy("Year", "Month") \
        .avg(col_name) \
        .orderBy(["Year", "Month"]) \
        .withColumnRenamed(f'avg({col_name})', col_name) \
        .withColumn(col_name + "_change",
                         col(col_name) - lag(col(col_name),
                             1).over(Window.orderBy(["Year", "Month"]))
                         )


def candle_sticks(data, Month: int, Year: int, saveoption: bool = None):
    """
    data : pyspark dataframe
    month : number related to the focus month
    year : number related to the focus year
    saveoption : save the figure into a file "month_year.png"
    """
    data = data.withColumn('monthandday', date_format(data.Date, "d MMM"))
    data = data.withColumn('day', dayofmonth(data.Date))
    data = data.withColumn('month', month(data.Date))
    data = data.withColumn('year', year(data.Date))

    data = data.filter(data.month == Month).filter(data.year == Year)
    pd_am = data.toPandas()
    pd_am.index = pd_am.Date
    pd_am = pd_am.drop(columns="Date")
    up = pd_am[pd_am.Close >= pd_am.Open]
    down = pd_am[pd_am.Close < pd_am.Open]

    heigh = 0.1
    width = 0.5

    plt.figure(figsize=(9, 7))
    plt.title("CandleStick Chart")

    plt.bar(up.monthandday, up.Close - up.Open, width, bottom=up.Open, color='green')
    plt.bar(up.monthandday, up.High - up.Close, heigh, bottom=up.Close, color='green')
    plt.bar(up.monthandday, up.Low - up.Open, heigh, bottom=up.Open, color='green')

    plt.bar(down.monthandday, down.Close - down.Open, width, bottom=down.Open, color='red')
    plt.bar(down.monthandday, down.High - down.Open, heigh, bottom=down.Open, color='red')
    plt.bar(down.monthandday, down.Low - down.Close, heigh, bottom=down.Close, color='red')

    plt.xticks(rotation=45, ha='right')
    plt.legend(["green", "red"])
    if saveoption:
        sgd = str(Month) + '_' + str(Year)
        plt.savefig(sgd)
    plt.show()


def correlate_two_dataframe(
        df1:DataFrame,
        df2:DataFrame,
        col_in_df1:str,
        col_in_df2:str = None) -> float:
    """
    by default, col_in_df2 is equal to col_in_df1
    """
    if col_in_df2 is None:
        col_in_df2 = col_in_df1
    if col_in_df1 == col_in_df2:
        col_in_df2 = f'{col_in_df1}_2'
        df2 = df2.withColumnRenamed(col_in_df1, col_in_df2)

    df = df1.join(df2, 'Date', 'inner').select(col_in_df1, col_in_df2)
    return corr_two_columns(df, col_in_df1, col_in_df2)


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    dfs = []
    for f in ['AMAZON.csv', 'APPLE.csv', 'FACEBOOK.csv', 'GOOGLE.csv',
              'MICROSOFT.csv', 'TESLA.csv', 'ZOOM.csv']:
        print(f"\n{f}:")
        df = load_data(spark, 'stocks_data/' + f)
        # describe_data_frame(df)

        week_mean(df).show()
        month_mean(df).show()
        year_mean(df).show()

        dfs.append(df)

        df = change_day_to_day(df, 'Open')
        df = change_day_to_day(df, 'Close')
        change_month_to_month(df, 'Open').show()

    print(correlate_two_dataframe(dfs[0], dfs[1], 'Open'))
