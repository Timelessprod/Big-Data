"""
load dataframe from a csv
"""
import time
import datetime
from math import sqrt
from pyspark.sql.functions import isnan, when, count, datediff, mean, lag, col, avg, variance, stddev, asc, desc, sum
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
import numpy as np


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


def plot_corr_matrix(matrix):
    """
    Plot the correlation matrix
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(matrix, cmap=plt.cm.Blues, vmin=-1, vmax=1)
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.show()


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
    return df.stat.corr(col_in_df1, col_in_df2)


def relative_strength_index(df: DataFrame) -> DataFrame:
    """
    Add new column with relative strength index
    """
    return df.withColumn("rsi", 100 - 100 / (1 + max(df.Close - df.Open, 0) / (df.Open - df.Close)))


def compute_moving_average(stock_history: DataFrame, column: str, n: int = 5) -> DataFrame:
    ordered_window = Window.partitionBy("company_name").orderBy("Date").rowsBetween(-1 * n, 0)
    return stock_history.withColumn("daily_average", avg(stock_history[column]).over(ordered_window))


def compute_volatility(df: DataFrame, start_date: TimestampType(), end_date: TimestampType(), n: int = 5):
    selected_df = df.filter(col("Date") <= start_date).filter(col("Date") <= end_date)
    selected_w_average = compute_moving_average(selected_df, "Close", n)
    return selected_w_average.select(stddev("daily_average"))


def compute_momentum(stock_history: DataFrame, n: int = 5) -> DataFrame:
    return compute_moving_average(stock_history, "Close", n)


def upside_downside_ratio(stocks: list[DataFrame]) -> DataFrame:
    curated_stocks = []
    for i, stock in enumerate(stocks):
        stock_w_advancements = stock.withColumn(f"advancement_{i}", (stock["Close"] - stock["Open"]) * stock["Volume"])
        advancements_only = stock_w_advancements.select("Date", f"advancement_{i}")
        curated_stocks.append(advancements_only)

    all_joined = curated_stocks[0]
    for df_next in curated_stocks[1:]:
        all_joined = all_joined.join(df_next, on='Date', how='inner')

    def gen_pos(frame: DataFrame, pred: callable):
        res = frame["advancement_0"] * (frame["advancement_0"] > 0).cast("double")
        for index in range(len(stocks) - 1):
            res += frame[f"advancement_{index + 1}"] * (pred(frame[f"advancement_{index + 1}"])).cast("double")
        return res

    out = all_joined.withColumn("total_upside", gen_pos(all_joined, lambda x: x > 0))
    out = out.withColumn("total_downside", gen_pos(out, lambda x: x < 0))
    out = out.withColumn("upside_downside_ratio", out["total_upside"] / out["total_downside"])
    out = out.select("Date", "upside_downside_ratio")
    return out


def compute_average_over_period(frame: DataFrame, col_name: str, start_date: TimestampType(),
                                end_date: TimestampType()):
    return frame.filter(col("Date") >= start_date).filter(col("Date") <= end_date).agg(avg(col(col_name)))


def compute_daily_return(frame: DataFrame) -> DataFrame:
    return frame.withColumn("daily_return", (frame["Close"] - frame["Open"]) / frame["Open"] * 100)


def compute_return_rate_over_period(frame: DataFrame, start_date: TimestampType(), end_date: TimestampType()):
    curated_frame = frame.filter(col("Date") >= start_date).filter(col("Date") <= end_date)
    open_val = curated_frame.filter(col("Date") == start_date).head(1)[0].__getitem__("Open")
    close_val = curated_frame.filter(col("Date") == end_date).head(1)[0].__getitem__("Close")
    return (close_val - open_val) / open_val


def highest_daily_return(stocks: list[DataFrame]) -> DataFrame:
    stocks_w_daily_return = list(map(lambda x: compute_daily_return(x), stocks))
    all_joined = stocks_w_daily_return[0]
    for df_next in stocks_w_daily_return[1:]:
        all_joined = all_joined.union(df_next)
    return all_joined.sort(desc('daily_return')).dropDuplicates(['date']).select("Date", "company_name", "daily_return")


def highest_period_return(stocks: list[DataFrame], start_date: TimestampType(), end_date: TimestampType()):
    highest_val = compute_return_rate_over_period(stocks[0], start_date, end_date)
    name = stocks[0].head(1)[0].__getitem__("company_name")
    for stock in stocks[1:]:
        return_rate = compute_return_rate_over_period(stock, start_date, end_date)
        if return_rate > highest_val:
            highest_val = return_rate
            name = stock.head(1)[0].__getitem__("company_name")
    return name, highest_val


def compute_intraday_momentum_index(stock: DataFrame, n: int = 14):
    ordered_window = Window.partitionBy("company_name").orderBy("Date").rowsBetween(-1 * n, 0)
    stock_w_daily_gains = stock.withColumn("daily_gain", stock["Close"] - stock["Open"])
    stock_w_daily_gains.show()
    stock_w_daily_gl = stock_w_daily_gains.withColumn("daily_loss",
                                                      stock_w_daily_gains["Open"] - stock_w_daily_gains["Close"])
    stock_w_daily_gbool = stock_w_daily_gl.withColumn("gain_bool",
                                                      (stock_w_daily_gl["Close"] > stock_w_daily_gl["Open"])
                                                      .cast("Double"))
    stock_w_daily_glbool = stock_w_daily_gbool.withColumn("loss_bool",
                                                          (stock_w_daily_gbool["Open"] > stock_w_daily_gbool["Close"])
                                                          .cast("Double"))
    stock_w_daily_netg = stock_w_daily_glbool.withColumn("net_gain", stock_w_daily_glbool["daily_gain"] *
                                                         stock_w_daily_glbool["gain_bool"])
    stock_w_daily_netgl = stock_w_daily_netg.withColumn("net_loss", stock_w_daily_netg["daily_loss"] *
                                                         stock_w_daily_glbool["loss_bool"])
    stock_w_momentum_gains = stock_w_daily_netgl.withColumn("mgains", sum(stock_w_daily_netgl["net_gain"])
                                                            .over(ordered_window))
    stock_w_mgl = stock_w_momentum_gains.withColumn("mlosses", sum(stock_w_momentum_gains["net_loss"])
                                                    .over(ordered_window))
    stock_w_imi = stock_w_mgl.withColumn("imi", stock_w_mgl["mgains"] /
                                         (stock_w_mgl["mgains"] + stock_w_mgl["mlosses"]) * 100)
    return stock_w_imi


def compute_money_flow_index(stock: DataFrame, n: int = 14):
    ordered_window = Window.partitionBy("company_name").orderBy("Date").rowsBetween(-1 * n, 0)
    one_window = Window.partitionBy("company_name").orderBy("Date")
    stock_w_typical = stock.withColumn("typical_price", stock["High"] + stock["Low"] + stock["Close"])
    stock_w_raw = stock_w_typical.withColumn("raw_money_flow",
                                             stock_w_typical["typical_price"] * stock_w_typical["Volume"])
    stock_w_is_pos = stock_w_raw.withColumn("is_pos_flow",
                                            (lag(stock_w_raw["typical_price"]).over(one_window)
                                             < stock_w_raw["typical_price"])
                                            .cast("Double"))
    stock_w_ispl = stock_w_is_pos.withColumn("is_neg_flow",
                                             (lag(stock_w_is_pos["typical_price"]).over(one_window)
                                              > stock_w_is_pos["typical_price"])
                                             .cast("Double"))
    stock_w_period_pos = stock_w_ispl.withColumn("pos_period_flow",
                                                 sum(stock_w_ispl["is_pos_flow"] * stock_w_ispl["raw_money_flow"])
                                                 .over(ordered_window))
    stock_w_period_pn = stock_w_period_pos.withColumn("neg_period_flow",
                                                      sum(stock_w_period_pos["is_neg_flow"] * stock_w_period_pos[
                                                          "raw_money_flow"])
                                                      .over(ordered_window))
    stock_w_ratio = stock_w_period_pn.withColumn("money_flow_ratio",
                                                 stock_w_period_pn["pos_period_flow"] / stock_w_period_pn[
                                                     "neg_period_flow"])
    return stock_w_ratio.withColumn("money_flow_index", 100 / (1 + stock_w_ratio["money_flow_ratio"]))


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
    pearson_corr = data_frame.stat.corr('High', 'Low')
    print(pearson_corr)
    plot_corr_matrix(pearson_corr.toArray())
    print("Correlation matrix:")
    corr = corr_matrix(data_frame)
    print(corr, '\n')

    print("RSI")
    data_frame = relative_strength_index(data_frame)
    print(data_frame.select("rsi").describe())

    print("Moving Average")
    avgdf = compute_moving_average(data_frame, "Close")
    avgdf.show()

    print("volatility")
    format_date = "%Y-%m-%d %H:%M:%S"
    start = datetime.datetime.strptime("2019-04-18 00:00:00", format_date)
    end = datetime.datetime.strptime("2019-05-16 00:00:00", format_date)
    vol = compute_volatility(df, start, end)
    vol.show()
    print(f"average opening from {start} to {end}")
    compute_average_over_period(df, "Open", start, end).show()
    print(f"average closing from {start} to {end}")
    compute_average_over_period(df, "Close", start, end).show()
    print("daily return rate")
    compute_daily_return(df).show()
    print(f"return rate from {start} to {end}")
    print(compute_return_rate_over_period(df, start, end))
    print("IMI score")
    compute_intraday_momentum_index(df).show()
    print("Money Flow Index")
    compute_money_flow_index(df).show()


if __name__ == "__main__":
    spark = create_spark_session("Spark_Application_Name")
    all_datasets = []
    for f in ['AMAZON.csv', 'APPLE.csv', 'FACEBOOK.csv', 'GOOGLE.csv',
              'MICROSOFT.csv', 'TESLA.csv', 'ZOOM.csv']:
        print(f"\n{f}:")
        df = load_data(spark, 'stocks_data/' + f)
        all_datasets.append(df)
        describe_data_frame(df)
    print("upside_downside_ratio")
    upside_downside_ratio(all_datasets).show()
    print("highest daily return")
    highest_daily_return(all_datasets).show()
    date_format = "%Y-%m-%d %H:%M:%S"
    start = datetime.datetime.strptime("2019-04-18 00:00:00", date_format)
    end = datetime.datetime.strptime("2019-05-16 00:00:00", date_format)
    print(f"highest return from {start} to {end}")
    print(highest_period_return(all_datasets, start, end))
