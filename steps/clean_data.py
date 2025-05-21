from zenml import step
from pyspark.sql import DataFrame

@step
def clean_data(df: DataFrame) -> DataFrame:
    df = df.drop("Unnamed: 2", "Unnamed: 3", "Unnamed: 4")
    df = df.withColumnRenamed("v1", "TARGET").withColumnRenamed("v2", "MESSAGE")
    df = df.dropna()
    df = df.dropDuplicates()
    return df
