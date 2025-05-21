from zenml import step
from pyspark.sql import SparkSession, DataFrame

@step
def ingest_data() -> DataFrame:
    spark = (
        SparkSession.builder
        .appName("Spam Detection")
        .config("spark.security.manager.enabled", "false")
        .getOrCreate()
    )
    df = spark.read.csv("spam.csv", header=True, encoding='ISO-8859-1')
    return df