from zenml import step
from pyspark.sql import DataFrame
from typing import Tuple
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand

@step
def preprocess_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    df = df.orderBy(rand())

    label_indexer = StringIndexer(inputCol="TARGET", outputCol="label")
    tokenizer = Tokenizer(inputCol="MESSAGE", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")

    pipeline = Pipeline(stages=[label_indexer, tokenizer, remover, vectorizer])
    model = pipeline.fit(df)
    transformed = model.transform(df)

    train, test = transformed.randomSplit([0.8, 0.2], seed=3)
    train = train.select("features", "label")
    test = test.select("features", "label")
    return train, test
