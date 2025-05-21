from zenml import step
from typing import Any
from pyspark.ml.classification import LogisticRegression

@step
def train_model(train_data: Any) -> Any:
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_data)
    return model
