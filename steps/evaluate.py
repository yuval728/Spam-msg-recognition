from zenml import step
from typing import Any
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow

@step
def evaluate_model(model: Any, test_data: Any) -> None:
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc = evaluator.evaluate(predictions)
    mlflow.log_metric("accuracy", acc)
    print(f"Model accuracy: {acc * 100:.2f}%")