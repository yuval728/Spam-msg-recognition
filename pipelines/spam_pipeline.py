from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.preprocess import preprocess_data
from steps.train_model import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache=True)
def create_pipeline():
    raw_df = ingest_data()
    cleaned_df = clean_data(raw_df)
    train, test = preprocess_data(cleaned_df)
    model = train_model(train)
    evaluate_model(model, test)