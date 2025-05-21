from pipelines.spam_pipeline import create_pipeline
# from zenml.client import Client

# Client().activate_stack("mlflow_stack")

def main():
    pipeline = create_pipeline()
    pipeline.run()

if __name__ == "__main__":
    main()