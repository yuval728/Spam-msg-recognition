
import pickle

def load_model():
    with open('spam_model.pkl', 'rb') as f:
        spam_model = pickle.load(f)
    return spam_model

def predict(message, spam_model):
    prediction = spam_model.predict([message])[0]
    prediction = "Spam" if prediction == 1 else "Not Spam"
    prediction_proba = spam_model.predict_proba([message])[0]
    return prediction, prediction_proba

if __name__ == '__main__':
    spam_model = load_model()
    message = input('Enter a message: ')
    prediction, prediction_proba = predict(message, spam_model)
    print(f'Message is: {prediction} with probability {prediction_proba.max() * 100:.2f}')
    
