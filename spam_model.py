
import pickle

def load_model():
    with open('spam_model.pkl', 'rb') as f:
        spam_model = pickle.load(f)
    return spam_model

def predict(message, model=load_model()):
    spam_model = model
    prediction = spam_model.predict([message])[0]
    return 'Spam' if prediction == 1 else 'Not Spam'

if __name__ == '__main__':
    model=load_model()
    message = input('Enter a message: ')
    print(predict(message, model))
    
