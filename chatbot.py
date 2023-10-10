import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model


#data_file = open('intents.json').read()
#intents= json.loads(data_file)

#words = pickle.load(open('words.pkl', 'rb'))
file_path = 'C:\\Users\\poorn\\PycharmProjects\\pythonProject1\\words.pkl'

try:
    with open(file_path, 'rb') as file:
        words = pickle.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
#classes = pickle.load(open('classes.pkl', 'rb'))
file_path = 'C:\\Users\\poorn\\PycharmProjects\\pythonProject1\\classes.pkl'

try:
    with open(file_path, 'rb') as file:
        classes = pickle.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")

#model = load_model('chatbot_model.h5')

model_path = 'C:\\Users\\poorn\\PycharmProjects\\pythonProject1\\chatbot_model.h5'

try:
    model = load_model(model_path)
except OSError:
    print(f"File '{model_path}' not found.")
lemmatizer = WordNetLemmatizer()
file_path = 'C:\\Users\\poorn\\PycharmProjects\\pythonProject1\\intents.json'

try:
    with open(file_path, 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence,model):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


#print("GO! Bot is running!")

#while True:
   # message = input("")
    #ints = predict_class(message)
    #res = get_response(ints, intents)
    #print(res)
