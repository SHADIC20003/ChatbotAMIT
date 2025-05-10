from flask import Flask, request, jsonify, send_from_directory
from transformers import BertTokenizer, BertModel
import torch
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer from specified directories
tokenizer = BertTokenizer.from_pretrained('bert_Tokenizer')
bert_model = BertModel.from_pretrained('bert_model')

# Load SVM model
svm_model = joblib.load('svm_model.joblib')

def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def get_bot_response(user_message):
    embeddings = get_bert_embeddings([user_message])
    prediction = svm_model.predict(embeddings)
    # Replace with your actual response logic if needed
    return prediction[0]

# Initialize chat history
chat_history = []

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    user_message = request.form.get('message', '').strip()
    
    if user_message:
        # Add user message to chat history
        chat_history.append({"text": user_message, "sender": "user", "timestamp": "10:01 AM"})
        
        # Generate a chatbot response
        bot_response = get_bot_response(user_message)
        chat_history.append({"text": bot_response, "sender": "chatbot", "timestamp": "10:02 AM"})
    
    return jsonify({'chat_history': chat_history})

@app.route('/new_chat', methods=['POST'])
def new_chat():
    global chat_history
    # Clear chat history
    chat_history = []
    return jsonify({'chat_history': chat_history})

if __name__ == '__main__':
    app.run(debug=True)
