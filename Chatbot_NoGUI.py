import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
import torch
import joblib

# Load the dataset from the provided Excel file
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

# Separate the features and target
X = df['User Inputs']
y = df['Chatbot Response']

# Text Preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

X = X.apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# Get BERT embeddings for training and testing data
X_train_embeddings = get_bert_embeddings(X_train.tolist())
X_test_embeddings = get_bert_embeddings(X_test.tolist())

# Train SVM model
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_embeddings, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test_embeddings)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report: \n{report}")

# Save the SVM model and BERT components
joblib.dump(svm, 'svm_model.joblib')
tokenizer.save_pretrained('bert_tokenizer')
model.save_pretrained('bert_model')

