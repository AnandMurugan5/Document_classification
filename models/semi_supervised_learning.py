from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import re
import PyPDF2
import os

# Load labeled data
labeled_df = pd.read_excel("./sample_data.xlsx")

# Function to clean and tokenize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

labeled_df['cleaned_text'] = labeled_df['Text'].apply(clean_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(labeled_df['cleaned_text'], labeled_df['category'], test_size=0.2, random_state=42)

# Convert text data to numerical vectors using TF-IDF for SVM
vectorizer_svm = TfidfVectorizer()
X_train_tfidf = vectorizer_svm.fit_transform(X_train)
X_test_tfidf = vectorizer_svm.transform(X_test)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Evaluate the SVM model
predictions = svm_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")

# Load unlabeled data (PDFs in 'Documents/' directory)
def load_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
        text = clean_text(text)
        return text

email_attachments_pdf = [os.path.join(root, filename) for root, dirs, files in os.walk('Documents/') for filename in files if filename.endswith('.pdf')]
documents = [load_pdf(path) for path in email_attachments_pdf]
unlabeled_df = pd.DataFrame()
unlabeled_df['text'] = documents
unlabeled_df['cleaned_text'] = unlabeled_df['text'].apply(clean_text)

# Predict labels for unlabeled data using the trained SVM model
X_unlabeled_tfidf = vectorizer_svm.transform(unlabeled_df['cleaned_text'])
predicted_labels_unlabeled = svm_classifier.predict(X_unlabeled_tfidf)

# Add predicted labels to the unlabeled DataFrame
unlabeled_df['predicted_label'] = predicted_labels_unlabeled
print(unlabeled_df[['text', 'predicted_label']])


import pickle

# Save the trained SVM classifier
with open('svm_classifier.pkl', 'wb') as file:
    pickle.dump(svm_classifier, file)

# Save the TF-IDF vectorizer
with open('document_tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer_svm, file)
