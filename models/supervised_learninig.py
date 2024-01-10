
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./training_data.csv')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Clean text documents
df['cleaned_text'] = df['text'].apply(clean_text)

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
    return tokens

stop_words = set(stopwords.words('english'))
df['tokenized_text'] = df['cleaned_text'].apply(tokenize_text)

tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(df['tokenized_text'])]


model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


# Feature extraction
X = [model.infer_vector(doc) for doc in df['tokenized_text']]
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = svm_classifier.score(X_test, y_test)
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# from joblib import dump

# dump(svm_classifier, 'model.joblib')
