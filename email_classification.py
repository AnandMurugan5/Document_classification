import nltk
import numpy as np
import warnings
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup as BS
from email_extraction import EmailData
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

email_client = EmailData('anandraman249@gmail.com', 'fogw nlsi euix lnbu')
email_client.login()
emails_data = email_client.fetch_emails()
emails_data = [email[1] for email in emails_data]

def clean_html(text):
    soup = BS(text, 'html.parser')
    return soup.get_text()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

cleaned_emails_data = [clean_html(email) for email in emails_data]
preprocessed_emails_data = [preprocess_text(email) for email in cleaned_emails_data]

cleaned_emails_data

preprocessed_emails_data

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_emails_data)

# LDA model
lda_model = LatentDirichletAllocation(n_components=7, random_state=42)
lda_model.fit(tfidf_matrix)

def top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx + 1}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


top_words(lda_model, tfidf_vectorizer.get_feature_names_out(), n_top_words=10)

lda_topics = lda_model.transform(tfidf_matrix)

kmeans = KMeans(n_clusters=5)  # You can adjust the number of clusters as needed
kmeans.fit(lda_topics)
clusters = kmeans.labels_

email_categories = np.array(['Work', 'Personal', 'Promotions', 'Finance', 'Travel', 'Health', 'Education', 'Technology', 'Entertainment', 'Events', 'Food', 'Shopping', 'Hobbies', 'ScienceSocial', 'Issues', 'Gaming', 'Security'])

X_train, X_test, topics_train, topics_test, cat_train, cat_test = train_test_split(tfidf_matrix, lda_topics, email_categories[clusters], test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, cat_train)
y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(cat_test, y_pred)
print(f"Accuracy:\n {accuracy}")
report = classification_report(cat_test, y_pred)
print(f"Classification Report:\n{report}")

import pickle

# Save the trained model to a file
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Save lda_model
with open('lda_model.pkl', 'wb') as file:
    pickle.dump(lda_model, file)

# Save nb_classifier
with open('nb_classifier.pkl', 'wb') as file:
    pickle.dump(nb_classifier, file)


