import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

sw = stopwords.words('english')
word_data = pickle.load( open("your_word_data.pkl"))
vectorizer = TfidfVectorizer(stop_words = 'english')

W = vectorizer.fit_transform(word_data)
words = vectorizer.get_feature_names()
