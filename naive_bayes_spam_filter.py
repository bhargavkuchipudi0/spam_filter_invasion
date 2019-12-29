import pandas as pd
import string
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# Create Tokenizer
def process_text(text):
    # 1 Removing punctuations
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # 2 Removing stop words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    # 3 Returning the clean text words
    return clean_words


def main():
    # Data pre-processing
    messages = pd.read_csv('spam.csv', encoding='latin-1')
    messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    messages = messages.rename(columns={'v1': 'class', 'v2': 'text'})
    messages['length'] = messages['text'].apply(len)
    messages['text'].apply(process_text).head()
    msg_train, msg_test, class_train, class_test = train_test_split(messages['text'],messages['class'],test_size=0.2)

    # Model Creation
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=process_text)),  # converts strings to integer counts
        ('tfidf', TfidfTransformer()),  # converts integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB())  # train on TF-IDF vectors with Naive Bayes classifier
    ])

    classifier = pipeline.fit(msg_train,class_train)
    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    predictions = pipeline.predict(msg_test)
    print(classification_report(class_test, predictions))


if __name__ == '__main__':
    main()
