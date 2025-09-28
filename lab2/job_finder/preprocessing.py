import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('spanish'))
nltk.download('punkt')
nltk.download('stopwords')

def limpiar_texto(texto):
      texto = texto.lower()
      return re.sub(r'[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ /]', '', texto)

def tokenizar_df(df, columna):
    df[columna + '_tokens'] = df[columna].apply(word_tokenize)
    return df

def eliminar_stopwords_df(df, columna):
    df[columna + '_sin_stopwords'] = df[columna + '_tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
    return df

def lematizar_df(df, columna):
    lemmatizer = WordNetLemmatizer()
    df[columna + '_lematizado'] = df[columna + '_sin_stopwords'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
    return df

def unir_tokens_df(df, columna):
    df[columna + '_final'] = df[columna + '_lematizado'].apply(lambda tokens: ' '.join(tokens))
    return df

def vectorizar_texto(df, columna, path_vectorizer="static/data/vectorizer.pickle"):
    vectorizer = pickle.load(open(path_vectorizer, "rb"))
    X = vectorizer.transform(df[columna + '_final'])
    return X, vectorizer

def preprocess_text(df, columna, path_vectorizer="job_finder/static/data/vectorizer.pickle"):
    df[columna] = df[columna].astype(str)
    df[columna + '_limpio'] = df[columna].apply(limpiar_texto)
    df = tokenizar_df(df, columna + '_limpio')
    df = eliminar_stopwords_df(df, columna + '_limpio')
    df = lematizar_df(df, columna + '_limpio')
    df = unir_tokens_df(df, columna + '_limpio')
    X, vectorizer = vectorizar_texto(df, columna + '_limpio', path_vectorizer)
    return X, vectorizer, df