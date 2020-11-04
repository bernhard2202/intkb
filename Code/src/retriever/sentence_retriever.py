import logging
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
stemmer = PorterStemmer()

#TODO multiple keyword query
#TODO stem/lemma


def tokenize(text):
    tokens = [word.lower() for word in nltk.word_tokenize(text) if len(word) > 1]
    stems = [stemmer.stem(item) for item in tokens]
    return stems


def retrieve_sentences_multi_kw_query(queries, doc_files, k=5):
    # for query in queries:
    #     print('query: ', query)
    # for doc in doc_files:
    #     print('doc: ', doc)
    sentences = []
    sentences_original = []
    for doc in doc_files:
        if not os.path.exists(doc):
            logging.warn('Document not found {}'.format(doc))
            continue
        with open(doc, 'r') as f:
            doc_string = f.read()
        for para in doc_string.split('\n'):
            for sentence in nltk.sent_tokenize(para):
                sentence_processed = sentence
                if len(sentence_processed) < 2 or len(sentence_processed.split(' ')) < 2:
                    continue
                sentences_original.append(sentence)
                sentences.append(sentence_processed)

    if len(sentences) == 0:  # nothing retrieved or files not found
        return [], []

    vectorizer = TfidfVectorizer(tokenizer=tokenize)

    # if len(sentences) == 1:
    #     if len(sentence[0]) < 2:
    #         print('PASS')
    #         return [], []
    #     tfidf_matrix = vectorizer.fit_transform(sentences[0].split('\n'))
    # else:
    #     tfidf_matrix = vectorizer.fit_transform(sentences)

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        print('PASS')
        return [], []


    scores = np.zeros(len(sentences), dtype=np.float32)
    for query in queries:
        query_tfidf = vectorizer.transform([query])
        res = cosine_similarity(tfidf_matrix, query_tfidf).flatten()
        scores = np.maximum(scores, res)

    if len(scores) <= k:
        o_sort = np.argsort(-scores)
    else:
        o = np.argpartition(-scores, k)[0:k]
        o_sort = o[np.argsort(scores[o])]

    return np.take(sentences, o_sort), np.take(scores, o_sort)


def retrieve_sentences(query, doc_files, k=5):
    sentences = []
    for doc in doc_files:
        if not os.path.exists(doc):
            # logging.warn('Document not found {}'.format(doc))
            continue
        with open(doc, 'r') as f:
            doc_string = f.read()
        for sentence in nltk.sent_tokenize(doc_string):
            if len(sentence.split(' ')) < 2:
                continue
            sentences.append(sentence)

    if len(sentences) == 0:  # nothing retrieved or files not found
        return []

    vectorizer = TfidfVectorizer()  # (tokenizer=tokenize) # (stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    query_tfidf = vectorizer.transform([query])

    res = cosine_similarity(tfidf_matrix, query_tfidf).flatten()

    if len(res) <= k:
        o_sort = np.argsort(-res)
    else:
        o = np.argpartition(-res, k)[0:k]
        o_sort = o[np.argsort(res[o])]

    return np.take(sentences, o_sort)
