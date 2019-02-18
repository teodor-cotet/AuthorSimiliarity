import sys
from indexing import ElasticS
from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import Dict, Tuple, List, Any
from gensim.models.wrappers import FastText as FastTextWrapper
import pickle

class TextProcessings:

    FAST_TEXT_PATH = "fastText/cc.ro.300"
    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
    REPLACE_CHARS = {
        "/": ""
    }
    AUTHORS_EMBEDDINGS_FILE = 'embeddings_authors'
    WORD_DIM = 300

    def __init__(self):
        self.es = ElasticS(clean_instance=False)
        self.model_embeddings = None # from fasttext
        self.rom_spacy = spacy.load('../Readerbench-python/models/model3')

    # you need to git clone https://git.readerbench.com/ReaderBench/Readerbench-python in order for this to work
    def tokenize_text_spacy(self, text: str) -> List[str]:
        list_text = list(text)

        # some cleaning correct diacritics + eliminate \
        text = "".join([TextProcessings.CORRECT_DIACS[c] if c in TextProcessings.CORRECT_DIACS else c for c in list_text])
        list_text = list(text)
        text = "".join([TextProcessings.REPLACE_CHARS[c] if c in TextProcessings.REPLACE_CHARS else c for c in list_text])

        tokens = [token for token in self.rom_spacy(text)]
        tokens = list(filter(lambda token: token.dep_ != 'punct', tokens))
        return [token.lemma_ for token in tokens]

    def get_data_from_index(self, index, *argv) -> Dict[str, Any]:
        all_docs = self.es.get_all_docs(size=10000, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS, verbose=False)
        data = []
        for doc in all_docs["hits"]["hits"]:
            author = {arg: doc['_source'][arg] for arg in argv}
            data.append(author)
        return data

    # get text that describes author: description and quotes (name, quotes, and description has to be in data)  
    def get_raw_description_authors(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        parsed_data = []
        for author in data:
            auth = {}
            auth[AuthorInfo.NUME.value] = author[AuthorInfo.NUME.value]
            auth[AuthorInfo.DESCRIERE.value] = "\n".join(author[AuthorInfo.DESCRIERE.value])
            # names of the quotee are not relevent
            auth[AuthorInfo.DESCRIERE.value] += "\n".join([q[1] for q in author[AuthorInfo.CITATE.value]])
            parsed_data.append(auth)
        return parsed_data

    def get_tf_idf_score(self, corpus: List[str]) -> Dict[str, List[float]]:
        print('get tf_idf')
        feature_extraction = TfidfVectorizer(sublinear_tf=True,# tf =1 + log(tf)\
                                             min_df=1,\
                                             analyzer='word',\
                                             tokenizer=self.tokenize_text_spacy,\
                                             norm='l2')
        tf_idf_scores = feature_extraction.fit_transform(corpus)
        tokens = feature_extraction.get_feature_names()
        print('done tf_idf')
        scores = tf_idf_scores.toarray().T # scores[i][j] = token i, doc j
        map_scores = {token: scores[i] for i, token in enumerate(tokens)}
        return map_scores

    def get_authors_pondered_tokens(self, tf_idf_scores: Dict[str, List[float]],\
         parsed_texts: List[str], parsed_names: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        authors_pondered_tokens = {}
        for i, (text, name) in enumerate(zip(parsed_texts, parsed_names)):
            weights = []
            scored_tokens = []
            # works even if the tokenizer is not deterministic
            for token in self.tokenize_text_spacy(text):
                if token in tf_idf_scores:
                    weights.append(tf_idf_scores[token][i])
                    scored_tokens.append(token)
            authors_pondered_tokens[name] = []

            for token, weight in zip(scored_tokens, weights):
                authors_pondered_tokens[name].append((token, weight))
        return authors_pondered_tokens

    def get_embeddings_authors(self, authors_pondered_tokens: Dict[str, List[Tuple[str, float]]]):
        self.model_embeddings = FastTextWrapper.load_fasttext_format(TextProcessings.FAST_TEXT_PATH)
        authors_embeddings = []
        print('computing embeddings')
        with open('res.txt', 'w', encoding='utf-8') as f:
            for name, tokens_weights in authors_pondered_tokens.items():
                weighted_avg_author = np.float32([0] * TextProcessings.WORD_DIM)
                print(name, file=f)
                weights, tokens = [], []
                unique_tokens = set()
                for (token, weight) in tokens_weights:
                    if token in self.model_embeddings.wv.vocab and token not in unique_tokens:
                        weights.append(weight)
                        tokens.append(token)
                        unique_tokens.add(token)
                        
                # normalize weights s.t. is a prob distribution
                weights = normalize(np.float32([weights]), norm='l1')[0]
                print(weights, file=f)
                print(weights[0])
                for i, token in enumerate(tokens):
                    weighted_avg_author += weights[i] * self.model_embeddings.wv[token]
                    print(token, weights[i] , file=f)
                authors_embeddings.append((name, weighted_avg_author))
            pickle.dump(authors_embeddings, open(TextProcessings.AUTHORS_EMBEDDINGS_FILE, "wb"))
    
    def compute_word_embeddings_authors(self):
        data = self.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                                AuthorInfo.DESCRIERE.value,\
                                                AuthorInfo.CITATE.value,
                                                AuthorInfo.NUME.value)
        parsed_data = self.get_raw_description_authors(data)
        parsed_texts = [auth[AuthorInfo.DESCRIERE.value] for auth in parsed_data]
        parsed_names = [auth[AuthorInfo.NUME.value] for auth in parsed_data]
        tf_idf_scores = self.get_tf_idf_score(parsed_texts)
        authors_pondered_tokens = self.get_authors_pondered_tokens(tf_idf_scores, parsed_texts, parsed_names)
        self.get_embeddings_authors(authors_pondered_tokens)

if __name__ == "__main__":
    txt_processing = TextProcessings()
    txt_processing.compute_word_embeddings_authors()
    # data = txt_processing.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
    #                                           AuthorInfo.DESCRIERE.value,\
    #                                           AuthorInfo.CITATE.value,
    #                                           AuthorInfo.NUME.value)
    # parsed_data = txt_processing.get_raw_description_authors(data)
    # parsed_texts = [auth[AuthorInfo.DESCRIERE.value] for auth in parsed_data]
    # parsed_names = [auth[AuthorInfo.NUME.value] for auth in parsed_data]
    # tf_idf_scores = txt_processing.get_tf_idf_score(parsed_texts)
    # authors_pondered_tokens = txt_processing.get_authors_pondered_tokens(tf_idf_scores, parsed_texts)
    # txt_processing.get_embeddings_authors(authors_pondered_tokens)

    # with open('res.txt', 'w', encoding='utf-8') as f:
    #     datapoints = pickle.load(open(TextProcessings.AUTHORS_EMBEDDINGS_FILE, "rb"))
    #     for (name, w) in datapoints:
    #         print(name, w, file=f)
    
    

   





    