
import sys
from indexing import ElasticS
from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
rom_spacy = spacy.load('../Readerbench-python/models/model3')

# TODO 
# 1 change diacritics to good diacritics
# 2 check tf_idf scores

correct_diacs = {
    "ş": "ș",
	"Ş": "Ș",
	"ţ": "ț",
	"Ţ": "Ț",
}

# you need to git clone https://git.readerbench.com/ReaderBench/Readerbench-python in order for this to work
def tokenize_text_spacy(text):
    list_text = list(text)
    text = "".join([correct_diacs[c] if c in correct_diacs else c for c in list_text])
    tokens = [token for token in rom_spacy(text)]
    tokens = list(filter(lambda token: token.dep_ != 'punct', tokens))

    # with open('res.txt', 'w', encoding='utf-8') as f:
    #     for token in tokens:
    #         print(token.text, token.pos_, token.pos, token.dep_, token.lemma_, file=f)
    lemmas = [token.lemma_ for token in tokens]

    return [token.lemma_ for token in tokens]

class TextProcessings:
    def __init__(self):
        self.es = ElasticS(clean_instance=False)

    def get_data_from_index(self, index, *argv):
        all_docs = self.es.get_all_docs(size=10000, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS, verbose=False)
        data = []
        for doc in all_docs["hits"]["hits"]:
            author = {arg: doc['_source'][arg] for arg in argv}
            data.append(author)
        return data

    # get text that describes author: description and quotes (name, quotes, and description has to be in data)  
    def get_raw_description_authors(self, data):
        parsed_data = []
        for author in data:
            auth = {}
            auth[AuthorInfo.NUME.value] = author[AuthorInfo.NUME.value]
            auth[AuthorInfo.DESCRIERE.value] = "\n".join(author[AuthorInfo.DESCRIERE.value])
            # names of the quotee are not relevent
            auth[AuthorInfo.DESCRIERE.value] += "\n".join([q[1] for q in author[AuthorInfo.CITATE.value]])
            parsed_data.append(auth)
        return parsed_data

    def tf_idf_score(self, corpus):
        feature_extraction = TfidfVectorizer(sublinear_tf=True,# tf =1 + log(tf)\
                                             min_df=1,\
                                             analyzer='word',\
                                             tokenizer=tokenize_text_spacy,\
                                             norm='l2')
        # with open('res.txt', 'w', encoding='utf-8') as f:
        #     print(tokens, file=f)
        tf_idf_scores = feature_extraction.fit_transform(corpus)
        tokens = feature_extraction.get_feature_names()
        scores = tf_idf_scores.toarray().T # scores[i][j] = token i, doc j
        map_scores = {token: scores[i] for i, token in enumerate(tokens)}
        return map_scores
        # with open('res.txt', 'w', encoding='utf-8') as f:
        #     print(corpus[0], file=f)
        #     print(corpus[1], file=f)
        #     for i, token in enumerate(tokens):
        #         print(token, file=f)
        #         for j in range(1000):
        #             if scores[j][i] > 0:
        #                 print(scores[j][i], file=f)


if __name__ == "__main__":
    txt_processing = TextProcessings()
    data = txt_processing.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                              AuthorInfo.DESCRIERE.value,\
                                              AuthorInfo.CITATE.value,
                                              AuthorInfo.NUME.value)
    parsed_data = txt_processing.get_raw_description_authors(data)
    parsed_texts = [auth[AuthorInfo.DESCRIERE.value] for auth in parsed_data]
    tf_idf_scores = txt_processing.tf_idf_score(parsed_texts[0:2])

    with open('res.txt', 'w', encoding='utf-8') as f:
        print(tf_idf_scores, file=f)
        
    with open('res.txt', 'w', encoding='utf-8') as f:
        for i, text in enumerate(parsed_texts[:2]):
            values = []
            scored_tokens = []
            for token in tokenize_text_spacy(text):
                if token in tf_idf_scores:
                    values.append(tf_idf_scores[token][i])
                    scored_tokens.append(token)
            values = normalize(np.float32([values]), norm='l1')

            for i, token in enumerate(scored_tokens):
                print(token, values[0][i], file=f)





    