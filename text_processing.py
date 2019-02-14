
import sys
sys.path.append("../Readerbench-python")
sys.path.append("../Readerbench-python/core")
from indexing import ElasticS
from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
rom_spacy = spacy.load('../Readerbench-python/models/model3')

# TODO 
# 1 change diacritics to good diacritics
# 2 check tf_idf scores

# you need to git clone https://git.readerbench.com/ReaderBench/Readerbench-python in order for this to work
def tokenize_text_spacy(text):
    tokens = [token for token in rom_spacy(text)]
    tokens = list(filter(lambda token: token.dep_ != 'punct', tokens))
    # with open('res.txt', 'w', encoding='utf-8') as f:
    #     for token in tokens:
    #         print(token.text, token.pos_, token.pos, token.dep_, token.lemma_, file=f)
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
                                             min_df=3,\
                                             analyzer='word',\
                                             tokenizer=tokenize_text_spacy)
        # with open('res.txt', 'w', encoding='utf-8') as f:
        #     print(tokens, file=f)
        tf_idf_scores = feature_extraction.fit_transform(corpus)
        tokens = feature_extraction.get_feature_names()
        scores = tf_idf_scores.toarray()

        with open('res.txt', 'w', encoding='utf-8') as f:
            print(corpus[0], file=f)
            print(corpus[1], file=f)
            for i, token in enumerate(tokens):
                print(token, file=f)
                print(scores[0][i], file=f)
                print(scores[1][i], file=f)
                print(scores[2][i], file=f)
                print(scores[3][i], file=f)
                print(scores[4][i], file=f)
                print(scores[5][i], file=f)
                print(scores[6][i], file=f)
                print(scores[7][i], file=f)
                print(scores[8][i], file=f)
                print(scores[9][i], file=f)
                print(scores[10][i], file=f)
                print(scores[11][i], file=f)


if __name__ == "__main__":
    txt_processing = TextProcessings()
    data = txt_processing.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                              AuthorInfo.DESCRIERE.value,\
                                              AuthorInfo.CITATE.value,
                                              AuthorInfo.NUME.value)
    parsed_data = txt_processing.get_raw_description_authors(data)
    parsed_texts = [auth[AuthorInfo.DESCRIERE.value] for auth in parsed_data]
    #tokens = txt_processing.tokenize_text_spacy(text)
    # with open('res.txt', 'w', encoding='utf-8') as f:
    #     print(parsed_texts[0:2], file=f)
    txt_processing.tf_idf_score(parsed_texts[0:200])




    