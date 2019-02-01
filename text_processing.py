from indexing import ElasticS
from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
import sys
sys.path.append("../Readerbench-python/core/")
import spacy
from spacy_parser import SpacyParser
from spacy_parser import convertToPenn


class TextProcessings:
    def __init__(self):
        self.es = ElasticS(clean_instance=False)

    def prelucrate_descriptions(self):
        all_docs = es.get_all_docs(size=10000, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS, verbose=False)
        all_descriptions = [ "\n".join(doc["_source"]["description"]) for doc in all_docs["hits"]["hits"]]
        all_names = [ doc["_source"]["name"] for doc in all_docs["hits"]["hits"] ]

        rom_spacy = spacy.load('models/model3')
        d1 = rom_spacy("am mâncat")
        for x in d1:
            print(dir(x))
            print(x.lemma, x.lemma_)
            print(x.pos_)
            print(x.dep_)
        
if __name__ == "__main__":
    txt_processing = TextProcessings()
    txt_processing.prelucrate_descriptions()
