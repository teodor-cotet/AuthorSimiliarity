from utils import Selectors, AuthorInfo, PublicationInfo, Elastic
from elasticsearch import Elasticsearch
from parse_htmls import HtmlParser

class ElasticS():

    def __init__(self, clean_instance=False):
        if clean_instance == False:
            self.es = self.get_elastic_instance()
        else:
            self.es = self.get_clean_elastic_instance()

    def get_clean_elastic_instance(self):
        es = Elasticsearch([{'host': Elastic.ELASTIC_HOST.value, 'port': Elastic.ELASTIC_PORT.value}])
        print('[Elasticsearch] Connected to es server on host: {} and port: {}'.format(Elastic.ELASTIC_HOST.value, Elastic.ELASTIC_PORT.value))

        if es.indices.exists(index=Elastic.ELASTIC_INDEX_AUTHORS.value) == True:
            es.indices.delete(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=[400, 404])
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_AUTHORS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            print('[Elasticsearch] index {} created'.format(Elastic.ELASTIC_INDEX_AUTHORS.value))
        
        if es.indices.exists(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value) == True:
            es.indices.delete(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=[400, 404])
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            print('[Elasticsearch] Index {} created'.format(Elastic.ELASTIC_INDEX_PUBLICATIONS.value))
        return es

    def get_elastic_instance(self):
        es = Elasticsearch([{'host': Elastic.ELASTIC_HOST.value, 'port': Elastic.ELASTIC_PORT.value}])
        print('[Elasticsearch] Connected to es server on host: {} and port: {}'.format(Elastic.ELASTIC_HOST.value, Elastic.ELASTIC_PORT.value))

        if es.indices.exists(index=Elastic.ELASTIC_INDEX_AUTHORS.value) == True:
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_AUTHORS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_AUTHORS.value, ignore=400)
            print('[Elasticsearch] index {} created'.format(Elastic.ELASTIC_INDEX_AUTHORS.value))
        
        if es.indices.exists(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value) == True:
            es.indices.refresh(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value)
        else:
            es.indices.create(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value, ignore=400)
            print('[Elasticsearch] Index {} created'.format(Elastic.ELASTIC_INDEX_PUBLICATIONS.value))
        return es

    def index_authors(self, authors_info):
        print('[Elasticsearch] Started indexing {} in {}'\
        .format(len(authors_info), Elastic.ELASTIC_INDEX_AUTHORS))

        for author in authors_info:
            self.es.index(index=Elastic.ELASTIC_INDEX_AUTHORS.value,\
                doc_type=Elastic.ELASTIC_DOC_TYPE_AUTHOR.value, body=author)
        
        print('[Elasticsearch] Finished indexing {} authors in {}'\
        .format(len(authors_info), Elastic.ELASTIC_INDEX_AUTHORS.value))
    
    def index_publications(self, pubs_info):
        print('[Elasticsearch] Started indexing {} in {}'\
        .format(len(pubs_info), Elastic.ELASTIC_INDEX_PUBLICATIONS))

        for publication in pubs_info:
            self.es.index(index=Elastic.ELASTIC_INDEX_PUBLICATIONS.value,\
                doc_type=Elastic.ELASTIC_DOC_TYPE_PUBLICATION.value, body=publication)
        
        print('[Elasticsearch] Finished indexing {} publications in {}'\
        .format(len(pubs_info), Elastic.ELASTIC_INDEX_PUBLICATIONS.value))

if __name__ == "__main__":
    es = ElasticS(clean_instance=True)
    parser = HtmlParser()
    authors_info, pubs_info = parser.parse('corpora/htmls')
    es.index_authors(authors_info)
    es.index_publications(pubs_info)
    # for x in authors_info[1]:
    #     print(x, authors_info[1][x])



