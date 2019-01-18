from enum import Enum

class Selectors(Enum):
    CLASS_AUTOR_BIG = '_idGenObjectLayout-1'
    CLASS_AUTOR_SMALL = 'Heuristica-PRIMUL'
    # reviste etc
    CLASS_PUBLICATIE = 'Heuristica_PRIMUL_Semnat'
    
    CLASS_TEXT1 = 'Heuristica-BASE'
    CLASS_TEXT2 = 'Normal'

    CLASS_SCRIERI = 'Heuristica___mici'
    CLASS_BIBLIO = 'Heuristica___m_RPR'
    CLASS_CITAT_TEXT = 'Heuristica_CITAT'
    CLASS_CITAT_AUTOR = 'Heuristica_CITAT_semnat'

class AuthorInfo(Enum):
    NUME = 'name'
    PSEUDONIM = 'pseudonim'
    AN_NASTERE = 'birth_date'
    AN_DECES = 'deceased_date'
    FUNCTII = 'titles'
    DESCRIERE = 'description'
    DESCRIERE_SCURTA = 'short_description'
    CITATE = 'quotes'
    SCRIERI = 'writings'
    ANI_PUBLICARE = 'publishing_years'
    REP_BIBLIO = 'references'
    AUTOR_IMPORTANT = 'important_author'
    NUMAR_CUVINTE = 'words_count'
    IMPORTANTA = 'importance'

class PublicationInfo(Enum):
    NUME = 'name'
    DESCRIERE = 'description'

class Elastic(Enum):
    ELASTIC_PORT = 9200
    ELASTIC_HOST = "localhost"
    ELASTIC_INDEX_AUTHORS = 'index-authors'
    ELASTIC_INDEX_PUBLICATIONS = 'index-publications'
    ELASTIC_DOC_TYPE_AUTHOR = 'author'
    ELASTIC_DOC_TYPE_PUBLICATION = 'publication'