# coding=UTF8
from pyquery import PyQuery as pq
from os import listdir
from os.path import isfile, join
import nltk
import re
nltk.download('punkt')

import re
from utils import Selectors, AuthorInfo, PublicationInfo

class HtmlParser:
    min_words = 500
    max_words = 0
    key_words_publications = \
            ['ASOCIAŢIA', 'LITERARĂ', 'ROMÂNIEI', 'CULTURALĂ', 'ROMÂNEASCĂ', \
            'literatura', 'română', 'poporului', 'român', 'AsociaŢiunea',\
            'literaturii', 'române', 'CRONICA', 'DOMNIEI', 'Istoria', 'Descrierea',\
            'românii', 'CRITICĂ', 'ACADEMIA', 'DESIDERIE', 'românilor', 'ARBOROASA', 'CERCUL',\
            'LITERAR', 'CÂNTECE', 'CĂRŢI', 'BASM', 'poveste', 'BAROC', 'BALCANISM',\
            'DOUĂMIISM', 'COLINDĂ', 'COMPARATISM', 'DESCÂNTEC', 'CRITERION', 'CALENDAR',\
            'AUTOBIOGRAFIE', 'AVANGARDĂ', 'CONTEMPORANUL', 'BALADĂ', 'BIOGRAFIE', 'BIZANTINISM',\
            'CORESPONDENȚĂ', 'DOUĂSPREZECE', 'ROMANIA', 'DADAISM', 'CONVORBIRI',\
            'LITERARE', 'CHIRALINA', 'ARCHIRIE', 'CODICELE', 'ALEXANDRIA']

    def __init__(self):
        self.authors_info = []
        self.pubs_info = []

        self.allowed_classes = [sel.value for sel in Selectors]
        for i in range(len(HtmlParser.key_words_publications)):
            HtmlParser.key_words_publications[i] = HtmlParser.key_words_publications[i].lower()
    
    def parse(self, path_files):
        html_files = [f for f in listdir(path_files) if isfile(join(path_files, f))]
        for html_file in html_files:
            self.parse_file(join(path_files, html_file))
        return self.authors_info, self.pubs_info

    def parse_file(self, html_file):

        with open(html_file, 'r', encoding='utf-8') as f:
            s = f.read()
        p = pq(s)

        items = p('body > ._idGenObjectStyleOverride-1').children()
        last_class = 'None'
        index = 0

        while index < len(items):
            current_class = items.eq(index).attr('class').split(' ')
            item = items.eq(index)
            current_class = current_class[0]

            if current_class in self.allowed_classes:
                # big author
                if current_class == Selectors.CLASS_AUTOR_BIG.value:

                    last_class = current_class
                    name_words, short_description = self.get_name_and_short_description_big_author(item)

                    # failed to extract name
                    if name_words == None:
                        index += 1
                        if len(item.text()) == 0: # not a real section, just new big paragraph, skip section
                            last_class = None
                        continue
                    # no publications for big authors
                    index, author_info, last_class = self.parse_big_author(\
                        items, index + 1, name_words, short_description=short_description)
                    index -= 1
                    self.authors_info.append(author_info)

                # small author
                elif (current_class == Selectors.CLASS_AUTOR_SMALL.value or\
                    current_class == Selectors.CLASS_PUBLICATIE.value) and\
                    last_class != Selectors.CLASS_AUTOR_BIG.value:
                    last_class = current_class
                    name_words, short_description = self.get_name_and_short_description_author(item)
                    # failed to extract name
                    if name_words == None:
                        index += 1
                        continue
                    # publication or something else
                    is_pub = self.is_publication(name_words)
                    
                    if is_pub == True or current_class == Selectors.CLASS_PUBLICATIE.value:
                        index, pub_info, last_class = self.parse_publication(items, index, name_words)
                        index -= 1
                        self.pubs_info.append(pub_info)
                    else:
                        # start parsing an author
                        index, author_info, last_class = self.parse_author(items, index, name_words, short_description)
                        index -= 1
                        self.authors_info.append(author_info)
                else:
                    last_class = current_class
            index += 1

    def get_name_and_short_description_big_author(self, item):
        name_words = nltk.word_tokenize(item('.Titlu-Nume-H').text())

        if len(name_words) == 0:
            children = item('div > div .Heuristica-PRIMUL').children()
            name_words = children.eq(0).text() + " " + children.eq(1).text()
            name_words = nltk.word_tokenize(name_words)
            if len(name_words) != 0:
                short_description = item.text().replace('\n', ' ')
            else:
                return None, None
        else:
            short_description = item('.Titlu-mic-DATE').text()
        return name_words, short_description

    def get_name_and_short_description_author(self, item):
        # extract name of the author, as good as possible
        name_text = item('.titlu-12').text().replace(",", " ")
        name_words = nltk.word_tokenize(name_text)
            
        if len(name_words) == 1:
            name_text = item.children().eq(1).text().replace(",", " ")
            name_words += nltk.word_tokenize(name_text)
        # todo get short description
        sent_taken = 0
        short_description = ""
        
        while len(nltk.word_tokenize(short_description)) < 7 or\
             len(short_description) < 30 or\
            short_description[-1] != '.':
            short_description += nltk.sent_tokenize(item.text())[sent_taken]
            sent_taken += 1

        if len(name_words) <= 1:
            return None, None
        return name_words, short_description

    def is_publication(self, name_words):
        for word in name_words:
            if word.lower() in HtmlParser.key_words_publications:
                return True
        return False

    def parse_author(self, items, index, name_words, short_description=None):
        
        author_info = {}
        author_info[AuthorInfo.DESCRIERE.value] = [items.eq(index).text()]
        author_info[AuthorInfo.NUMAR_CUVINTE.value] = str(len(items.eq(index).text().split()))

        # put name
        for i in range(len(name_words)):
            name_words[i] = name_words[i].replace(" ", "")
        author_info[AuthorInfo.NUME.value] = (" ".join(name_words)).replace(",", "")
        author_info[AuthorInfo.CITATE.value] = []
        author_info[AuthorInfo.DESCRIERE_SCURTA.value] = short_description
        author_info[AuthorInfo.AUTOR_IMPORTANT.value] = "nu"

        last_quote = False
        index += 1
        current_class = 'None'
        while index < len(items):
            item = items.eq(index)
            current_class = item.attr('class').split(' ')[0]

            if current_class in self.allowed_classes:
                # description
                if current_class == Selectors.CLASS_TEXT1.value or\
                    current_class == Selectors.CLASS_TEXT2.value:
                    author_info[AuthorInfo.DESCRIERE.value].append(item.text().strip(' \n'))
                    author_info[AuthorInfo.NUMAR_CUVINTE.value] = str(int(author_info[AuthorInfo.NUMAR_CUVINTE.value]) + len(item.text().strip()))
                    last_quote = False
                # writings
                elif current_class == Selectors.CLASS_SCRIERI.value:
                    author_info[AuthorInfo.SCRIERI.value] = \
                        [writing.strip(' \n') for writing in item.text().replace("SCRIERI:", "").split(';')]
                    years_in_writing = [re.findall('(\d{4})',year) for year in author_info[AuthorInfo.SCRIERI.value]]
                    year_in_writing_simple = []
                    if any(years_in_writing):
                        for year in years_in_writing:
                            if(any(year)):
                                    year_in_writing_simple.append(int(year[-1]))
                    author_info[AuthorInfo.ANI_PUBLICARE.value] = year_in_writing_simple
                    last_quote = False
                # references
                elif current_class == Selectors.CLASS_BIBLIO.value:
                    author_info[AuthorInfo.REP_BIBLIO.value] = \
                      [ref.strip(' \n') for ref in item.text().replace("Repere bibliografice:", "").split(';')]
                    last_quote = False
                # quotes
                elif current_class == Selectors.CLASS_CITAT_TEXT.value:
                    txt_quotes = item.text().strip(' \n')
                    last_quote = True
                elif current_class == Selectors.CLASS_CITAT_AUTOR.value and last_quote == True:
                    author_quote = item.text().strip(' \n')
                    author_info[AuthorInfo.CITATE.value].append((author_quote, txt_quotes))
                    last_quote = False
                elif current_class == Selectors.CLASS_AUTOR_SMALL.value or\
                    current_class == Selectors.CLASS_AUTOR_BIG.value or\
                    current_class == Selectors.CLASS_PUBLICATIE.value:
                    return index, author_info, current_class
            else:
                return index, author_info, current_class
            
            index += 1

        return index, author_info, current_class
    
    def parse_big_author(self, items, index, name_words, short_description=None):
        
        author_info = {}
        author_info[AuthorInfo.DESCRIERE.value] = [items.eq(index).text()]
        author_info[AuthorInfo.NUMAR_CUVINTE.value] = str(len(items.eq(index).text().split()))
        author_info[AuthorInfo.DESCRIERE_SCURTA.value] = short_description
        # put name
        for i in range(len(name_words)):
            name_words[i] = name_words[i].replace(" ", "")
        author_info[AuthorInfo.NUME.value] = (" ".join(name_words)).replace(",", "")
        author_info[AuthorInfo.CITATE.value] = []
        author_info[AuthorInfo.AUTOR_IMPORTANT.value] = "da"

        last_quote = False
        item = items.eq(index)
        current_class = 'None'

        # try:
        #     current_class = item.attr('class').split(' ')[0]
        #     if current_class == Selectors.CLASS_AUTOR_SMALL.value or\
        #         current_class == Selectors.CLASS_PUBLICATIE.value:
        #         author_info[AuthorInfo.DESCRIERE.value].append(item.text().strip(' \n'))
        #         author_info[AuthorInfo.NUMAR_CUVINTE.value] = str(int(author_info[AuthorInfo.NUMAR_CUVINTE.value]) + len(item.text().strip()))
        #     else:
        #         print('smth went wrong', current_class)
        # except:
        #     print('error')

        index += 1

        while index < len(items):
            item = items.eq(index)
            current_class = item.attr('class').split(' ')[0]

            if current_class in self.allowed_classes:
                # description
                if current_class == Selectors.CLASS_TEXT1.value or\
                    current_class == Selectors.CLASS_TEXT2.value:
                    author_info[AuthorInfo.DESCRIERE.value].append(item.text().strip(' \n'))
                    author_info[AuthorInfo.NUMAR_CUVINTE.value] = str(int(author_info[AuthorInfo.NUMAR_CUVINTE.value]) + len(item.text().strip()))
                    last_quote = False
                # writings
                elif current_class == Selectors.CLASS_SCRIERI.value:
                    author_info[AuthorInfo.SCRIERI.value] = \
                        [writing.strip(' \n') for writing in item.text().replace("SCRIERI:", "").split(';')]
                    years_in_writing = [re.findall('(\d{4})',year) for year in author_info[AuthorInfo.SCRIERI.value]]
                    year_in_writing_simple = []
                    if any(years_in_writing):
                        for year in years_in_writing:
                            if(any(year)):
                                    year_in_writing_simple.append(int(year[-1]))
                    author_info[AuthorInfo.ANI_PUBLICARE.value ] = year_in_writing_simple
                    last_quote = False
                # references
                elif current_class == Selectors.CLASS_BIBLIO.value:
                    author_info[AuthorInfo.REP_BIBLIO.value] = \
                      [ref.strip(' \n') for ref in item.text().replace("Repere bibliografice:", "").split(';')]
                    last_quote = False
                # quotes
                elif current_class == Selectors.CLASS_CITAT_TEXT.value:
                    txt_quotes = item.text().strip(' \n')
                    last_quote = True
                elif current_class == Selectors.CLASS_CITAT_AUTOR.value and last_quote == True:
                    author_quote = item.text().strip(' \n')
                    author_info[AuthorInfo.CITATE.value].append((author_quote, txt_quotes))
                    last_quote = False
                elif current_class == Selectors.CLASS_AUTOR_SMALL.value or\
                    current_class == Selectors.CLASS_AUTOR_BIG.value or\
                    current_class == Selectors.CLASS_PUBLICATIE.value:
                    return index, author_info, current_class
            else:
                return index, author_info, current_class
            
            index += 1

        return index, author_info, current_class

    def parse_publication(self, items, index, name_words):

        publication_info = {}
        publication_info[PublicationInfo.DESCRIERE.value] = [items.eq(index).text()]

        # put name
        for i in range(len(name_words)):
            name_words[i] = name_words[i].replace(" ", "")
        publication_info[PublicationInfo.NUME.value] = (" ".join(name_words)).replace(",", "")

        index += 1
        current_class = 'None'

        while index < len(items):
            item = items.eq(index)
            current_class = item.attr('class').split(' ')[0]

            if current_class in self.allowed_classes:
                # description
                if current_class == Selectors.CLASS_TEXT1.value or\
                    current_class == Selectors.CLASS_TEXT2.value:
                    publication_info[PublicationInfo.DESCRIERE.value].append(item.text().strip(' \n'))
                # writings
                elif current_class == Selectors.CLASS_AUTOR_SMALL.value or\
                    current_class == Selectors.CLASS_AUTOR_BIG.value or\
                    current_class == Selectors.CLASS_PUBLICATIE.value:
                    return index, publication_info, current_class
            else:
                return index, publication_info, current_class
            
            index += 1

        return index, publication_info, current_class


if __name__ == "__main__":
    parser = HtmlParser()
    authors, pubs = parser.parse('corpora/htmls')    
    #print(authors[0].encode("utf-8"))