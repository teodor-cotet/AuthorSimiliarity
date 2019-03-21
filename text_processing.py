import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import spacy
from gensim.models.wrappers import FastText as FastTextWrapper
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.cluster import AffinityPropagation
from indexing import ElasticS
from utils import AuthorInfo, Elastic, PublicationInfo, Selectors

# readme
# txt_processing.compute_word_embeddings_authors() - to store in AUTHORS_EMBEDDINGS_FILE the word embeddings
# txt_processing.compute_pca() - to compute pca using word embeddings stored previously
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
        tokens = list(filter(lambda token: token.is_stop == False , tokens))
        return [token.lemma_ for token in tokens]

    def get_data_from_index(self, index, *argv) -> List[Dict[str, Any]]:
        all_docs = self.es.get_all_docs(size=10000, index_to_search=Elastic.ELASTIC_INDEX_AUTHORS, verbose=False)
        data = []
        for doc in all_docs["hits"]["hits"]:
            author = {arg: doc['_source'][arg] if arg in doc['_source'] else None for arg in argv}
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
        feature_extraction = TfidfVectorizer(sublinear_tf=True,# tf = s1 + log(tf)\
                                             min_df=2,\
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
        with open('300_authors_embeddings.txt', 'w', encoding='utf-8') as f:
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
                # print(weights, file=f)
                # print(weights[0])
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
    
    def get_word_embeddings_authors_file(self):
        return pickle.load(open(TextProcessings.AUTHORS_EMBEDDINGS_FILE, "rb"))

    def compute_pca(self, comp=17):
        authors = txt_processing.get_word_embeddings_authors_file()
        authors_vecs = np.float32([auth[1] for auth in authors])
        authors_names = [auth[0] for auth in authors]
        # normalize 
        st_x = StandardScaler().fit_transform(authors_vecs)
        # pca on 10 comp
        pca = PCA(n_components=comp, copy=True, whiten=True)
        # projected vectors
        projected = pca.fit_transform(st_x)
        print('variance per comp')
        for rat in pca.explained_variance_ratio_:
            print(rat)
        #print(pca.explained_variance_ratio_)
        return authors_names, projected
    
    def wirte_file_tensorboard(self, authors_names, projected):
        assert len(projected) == len(authors_names), 'assert error lengths'
        dim = 300
        with open("300_authors.model", "w", encoding='utf-8') as f:
            f.write("{} {}\n".format(len(projected), dim))
            for i, _ in enumerate(projected):
                d = (projected[i], authors_names[i])
                concat_name = "_".join(d[1].split(' '))
                f.write("{} ".format(concat_name))
                for i, v in enumerate(d[0][:dim]):
                    if i == len(d[0][:dim]) - 1:
                        f.write("{}".format(v))
                    else:
                        f.write("{} ".format(v))
                f.write("\n")
    
    def compute_jaccard_distance(self, l1, l2):
        uni = len(set(l1).union(set(l2)))
        inter = len(set(l1).intersection(set(l2)))
        return inter / uni if uni > 0 else 0

    def get_references_dist_matrix(self):
        authors_ref = self.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                                    AuthorInfo.NUME.value,\
                                                    AuthorInfo.REF_AUTHORS.value)
        dist_matrix = {}
        with open("res.txt", "w", encoding='utf-8') as f:
            for auth1 in authors_ref:
                name_auth1 = auth1[AuthorInfo.NUME.value]
                ref_list1 = auth1[AuthorInfo.REF_AUTHORS.value]
                if ref_list1 is None:
                    ref_list1 = []
                dist_matrix[name_auth1] = {}
                for auth2 in authors_ref:
                    name_auth2 = auth2[AuthorInfo.NUME.value]
                    ref_list2 = auth2[AuthorInfo.REF_AUTHORS.value]
                    if ref_list2 is None:
                        ref_list2 = []
                    dist_matrix[name_auth1][name_auth2] = self.compute_jaccard_distance(ref_list1, ref_list2)
        return dist_matrix

    def substitute_professions_list(self, prof_list):
        subst_list = []
        for profession in prof_list:
            if profession == "etnolog":
                subst_list.append(0)
            if 'critic' == profession:
                subst_list.append(1)
            if profession == 'autor dramatic' or profession == 'autoare' or profession == 'prozator' or profession == 'prozatoare' or 'eseist' == profession:
                subst_list.append(2)
            if 'folclorist' == profession or profession == 'culegător de folclor':
                subst_list.append(3)
            if profession == 'traducător' or profession == 'traducătoare':
                subst_list.append(4)
            if 'poet' == profession or profession == 'versificator' or profession == 'versificatoare' or profession == 'autor de versuri':
                subst_list.append(5)
            if 'dramaturg' == profession:
                subst_list.append(6)
            if 'slavist' == profession:
                subst_list.append(7)
            if 'jurnalist' == profession or 'gazetar' == profession or 'cronicar' == profession or 'publicist' == profession or 'ziarist' == profession:
                subst_list.append(8)
            if profession == "românist german":
                subst_list.append(9)
            if 'comparatist' == profession:
                subst_list.append(10)
            if profession == "editor":
                subst_list.append(11)
            if 'memorialist' == profession or 'diarist' == profession:
                subst_list.append(12)
            if 'clasicist' == profession:
                subst_list.append(13)
            if 'stilistician' == profession:
                subst_list.append(14)
            if 'filolog' == profession:
                subst_list.append(15)
            if 'cărturar' == profession:
                subst_list.append(16)
            if 'anglist' == profession:
                subst_list.append(17)
            if 'imnograf' == profession:
                subst_list.append(18)
            if 'istoric' == profession:
                subst_list.append(19)
            if 'estetician' == profession:
                subst_list.append(20)
            if 'teatrolog' == profession or 'critic de teatru' == profession:
                subst_list.append(21)
            if 'teoretician' == profession:
                subst_list.append(22)
            if 'povestitor' == profession:
                subst_list.append(23)
            if 'eminescolog' == profession:
                subst_list.append(24)
        return subst_list

    def get_functions_dist_matrix(self):
        authors_prof = self.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                                    AuthorInfo.NUME.value,\
                                                    AuthorInfo.PROFESII.value)
                                                    
        dist_matrix = {}
        with open("res.txt", "w", encoding='utf-8') as f:
            for auth1 in authors_prof:
                name_auth1 = auth1[AuthorInfo.NUME.value]
                prof_list1 = auth1[AuthorInfo.PROFESII.value]
                if prof_list1 is None:
                    prof_list1 = []
                prof_list1 = self.substitute_professions_list(prof_list1)
                dist_matrix[name_auth1] = {}
                for auth2 in authors_prof:
                    name_auth2 = auth2[AuthorInfo.NUME.value]
                    prof_list2 = auth2[AuthorInfo.PROFESII.value]
                    if prof_list2 is None:
                        prof_list2 = []
                    prof_list2 = self.substitute_professions_list(prof_list2)
                    dist_matrix[name_auth1][name_auth2] = self.compute_jaccard_distance(prof_list1, prof_list2)
        return dist_matrix

    def min_max_years_scaling(self, years):
        min_max_scaler = MinMaxScaler()
        years = np.float32(years)
        years = years.reshape((len(years), 1))
        years_scaled = min_max_scaler.fit_transform(years)
        years_scaled = [y[0] for y in years_scaled]
        return years_scaled

    def fill_missing_pub_year(self, authors: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, any]]:
        train_names, train_projs, train_pub_years = [], [], []
        test_names, test_projs = [], []
        for auth, d in authors.items():
            if d['year'] is None:
                test_projs.append(d['proj'])
                test_names.append(auth)
            else:
                train_projs.append(d['proj'])
                train_pub_years.append(d['year'])
                train_names.append(auth)
            
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(train_projs, train_pub_years)

        for i, proj in enumerate(test_projs):
            authors[test_names[i]]['year']  = knn.predict([proj])[0]

    def get_features_knn(self):
        authors_names, projected = self.compute_pca()
        authors = {}
        authors_proj = {}
        for author, proj in zip(authors_names, projected):
            authors_proj[author] = proj
        authors_year = self.get_data_from_index(Elastic.ELASTIC_INDEX_AUTHORS.value,\
                                                    AuthorInfo.MEDIE_ANI_PUBLICARE.value,\
                                                    AuthorInfo.NUME.value)
        for author_year_dic in authors_year:
            author_name = author_year_dic[AuthorInfo.NUME.value]
            author_year = author_year_dic[AuthorInfo.MEDIE_ANI_PUBLICARE.value]
            authors[author_name] = {}
            authors[author_name]['year'] = author_year

            if author_name in authors_proj:
                authors[author_name]['proj'] = authors_proj[author_name]

        self.fill_missing_pub_year(authors)
        self.years, self.names, self.projs = [], [], []
        with open("res.txt", "w", encoding='utf-8') as f:
            for auth, d in authors.items():
                year = d['year']
                proj = d['proj']
                self.years.append(float(year))
                self.projs.append(proj)
                self.names.append(auth)
        self.indices = [i for i in range(len(self.years))]
        self.years = self.min_max_years_scaling(self.years)

        #datapoints = [y + p for y, p in zip(years, projs)] 
        #fclust = fclusterdata(datapoints, 1.0, metric=mydist)
        self.dist_functions = self.get_functions_dist_matrix()
        self.dist_refs = self.get_references_dist_matrix()

    def compute_distance(self, p1, p2):
        p1 = int(p1)
        p2 = int(p2)
        name1 = self.names[p1]
        name2 = self.names[p2]
        dist_funct = self.dist_functions[name1][name2] # [0, 1]
        dist_refs = self.dist_refs[name1][name2] # [0, 1]
        dist_year = abs(self.years[p1] - self.years[p2])
        proj1, proj2 = self.projs[p1], self.projs[p2]
        cos_dist = cosine_distances(np.asarray([proj1]), np.asarray([proj2]))[0][0]
        return (1 - dist_funct) + (1 - dist_refs) + dist_year + len(proj1) * cos_dist / 2

    def find_nn(self):
        self.get_features_knn()
        with open("all_nn_without_stop_words.txt", "w", encoding='utf-8') as f:
            self.get_features_knn()
            samples = [[i] for i in self.indices]
            for i in self.indices:
                dists = []
                for j in self.indices:
                    dists.append((self.names[j], self.compute_distance(i, j)))
                dists = sorted(dists, key = lambda x: x[1])
                for (nj, d) in dists:
                    print(self.names[i], nj, d/20, file=f)

    def clustering(self):
        with open("clusters_without_stopwords.txt", "w", encoding='utf-8') as f:
            self.get_features_knn()
            n = len(self.indices)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i][j] = self.compute_distance(i, j)
            aff_prop  = AffinityPropagation(affinity='precomputed')
            aff_prop.fit(dist_matrix)
            nr_clusters = max(aff_prop.labels_) + 1
            for i in range(nr_clusters):
                print('cluster ', i, file=f)
                for index, j in enumerate(aff_prop.labels_):
                    if j == i:
                        print(self.names[index], file=f)

if __name__ == "__main__":
    txt_processing = TextProcessings()
    # txt_processing.compute_word_embeddings_authors() - to store in AUTHORS_EMBEDDINGS_FILE the word embeddings
    # txt_processing.compute_pca() - to compute pca using word embeddings stored previously
    # authors = txt_processing.get_word_embeddings_authors_file()
    txt_processing.find_nn()
    
# a custom function that just computes Euclidean distance
