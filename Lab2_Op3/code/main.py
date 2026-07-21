import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

def get_data(n_features=1000):
    categories = [
        'comp.graphics', 'sci.med', 
        'talk.politics.misc', 'soc.religion.christian'
    ]
    
    # categories = [
    #     'sci.crypt',
    #     'sci.space',
    #     'rec.sport.baseball',
    #     'misc.forsale'
    # ]

    dataset = fetch_20newsgroups(
        subset='all', categories=categories, shuffle=True, 
        random_state=42, remove=('headers', 'footers', 'quotes')
    )
    
    valid_indices = [i for i, text in enumerate(dataset.data) if len(text.strip()) > 50]
    raw_texts = [dataset.data[i] for i in valid_indices]
    targets = [dataset.target[i] for i in valid_indices]
    
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, 
                                 max_features=n_features, stop_words='english')
    
    tfidf_matrix = vectorizer.fit_transform(raw_texts)
    V = tfidf_matrix.toarray().T
    feature_names = vectorizer.get_feature_names_out()
    
    return V, feature_names, dataset.target_names, raw_texts, targets

class CustomNMF:
    def __init__(self, n_components, max_iter=200, tol=1e-4):
        self.n_components = n_components 
        self.max_iter = max_iter         
        self.tol = tol                   
        self.W = None
        self.H = None
        
    def fit(self, V):
        m, n = V.shape
        k = self.n_components
        
        np.random.seed(42)
        self.W = np.random.rand(m, k)
        self.H = np.random.rand(k, n)
        
        epsilon = 1e-9
        
        for i in range(self.max_iter):
            W_prev = self.W.copy()
            H_prev = self.H.copy()
            
            WtV = self.W.T @ V
            WtWH = self.W.T @ self.W @ self.H + epsilon
            self.H = self.H * (WtV / WtWH)
            
            VHt = V @ self.H.T
            WHHt = self.W @ self.H @ self.H.T + epsilon
            self.W = self.W * (VHt / WHHt)
            
            if np.linalg.norm(self.W - W_prev, 'fro') < self.tol and \
               np.linalg.norm(self.H - H_prev, 'fro') < self.tol:
                break
                
        return self.W, self.H

def run_pipeline(n_topics=4, n_features=1000):
    V, feature_names, target_names, raw_texts, targets = get_data(n_features)
    
    custom_nmf = CustomNMF(n_components=n_topics, max_iter=200)
    W_nmf, H_nmf = custom_nmf.fit(V)
    nmf_doc_topic = H_nmf.T 
    nmf_topic_word = W_nmf.T
    
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    svd_doc_topic = svd.fit_transform(V.T) 
    svd_topic_word = svd.components_
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tf_matrix = tf_vectorizer.fit_transform(raw_texts)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, random_state=42)
    lda_doc_topic = lda.fit_transform(tf_matrix) 
    lda_topic_word = lda.components_
    
    results = {
        'NMF': {'topic_word': nmf_topic_word, 'doc_topic': nmf_doc_topic, 'features': feature_names},
        'SVD': {'topic_word': svd_topic_word, 'doc_topic': svd_doc_topic, 'features': feature_names},
        'LDA': {'topic_word': lda_topic_word, 'doc_topic': lda_doc_topic, 'features': tf_feature_names}
    }
    
    return raw_texts, targets, target_names, results