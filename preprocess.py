import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from nltk import download, ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean
from vncorenlp import VnCoreNLP

# Xác định device là GPU nếu có, nếu không thì là CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình VnCoreNLP
vncorenlp = VnCoreNLP(r"C:\Users\nguye\Downloads\VnCoreNLP-master\VnCoreNLP-master\VnCoreNLP-1.2.jar", 
                      annotators="wseg,pos,ner", 
                      port=62328)

# Tải các gói cần thiết của NLTK
download('punkt')

# Hàm tải từ dừng từ file
def load_stopwords(directory):
    stopwords = set()
    for filename in ['vietnamese-stopwords.txt', 'vietnamese-stopwords-dash.txt']:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords.update(line.strip().replace('_', ' ') for line in f)
        else:
            print(f"File not found: {file_path}")
    return stopwords

# Hàm tải từ đồng nghĩa từ các file
def load_synonyms_from_files(directory):
    synonyms_dict = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split(',')
                    base_word = words[0].strip()
                    synonyms = set(word.strip() for word in words[1:])
                    synonyms_dict.setdefault(base_word, set()).update(synonyms)
    return synonyms_dict

# Hàm tải dữ liệu JSON
def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

# Hàm lưu dữ liệu JSON
def convert_floats(data):
    if isinstance(data, dict):
        return {k: convert_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_floats(i) for i in data]
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

# Tải PhoBERT model và tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

# Hàm trích xuất embeddings ngữ cảnh từ PhoBERT
def get_phobert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Hàm tính cosine similarity
def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Hàm tính Euclidean Distance
def compute_euclidean_distance(vec1, vec2):
    try:
        dist = euclidean(vec1, vec2)
        return 1 / (1 + dist)  # Đảo ngược khoảng cách để phù hợp với cách tính độ tương đồng
    except ZeroDivisionError:
        return 1.0

# Hàm tiền xử lý văn bản với n-grams, POS tagging, và NER
def preprocess_text_combined(text, synonyms_dict, stopwords, tfidf_vectorizer, max_length=1000):
    def split_text(text, max_length):
        sentences = text.split('. ')
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += ' ' + sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def preprocess_chunk(chunk, synonyms_dict, stopwords, tfidf_vectorizer):
        annotated = vncorenlp.annotate(chunk)
        tokens_with_pos = [(word['form'], word['posTag']) for sent in annotated['sentences'] for word in sent]
        
        processed_tokens = []
        for word, pos in tokens_with_pos:
            if word.lower() not in stopwords:
                synonym = synonyms_dict.get(word, word)
                processed_tokens.append(f"{synonym}_{pos}")
        
        bigrams = list(ngrams(processed_tokens, 2))
        trigrams = list(ngrams(processed_tokens, 3))
        
        all_grams = processed_tokens + [f"{w1}_{w2}" for w1, w2 in bigrams] + [f"{w1}_{w2}_{w3}" for w1, w2, w3 in trigrams]
        
        processed_chunk = ' '.join(all_grams)
        tfidf_vector = tfidf_vectorizer.transform([processed_chunk]).toarray()
        return processed_chunk, tfidf_vector

    text_chunks = split_text(text, max_length)
    all_named_entities = []
    all_tokens = []
    all_tfidf_vectors = []

    for chunk in text_chunks:
        try:
            named_entities = vncorenlp.ner(chunk)[0]
        except Exception as e:
            print(f"Error during NER processing: {e}")
            named_entities = []

        all_named_entities.extend(named_entities)
        processed_chunk, tfidf_vector = preprocess_chunk(chunk, synonyms_dict, stopwords, tfidf_vectorizer)
        all_tokens.extend(processed_chunk.split())
        all_tfidf_vectors.append(tfidf_vector.flatten())

    embeddings = get_phobert_embeddings(text)
    avg_tfidf_vector = np.mean(all_tfidf_vectors, axis=0) if all_tfidf_vectors else np.zeros(tfidf_vectorizer.get_feature_names_out().shape[0])

    return {
        'tokens': all_tokens,
        'embeddings': embeddings,
        'tfidf': avg_tfidf_vector
    }, all_named_entities

def preprocess_data(data, legal_passages, synonyms_dict, stopwords):
    legal_dict = {law['id']: {article['id']: article['text']
                              for article in law.get('articles', [])} 
                  for law in legal_passages}

    # Tạo một danh sách các văn bản từ dữ liệu và văn bản pháp lý
    all_texts = []
    for item in data:
        all_texts.append(item.get('statement', ''))
        for passage in item.get('legal_passages', []):
            law_id = passage.get('law_id')
            article_id = passage.get('article_id')
            if law_id in legal_dict and article_id in legal_dict[law_id]:
                passage_text = legal_dict[law_id][article_id]
                all_texts.append(passage_text)

    # Khởi tạo TF-IDF vectorizer với tất cả các văn bản và n-grams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3)).fit(all_texts)
    
    processed_data = [preprocess_item(item, legal_dict, synonyms_dict, stopwords, tfidf_vectorizer) 
                      for item in data]
    
    return processed_data

def preprocess_item(item, legal_dict, synonyms_dict, stopwords, tfidf_vectorizer):
    statement_text = item.get('statement', '')
    processed_statement, _ = preprocess_text_combined(statement_text, synonyms_dict, stopwords, tfidf_vectorizer)

    legal_passages_embeddings = []
    legal_passages_tfidf = []
    for passage in item.get('legal_passages', []):
        law_id = passage.get('law_id')
        article_id = passage.get('article_id')
        if law_id in legal_dict and article_id in legal_dict[law_id]:
            passage_text = legal_dict[law_id][article_id]
            processed_passage, _ = preprocess_text_combined(passage_text, synonyms_dict, stopwords, tfidf_vectorizer)
            legal_passages_embeddings.append(processed_passage['embeddings'])
            legal_passages_tfidf.append(processed_passage['tfidf'])

    statement_features = processed_statement['embeddings']
    statement_tfidf = processed_statement['tfidf']

    similarities = []
    for passage_features, passage_tfidf in zip(legal_passages_embeddings, legal_passages_tfidf):
        cosine_sim = compute_cosine_similarity(statement_features, passage_features)
        euclidean_dist = compute_euclidean_distance(statement_features, passage_features)
        tfidf_cosine_sim = compute_cosine_similarity(statement_tfidf.flatten(), passage_tfidf.flatten())
        similarities.append((cosine_sim, euclidean_dist, tfidf_cosine_sim))

    return {
        'example_id': item.get('example_id', ''),
        'statement': statement_text,
        'tokens': processed_statement['tokens'],
        'original_label': item.get('label', 'Unknown'),
        'statement_features': statement_features.tolist(),
        'statement_tfidf': statement_tfidf.tolist(),
        'legal_passages_embeddings': [emb.tolist() for emb in legal_passages_embeddings],
        'legal_passages_tfidf': [tfidf.tolist() for tfidf in legal_passages_tfidf],
        'similarities': similarities
    }

def main():
    stopwords_directory = 'vi-stopword'
    synonyms_directory = 'vi-wordnet'
    
    stopwords = load_stopwords(stopwords_directory)
    synonyms_dict = load_synonyms_from_files(synonyms_directory)
    
    train_file = 'train.json'
    test_file = 'test.json'
    legal_passages_file = 'legal_passages.json'
    
    train_data = load_json(train_file)
    test_data = load_json(test_file)
    legal_passages = load_json(legal_passages_file)
    
    if train_data and legal_passages:
        processed_train_data = preprocess_data(train_data, legal_passages, synonyms_dict, stopwords)
        save_json(processed_train_data, 'processed_train.json')
    
    if test_data and legal_passages:
        processed_test_data = preprocess_data(test_data, legal_passages, synonyms_dict, stopwords)
        save_json(processed_test_data, 'processed_test.json')

if __name__ == "__main__":
    main()
