"""Batch comparison service helpers (auto-extracted)."""
import os, json, tempfile
from sklearn.metrics.pairwise import cosine_similarity
from app.services.file_processing import preprocess_text_for_similarity, remove_template_content, _make_vectorizer

def _precompute_tfidf_for_files(file_data, template_text=None):
    texts = []
    for fd in file_data:
        clean = preprocess_text_for_similarity(fd['text'], template_text)
        if template_text:
            clean = remove_template_content(clean, template_text)
        texts.append(clean)
    vectorizer = _make_vectorizer(stop_words=None, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def _compute_pair_similarity_from_matrix(tfidf_matrix, i, j):
    sim = cosine_similarity(tfidf_matrix[i:i + 1], tfidf_matrix[j:j + 1])[0][0]
    return sim

def store_batch_comparison_temp(data):
    fd, path = tempfile.mkstemp(suffix='.json', prefix='comp_', text=True)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    return path

def load_batch_comparison_temp(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# g.download_tokens, file_cache_manager, add_to_cache, load_cache_from_db
# are imported from app.globals and app.services.file_cache at top of file.
