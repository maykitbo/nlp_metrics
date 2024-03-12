from transformers import AutoModel, AutoTokenizer, DistilBertModel, DistilBertTokenizer, DistilBertConfig
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load English stop words
stop_words = set(stopwords.words('english'))


# ess_model = SentenceTransformer('LaBSE')


model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()



def get_bert_embeddings(text, tokenizer, model):
    # Tokenize input text and prepare input tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Forward pass, get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings
    embeddings = outputs.last_hidden_state
    return embeddings[0]


def tokenize_and_get_words_1(text, tokenizer, model):
    # Tokenize input text and prepare input tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    
    # Convert input IDs to list of words/tokens
    tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in input_ids.tolist()[0]]
    
    # Forward pass to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    
    return tokens, embeddings


def tokenize_and_get_words_2(text, tokenizer, model):
    # Tokenize input text and prepare input tensors
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    
    # Convert input IDs to list of words/tokens
    tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in input_ids.tolist()[0]]
    
    # Filter out stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    filtered_tokens = []
    filtered_indices = []
    
    for i, token in enumerate(tokens):
        # Remove '##' used by BERT for subwords
        clean_token = token.replace("##", "")
        if clean_token not in stop_words and clean_token not in punctuation:
            filtered_tokens.append(token)
            filtered_indices.append(i)
    
    # Forward pass to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
    
    # filtered_tokens = list(set(filtered_tokens))

    # Filter embeddings to only include those that correspond to filtered_tokens
    filtered_embeddings = embeddings[filtered_indices]
    
    return filtered_tokens, filtered_embeddings


def find_clusters(embeddings, sim_threshold=0.8):
    # Compute cosine similarity matrix
    cos_sim_matrix = cosine_similarity(embeddings, embeddings)
    
    # Cluster embeddings based on cosine similarity threshold
    clusters = []
    for i, embedding in enumerate(embeddings):
        in_cluster = [j for j, sim in enumerate(cos_sim_matrix[i]) if sim > sim_threshold]
        clusters.append(in_cluster)
    
    return clusters, cos_sim_matrix


def score_v1(text, sim_threshold=0.8):
    tokens, embeddings = tokenize_and_get_words_2(text, tokenizer, model)
    clusters, cos_sim_matrix = find_clusters(embeddings, sim_threshold)
    word_clusters = clusters_to_words(clusters, tokens)
    return word_clusters, clusters, cos_sim_matrix, tokens, embeddings


def clusters_to_words(clusters, tokens):
    words_clusters = []
    for cluster in clusters:
        # Map each token ID in the cluster to its corresponding word
        words = [tokens[i] for i in cluster]
        words_clusters.append(words)
    return words_clusters

