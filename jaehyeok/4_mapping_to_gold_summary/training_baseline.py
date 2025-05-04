import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import nltk
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import joblib
import logging
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"./jaehyeok/logs/training_baseline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def load_dataset(dataset_path: str) -> Dict:
    """
    Load the dataset created by make_datasets.py and process mappings 
    to identify positive and negative examples.
    """
    logging.info(f"Loading dataset from {dataset_path}")
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Process each paper to enhance the data structure
        for paper_id, paper_data in dataset.items():
            # Extract the mappings
            mappings = paper_data.get("mappings", [])
            
            # Create a set of positive paper_idx values (sentences used in extractive summary)
            positive_indices = set(mapping["paper_idx"] for mapping in mappings)
            
            # Create a mapping from paper_idx to corresponding summary sentences
            idx_to_summary = {}
            for mapping in mappings:
                paper_idx = mapping["paper_idx"]
                if paper_idx not in idx_to_summary:
                    idx_to_summary[paper_idx] = []
                
                idx_to_summary[paper_idx].append({
                    "summary_idx": mapping["summary_idx"],
                    "summary_sentence": mapping["summary_sentence"],
                    "similarity": mapping["similarity"]
                })
            
            # Add the enhanced information to the paper_data
            paper_data["positive_indices"] = list(positive_indices)
            paper_data["idx_to_summary"] = idx_to_summary
        
        logging.info(f"Loaded dataset with {len(dataset)} papers")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return {}

def load_papers(papers_path: str) -> Dict[str, str]:
    """Load all paper texts from the specified directory."""
    logging.info(f"Loading papers from {papers_path}")
    papers = {}
    files = [f for f in os.listdir(papers_path) if f.endswith('.txt')]
    
    for file in tqdm(files, desc="Loading papers"):
        paper_id = file.split(".")[0]
        try:
            with open(os.path.join(papers_path, file), "r") as f:
                papers[paper_id] = f.read()
        except Exception as e:
            logging.error(f"Error loading paper {paper_id}: {str(e)}")
    
    logging.info(f"Successfully loaded {len(papers)} papers")
    return papers

def load_section_info(json_dir: str, paper_id: str) -> List[Tuple[str, List[str]]]:
    """Load section information from JSON files."""
    try:
        json_path = os.path.join(json_dir, f"{paper_id}.json")
        if not os.path.exists(json_path):
            return None
            
        with open(json_path, 'r') as f:
            paper_data = json.load(f)
            
        sections = []
        if 'sections' in paper_data:
            for section in paper_data['sections']:
                heading = section.get('heading', '')
                text = section.get('text', '')
                # Split text into paragraphs
                paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
                if not paragraphs and text.strip():
                    paragraphs = [text.strip()]
                    
                sections.append((heading, paragraphs))
            
            return sections
    except Exception as e:
        logging.warning(f"Error loading section info for paper {paper_id}: {str(e)}")
        
    return None

def split_into_sections(paper_text: str, paper_id: str = None, json_dir: str = "./papers/json_files/") -> List[Tuple[str, List[str]]]:
    """Split paper text into sections and paragraphs, using JSON data when available."""
    # First try to get sections from JSON if paper_id is provided
    if paper_id:
        sections = load_section_info(json_dir, paper_id)
        if sections:
            logging.debug(f"Using JSON section info for paper {paper_id}: {len(sections)} sections found")
            return sections
            
    # Fallback to heuristic-based approach
    logging.debug(f"Using heuristic section detection for paper {paper_id if paper_id else 'unknown'}")
    sections = []
    current_section = ""
    current_paragraphs = []
    
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', paper_text) if p.strip()]
    
    for para in paragraphs:
        # Check if this paragraph looks like a section header
        if re.match(r'^[A-Z][A-Z\s]{3,}', para) or re.match(r'^\d+\.?\s+[A-Z][A-Z\s]*', para):
            # Save previous section if it exists
            if current_paragraphs:
                sections.append((current_section, current_paragraphs))
                current_paragraphs = []
            current_section = para
        else:
            current_paragraphs.append(para)
    
    # Add the last section
    if current_paragraphs:
        sections.append((current_section, current_paragraphs))
        
    return sections

def extract_sentence_features(
    paper_id: str,
    paper_text: str,
    paper_sentences: List[str],
    tfidf_vectorizer: TfidfVectorizer = None,
    sentence_model: SentenceTransformer = None,
    json_dir: str = "./papers/json_files/"
) -> List[Dict]:
    """Extract features for each sentence in the paper."""
    stop_words = set(stopwords.words('english'))
    # Pass paper_id to split_into_sections to enable JSON-based section extraction
    sections = split_into_sections(paper_text, paper_id, json_dir)
    
    # If no sections were found, treat the whole document as one section
    if not sections:
        sections = [("Document", [paper_text])]
    
    # Map each sentence to its position in the document
    sentence_positions = {}
    total_sentences = len(paper_sentences)
    sentence_idx = 0
    
    features = []
    section_idx = 0
    
    for section_title, paragraphs in sections:
        para_idx = 0
        for para in paragraphs:
            para_sentences = sent_tokenize(para.strip())
            for s_idx, sent in enumerate(para_sentences):
                sent = sent.strip()
                if sent in paper_sentences:
                    idx = paper_sentences.index(sent)
                    
                    # 1. Position features
                    relative_position = idx / total_sentences
                    sentence_positions[idx] = {
                        'section_idx': section_idx,
                        'paragraph_idx': para_idx,
                        'sentence_idx': s_idx,
                        'relative_position': relative_position
                    }
                    
                sentence_idx += 1
            para_idx += 1
        section_idx += 1
    
    # Calculate TF-IDF if vectorizer is provided
    if tfidf_vectorizer:
        tfidf_matrix = tfidf_vectorizer.transform(paper_sentences)
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
    
    # Extract features for each sentence
    for i, sentence in enumerate(paper_sentences):
        # Get position info (default if not found)
        pos_info = sentence_positions.get(i, {
            'section_idx': 0,
            'paragraph_idx': 0,
            'sentence_idx': 0,
            'relative_position': i / total_sentences
        })
        
        # 2. Sentence length
        tokens = sentence.split()
        sentence_length = len(tokens)
        
        # 3. Keyword features
        pos_tags = pos_tag(tokens)
        nouns = [word.lower() for word, tag in pos_tags if tag.startswith('N')]
        verbs = [word.lower() for word, tag in pos_tags if tag.startswith('V')]
        
        noun_ratio = len(nouns) / max(1, sentence_length)
        verb_ratio = len(verbs) / max(1, sentence_length)
        
        # Calculate non-stopword ratio
        non_stop_words = [word.lower() for word in tokens if word.lower() not in stop_words]
        non_stop_ratio = len(non_stop_words) / max(1, sentence_length)
        
        # 4. Similarity with adjacent sentences
        sim_prev, sim_next = 0.0, 0.0
        if sentence_model:
            if i > 0:
                emb_curr = sentence_model.encode(sentence)
                emb_prev = sentence_model.encode(paper_sentences[i-1])
                sim_prev = np.dot(emb_curr, emb_prev) / (np.linalg.norm(emb_curr) * np.linalg.norm(emb_prev))
            
            if i < len(paper_sentences) - 1:
                emb_curr = sentence_model.encode(sentence) if i == 0 else emb_curr
                emb_next = sentence_model.encode(paper_sentences[i+1])
                sim_next = np.dot(emb_curr, emb_next) / (np.linalg.norm(emb_curr) * np.linalg.norm(emb_next))
        
        # Compile features
        feature_dict = {
            'paper_id': paper_id,
            'sentence_idx': i,
            'sentence': sentence,
            # Position features
            'section_idx': pos_info['section_idx'],
            'paragraph_idx': pos_info['paragraph_idx'],
            'local_sentence_idx': pos_info['sentence_idx'],
            'relative_position': pos_info['relative_position'],
            # Length features
            'sentence_length': sentence_length,
            'word_count': sentence_length,
            # Content features
            'noun_ratio': noun_ratio,
            'verb_ratio': verb_ratio,
            'non_stop_ratio': non_stop_ratio,
            # Similarity features
            'similarity_prev': sim_prev,
            'similarity_next': sim_next,
            'similarity_avg': (sim_prev + sim_next) / 2 if (i > 0 and i < len(paper_sentences) - 1) else 
                              sim_next if i == 0 else sim_prev
        }
        
        # Add TF-IDF if available
        if tfidf_vectorizer:
            feature_dict['tfidf_score'] = tfidf_scores[i]
        
        features.append(feature_dict)
    
    return features

def prepare_training_data(dataset, papers_path, use_sentence_transformer=True, json_dir="./papers/json_files/"):
    """Prepare the training data from the mappings dataset."""
    logging.info("Preparing training data...")
    
    # Load papers
    papers = load_papers(papers_path)
    
    # Initialize sentence transformer if needed
    sentence_model = None
    if use_sentence_transformer:
        logging.info("Loading sentence transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # First pass: collect all sentences to train TF-IDF
    all_sentences = []
    paper_sentences_map = {}
    
    for paper_id, paper_data in tqdm(dataset.items(), desc="Collecting sentences"):
        if paper_id not in papers:
            logging.warning(f"Paper {paper_id} not found in papers directory")
            continue
            
        paper_text = papers[paper_id]
        paper_sentences = sent_tokenize(paper_text)
        paper_sentences_map[paper_id] = paper_sentences
        all_sentences.extend(paper_sentences)
    
    # Train TF-IDF vectorizer
    logging.info("Training TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_vectorizer.fit(all_sentences)
    
    # Second pass: extract features and create labels with balanced samples
    all_features = []
    
    for paper_id, paper_data in tqdm(dataset.items(), desc="Extracting features"):
        if paper_id not in papers:
            continue
            
        paper_text = papers[paper_id]
        paper_sentences = paper_sentences_map[paper_id]
        
        # Get positive indices (sentences used in extractive summary)
        positive_indices = set(paper_data.get("positive_indices", []))
        
        # Extract features for all sentences
        sentence_features = extract_sentence_features(
            paper_id,
            paper_text,
            paper_sentences,
            tfidf_vectorizer,
            sentence_model,
            json_dir
        )
        
        # Separate positive and negative examples
        positive_features = []
        negative_features = []
        
        for feature in sentence_features:
            sentence_idx = feature['sentence_idx']
            if sentence_idx in positive_indices:
                feature['label'] = 1
                positive_features.append(feature)
            else:
                feature['label'] = 0
                negative_features.append(feature)
        
        # Balance the dataset by sampling equal number of negative examples
        num_positives = len(positive_features)
        if num_positives > 0 and len(negative_features) > 0:
            # If we have more negatives than positives, sample randomly
            if len(negative_features) > num_positives:
                import random
                negative_features = random.sample(negative_features, num_positives)
            
            logging.debug(f"Paper {paper_id}: {num_positives} positive, {len(negative_features)} negative examples")
            
            # Add the balanced samples to our dataset
            all_features.extend(positive_features)
            all_features.extend(negative_features)
        else:
            logging.warning(f"Paper {paper_id}: Could not balance classes. Positives: {num_positives}, Negatives: {len(negative_features)}")
            # Add whatever we have
            all_features.extend(positive_features)
            all_features.extend(negative_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    positive_count = df['label'].sum()
    total_count = len(df)
    logging.info(f"Created balanced dataset with {total_count} sentences, {positive_count} positive examples ({positive_count/total_count:.2%})")
    
    return df, tfidf_vectorizer

def train_and_evaluate_models(df, test_size=0.2, random_state=42):
    """Train and evaluate various classifiers."""
    logging.info("Training and evaluating models...")
    
    # Prepare feature columns (exclude irrelevant columns)
    feature_cols = [col for col in df.columns if col not in [
        'paper_id', 'sentence_idx', 'sentence', 'label'
    ]]
    
    X = df[feature_cols].fillna(0)
    y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"Training set: {X_train.shape[0]} examples, Test set: {X_test.shape[0]} examples")
    
    # Define models to train
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        logging.info(f"{name} cross-validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        results[name] = {
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_f1': cv_scores.mean()
        }
        
        logging.info(f"{name} test set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info("\n" + classification_report(y_test, y_pred))
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]['model']
    logging.info(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")
    
    return best_model, results, feature_cols

def main():
    parser = argparse.ArgumentParser(description="Train extractive summarization models")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="./jaehyeok/datasets/mapping_gold_summary_for_training_sets.json",
        help="Path to the dataset created by make_datasets.py"
    )
    parser.add_argument(
        "--papers_path", 
        type=str, 
        default="./papers/postprocessed/full_texts/",
        help="Path to the paper text files"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="./papers/json_files/",
        help="Path to the directory containing paper JSON files with section information"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./jaehyeok/models/",
        help="Directory to save model and results"
    )
    parser.add_argument(
        "--use_sentence_transformer",
        action="store_true",
        help="Use sentence transformer for similarity features"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Prepare training data - pass json_dir to use JSON section data
    df, tfidf_vectorizer = prepare_training_data(
        dataset, 
        args.papers_path,
        args.use_sentence_transformer,
        args.json_dir
    )
    
    # Train and evaluate models
    best_model, results, feature_cols = train_and_evaluate_models(df)
    
    # Save outputs
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the best model
    joblib.dump(best_model, os.path.join(args.output_dir, f"best_model_{timestamp}.joblib"))
    
    # Save the TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, os.path.join(args.output_dir, f"tfidf_vectorizer_{timestamp}.joblib"))
    
    # Save feature columns
    with open(os.path.join(args.output_dir, f"feature_cols_{timestamp}.json"), "w") as f:
        json.dump(feature_cols, f)
    
    # Save model results
    results_to_save = {name: {k: v for k, v in model_results.items() if k != 'model'} 
                       for name, model_results in results.items()}
    
    with open(os.path.join(args.output_dir, f"model_results_{timestamp}.json"), "w") as f:
        json.dump(results_to_save, f, indent=4)
    
    # Save a sample of the processed data
    df.sample(min(1000, len(df))).to_csv(
        os.path.join(args.output_dir, f"sample_training_data_{timestamp}.csv"), 
        index=False
    )
    
    logging.info(f"Training completed. Models and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()