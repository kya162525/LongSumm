import json
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import logging
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"./jaehyeok/logs/training_gnn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Reuse data loading functions from the baseline
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

# New classes and functions for the GNN-based approach

class SentenceGraphDataset:
    """Dataset for creating and managing sentence graphs for documents"""
    def __init__(
        self,
        dataset,
        papers_path,
        json_dir="./papers/json_files/",
        bert_model_name="allenai/scibert_scivocab_uncased",
        sentence_model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.dataset = dataset
        self.papers_path = papers_path
        self.json_dir = json_dir
        self.device = device
        self.similarity_threshold = similarity_threshold
        
        # Load papers
        self.papers = load_papers(papers_path)
        
        # Initialize sentence transformer for embeddings
        logging.info(f"Loading SentenceTransformer model: {sentence_model_name}")
        self.sentence_model = SentenceTransformer(sentence_model_name, device=device)
        
        # Store transition phrases for dependency detection
        self.transition_phrases = [
            "therefore", "thus", "hence", "consequently", "as a result",
            "however", "nevertheless", "in contrast", "on the other hand",
            "moreover", "furthermore", "additionally", "in addition",
            "for example", "for instance", "specifically", "in particular"
        ]
        
        # Store graph data for each paper
        self.graph_data = {}
        self.prepare_graphs()
    
    def get_position_features(self, paper_id, paper_sentences):
        """Extract position information for each sentence."""
        sections = split_into_sections(self.papers[paper_id], paper_id, self.json_dir)
        
        # Initialize position features
        position_features = np.zeros((len(paper_sentences), 3))  # [section_idx, para_idx, sent_idx]
        
        # If no sections were found, use default positions
        if not sections:
            for i in range(len(paper_sentences)):
                position_features[i] = [0, 0, i / len(paper_sentences)]
            return position_features
        
        # Map sentences to their positions
        sentence_to_pos = {}
        total_sections = len(sections)
        
        for section_idx, (_, paragraphs) in enumerate(sections):
            total_paragraphs = len(paragraphs)
            
            for para_idx, paragraph in enumerate(paragraphs):
                para_sentences = sent_tokenize(paragraph)
                total_sentences = len(para_sentences)
                
                for sent_idx, sent in enumerate(para_sentences):
                    if sent.strip() in paper_sentences:
                        idx = paper_sentences.index(sent.strip())
                        # Normalize positions
                        norm_section_idx = section_idx / max(1, total_sections - 1)
                        norm_para_idx = para_idx / max(1, total_paragraphs - 1)
                        norm_sent_idx = sent_idx / max(1, total_sentences - 1)
                        
                        sentence_to_pos[idx] = [norm_section_idx, norm_para_idx, norm_sent_idx]
        
        # Fill in the position features
        for i in range(len(paper_sentences)):
            if i in sentence_to_pos:
                position_features[i] = sentence_to_pos[i]
            else:
                # Default position for sentences we couldn't map
                position_features[i] = [0, 0, i / len(paper_sentences)]
        
        return position_features
    
    def has_transition_phrase(self, sentence):
        """Check if a sentence contains a transition phrase."""
        sentence_lower = sentence.lower()
        for phrase in self.transition_phrases:
            if phrase in sentence_lower:
                return True
        return False
    
    def build_graph(self, paper_id):
        """Build a graph for a single paper."""
        if paper_id not in self.papers:
            logging.warning(f"Paper {paper_id} not found in papers directory")
            return None
        
        paper_text = self.papers[paper_id]
        paper_sentences = sent_tokenize(paper_text)
        
        # Get positive indices (sentences used in extractive summary)
        positive_indices = set(self.dataset[paper_id].get("positive_indices", []))
        
        # Prepare labels
        labels = torch.zeros(len(paper_sentences), dtype=torch.float)
        for idx in positive_indices:
            if idx < len(paper_sentences):
                labels[idx] = 1.0
        
        # Get sentence embeddings
        logging.info(f"Generating embeddings for {len(paper_sentences)} sentences in paper {paper_id}")
        embeddings = self.sentence_model.encode(paper_sentences)
        
        # Get position features
        position_features = self.get_position_features(paper_id, paper_sentences)
        
        # Get length features
        length_features = np.array([[len(s.split())] for s in paper_sentences])
        
        # Normalize length features
        length_features = length_features / max(length_features.max(), 1)
        
        # Combine features
        combined_features = np.hstack((embeddings, position_features, length_features))
        
        # Create edge indices and weights
        edge_index = []
        edge_weight = []
        edge_type = []  # 0: adjacent, 1: similarity, 2: dependency
        
        # 1. Connect adjacent sentences (sequential context)
        for i in range(len(paper_sentences) - 1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])  # Bidirectional
            edge_weight.extend([1.0, 1.0])
            edge_type.extend([0, 0])
        
        # 2. Connect sentences based on similarity
        for i in range(len(paper_sentences)):
            for j in range(i + 2, len(paper_sentences)):  # Skip adjacent which we added above
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > self.similarity_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Bidirectional
                    edge_weight.extend([sim, sim])
                    edge_type.extend([1, 1])
        
        # 3. Connect sentences based on dependency relations (transition phrases)
        for i in range(len(paper_sentences)):
            if self.has_transition_phrase(paper_sentences[i]) and i > 0:
                edge_index.append([i-1, i])
                edge_index.append([i, i-1])  # Bidirectional
                edge_weight.extend([1.0, 1.0])
                edge_type.extend([2, 2])
        
        # 4. Add document node (optional)
        doc_node_idx = len(paper_sentences)
        doc_embedding = np.mean(embeddings, axis=0)
        doc_features = np.hstack((doc_embedding, np.zeros(3), np.array([1.0])))
        
        # Add document node to features
        combined_features = np.vstack((combined_features, doc_features))
        
        # Connect document node to all sentences
        for i in range(len(paper_sentences)):
            edge_index.append([doc_node_idx, i])
            edge_index.append([i, doc_node_idx])
            edge_weight.extend([1.0, 1.0])
            edge_type.extend([3, 3])
        
        # Add dummy label for document node (not used in loss)
        labels = torch.cat([labels, torch.zeros(1, dtype=torch.float)])
        
        # Convert to tensors
        x = torch.tensor(combined_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        # Create PyG Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_type=edge_type,
            y=labels,
            num_nodes=len(paper_sentences) + 1,  # +1 for document node
            paper_id=paper_id,
            sentences=paper_sentences,
            doc_node_idx=doc_node_idx
        )
        
        return graph_data
    
    def prepare_graphs(self):
        """Prepare graph data for all papers in the dataset."""
        logging.info("Building graphs for all papers...")
        
        for paper_id in tqdm(self.dataset.keys(), desc="Building graphs"):
            if paper_id in self.papers:
                graph_data = self.build_graph(paper_id)
                if graph_data:
                    self.graph_data[paper_id] = graph_data
        
        logging.info(f"Built graphs for {len(self.graph_data)} papers")
    
    def split_dataset(self, test_size=0.2, random_state=42):
        """Split the dataset into training and test sets."""
        paper_ids = list(self.graph_data.keys())
        train_ids, test_ids = train_test_split(
            paper_ids, test_size=test_size, random_state=random_state
        )
        
        train_graphs = [self.graph_data[pid] for pid in train_ids]
        test_graphs = [self.graph_data[pid] for pid in test_ids]
        
        logging.info(f"Split dataset: {len(train_graphs)} training graphs, {len(test_graphs)} test graphs")
        
        return train_graphs, test_graphs

class GNNSummarizer(nn.Module):
    """GNN-based model for extractive summarization"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, heads=4):
        super(GNNSummarizer, self).__init__()
        
        # GNN layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=2)
        
        # MLP for node classification
        self.mlp = nn.Sequential(
            nn.Linear(output_dim * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Normalization layers
        self.layernorm1 = nn.LayerNorm(hidden_dim * heads)
        self.layernorm2 = nn.LayerNorm(output_dim * 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, edge_index, edge_weight=None):
        # First GAT layer
        h = self.gat1(x, edge_index, edge_weight)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.layernorm1(h)
        h = self.dropout(h)
        
        # Second GAT layer
        h = self.gat2(h, edge_index, edge_weight)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.layernorm2(h)
        h = self.dropout(h)
        
        # MLP for node classification
        out = self.mlp(h)
        
        return torch.sigmoid(out).squeeze(1)

def train_gnn_model(model, train_graphs, test_graphs, epochs=30, lr=2e-5, weight_decay=1e-2, device="cuda"):
    """Train the GNN model."""
    logging.info(f"Training GNN model on {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index, data.edge_weight)
            
            # Get only sentence node outputs (exclude document node)
            sentence_out = out[:data.doc_node_idx]
            sentence_labels = data.y[:data.doc_node_idx]
            
            # Loss calculation with class weighting
            pos_weight = torch.tensor([(data.doc_node_idx - sentence_labels.sum()) / max(1, sentence_labels.sum())]).to(device)
            criterion = nn.BCELoss(reduction='mean', weight=pos_weight if sentence_labels.sum() > 0 else None)
            
            loss = criterion(sentence_out, sentence_labels)
            
            # Optional: Add ranking loss
            pos_indices = (sentence_labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (sentence_labels == 0).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Sample up to 5 positive-negative pairs
                num_pairs = min(5, len(pos_indices), len(neg_indices))
                pos_samples = pos_indices[torch.randperm(len(pos_indices))[:num_pairs]]
                neg_samples = neg_indices[torch.randperm(len(neg_indices))[:num_pairs]]
                
                # Ranking loss: ensure positive sentences score higher than negative ones
                ranking_loss = F.relu(1.0 - (sentence_out[pos_samples] - sentence_out[neg_samples])).mean()
                loss += 0.5 * ranking_loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} (Test)"):
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_weight)
                
                # Get only sentence nodes (exclude document node)
                sentence_out = out[:data.doc_node_idx]
                sentence_labels = data.y[:data.doc_node_idx]
                
                # Binary prediction (threshold = 0.5)
                pred = (sentence_out > 0.5).float()
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(sentence_labels.cpu().numpy())
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        logging.info(f"Epoch {epoch+1}/{epochs} - Test: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"./jaehyeok/models/gnn_summarizer_best.pt")
    
    logging.info(f"Best model from epoch {best_epoch+1} with F1: {best_f1:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"./jaehyeok/models/gnn_summarizer_best.pt"))
    return model, best_f1

def evaluate_model(model, test_graphs, device="cuda"):
    """Evaluate the trained GNN model."""
    logging.info("Evaluating model on test set...")
    
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    all_preds = []
    all_labels = []
    results = {}
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_weight)
            
            # Get only sentence nodes (exclude document node)
            sentence_out = out[:data.doc_node_idx]
            sentence_labels = data.y[:data.doc_node_idx]
            
            # Binary prediction (threshold = 0.5)
            pred = (sentence_out > 0.5).float()
            
            # Store results
            paper_id = data.paper_id
            sentences = data.sentences
            
            # Convert to numpy for easier handling
            scores = sentence_out.cpu().numpy()
            predictions = pred.cpu().numpy()
            labels = sentence_labels.cpu().numpy()
            
            results[paper_id] = {
                'sentences': sentences,
                'scores': scores.tolist(),
                'predictions': predictions.tolist(),
                'labels': labels.tolist()
            }
            
            all_preds.extend(predictions)
            all_labels.extend(labels)
    
    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    logging.info(f"Final Evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info("\n" + classification_report(all_labels, all_preds, zero_division=0))
    
    return results, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train GNN-based extractive summarization model")
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
        "--sentence_model", 
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings"
    )
    parser.add_argument(
        "--similarity_threshold", 
        type=float,
        default=0.6,
        help="Threshold for connecting nodes based on similarity"
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int,
        default=128,
        help="Hidden dimension for GNN layers"
    )
    parser.add_argument(
        "--output_dim", 
        type=int,
        default=64,
        help="Output dimension for GNN layers"
    )
    parser.add_argument(
        "--attention_heads", 
        type=int,
        default=4,
        help="Number of attention heads for GAT layers"
    )
    parser.add_argument(
        "--epochs", 
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float,
        default=1e-2,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Create graph dataset
    logging.info("Creating sentence graph dataset...")
    graph_dataset = SentenceGraphDataset(
        dataset,
        args.papers_path,
        args.json_dir,
        sentence_model_name=args.sentence_model,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )
    
    # Split dataset
    train_graphs, test_graphs = graph_dataset.split_dataset()
    
    # Calculate input dimension
    sample_graph = next(iter(train_graphs))
    input_dim = sample_graph.x.shape[1]
    
    # Create model
    logging.info(f"Creating GNN model with input dim {input_dim}")
    model = GNNSummarizer(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        heads=args.attention_heads
    )
    
    # Train model
    model, best_f1 = train_gnn_model(
        model,
        train_graphs,
        test_graphs,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        args.device
    )
    
    # Evaluate model
    results, precision, recall, f1 = evaluate_model(model, test_graphs, args.device)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.output_dir, f"gnn_results_{timestamp}.json")
    
    with open(results_path, "w") as f:
        # Save metrics and hyperparameters
        json.dump({
            "model": "GNN Summarizer",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_f1_during_training": best_f1,
            "hyperparameters": {
                "hidden_dim": args.hidden_dim,
                "output_dim": args.output_dim,
                "attention_heads": args.attention_heads,
                "similarity_threshold": args.similarity_threshold,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "sentence_model": args.sentence_model
            }
        }, f, indent=4)
    
    logging.info(f"Results saved to {results_path}")
    
    # Save a sample of predictions for analysis
    sample_results = {}
    for i, (paper_id, result) in enumerate(results.items()):
        if i >= 5:  # Save results for 5 papers
            break
        
        # Extract positive predictions
        positive_indices = [i for i, p in enumerate(result["predictions"]) if p == 1]
        
        # Extract top-scoring sentences if no positive predictions
        if not positive_indices:
            scores = np.array(result["scores"])
            positive_indices = scores.argsort()[-5:][::-1]
        
        # Record info for these sentences
        sample_results[paper_id] = {
            "positive_sentences": [result["sentences"][i] for i in positive_indices],
            "scores": [result["scores"][i] for i in positive_indices],
            "ground_truth": [i for i, l in enumerate(result["labels"]) if l == 1],
        }
    
    sample_path = os.path.join(args.output_dir, f"gnn_sample_predictions_{timestamp}.json")
    with open(sample_path, "w") as f:
        json.dump(sample_results, f, indent=4)
    
    logging.info(f"Sample predictions saved to {sample_path}")

if __name__ == "__main__":
    main()
