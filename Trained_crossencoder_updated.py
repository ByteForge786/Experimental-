import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    attribute_name: str
    description: str
    predicted_label: str
    scores: Dict[str, float]

class HierarchicalCrossEncoder:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=16):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = 512
        self.chunk_overlap = 50
        
        # Individual definitions for each label
        self.label_definitions = {
            'SENSITIVE_PII': [
                "This data reveals racial or ethnic origin of a person",
                "This data contains political opinions or religious beliefs",
                "This data includes health, genetic, or biometric information",
                # Add all individual definitions
            ],
            'CONFIDENTIAL': [
                "This is non-public business information",
                "This data contains internal proprietary information",
                # Add all individual definitions
            ],
            'LICENSED': [
                "This data is sourced from third-party providers",
                "This data is obtained under commercial agreement",
                # Add all individual definitions
            ],
            'NON_SENSITIVE_PII': [
                "This data contains basic contact information",
                "This data includes standard personal identifiers",
                # Add all individual definitions
            ]
        }
        
        # Cache for definition embeddings
        self.definition_embeddings = {}
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        chunk_size = self.max_length - 2  # Account for special tokens
        
        for i in range(0, len(tokens), chunk_size - self.chunk_overlap):
            chunk = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk)
            chunks.append(chunk_text)
        
        return chunks

    def get_hierarchical_embedding(self, text: str) -> torch.Tensor:
        """Get hierarchical embedding preserving context"""
        chunks = self.chunk_text(text)
        
        # Get chunk embeddings
        chunk_embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk,
                                  return_tensors="pt",
                                  max_length=self.max_length,
                                  truncation=True,
                                  padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                chunk_emb = outputs.hidden_states[-1][:, 0, :]  # CLS token
                chunk_embeddings.append(chunk_emb)

        if len(chunk_embeddings) == 1:
            return chunk_embeddings[0]

        # Stack chunk embeddings
        stacked_embeddings = torch.cat(chunk_embeddings, dim=0)
        
        # Self-attention between chunks
        attention_scores = torch.matmul(stacked_embeddings, stacked_embeddings.transpose(0, 1))
        attention_scores = attention_scores / np.sqrt(stacked_embeddings.size(-1))
        
        # Add positional bias
        position_bias = torch.linspace(1.0, 0.8, len(chunks)).to(self.device)
        attention_scores = attention_scores * position_bias.unsqueeze(1)
        
        # Softmax for attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply attention
        contextualized_emb = torch.matmul(attention_weights, stacked_embeddings)
        
        # Concatenate all contextualized embeddings
        final_embedding = contextualized_emb.reshape(1, -1)
        
        return final_embedding

    def cache_definition_embeddings(self):
        """Cache embeddings for all individual definitions"""
        logger.info("Caching definition embeddings...")
        self.model.eval()
        
        try:
            with torch.no_grad():
                for label, definitions in self.label_definitions.items():
                    self.definition_embeddings[label] = []
                    for definition in tqdm(definitions, desc=f"Processing {label} definitions"):
                        embedding = self.get_hierarchical_embedding(definition)
                        self.definition_embeddings[label].append(embedding)
                        
            logger.info("Definition embeddings cached successfully")
            
        except Exception as e:
            logger.error(f"Error caching definition embeddings: {str(e)}")
            raise

    def compute_similarity_score(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute similarity between embeddings"""
        if emb1.size(-1) != emb2.size(-1):
            max_dim = max(emb1.size(-1), emb2.size(-1))
            if emb1.size(-1) < max_dim:
                emb1 = torch.nn.functional.pad(emb1, (0, max_dim - emb1.size(-1)))
            else:
                emb2 = torch.nn.functional.pad(emb2, (0, max_dim - emb2.size(-1)))
        
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity.item()

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict sensitivity scores for a batch of texts"""
        self.model.eval()
        
        # Ensure definitions are cached
        if not self.definition_embeddings:
            self.cache_definition_embeddings()
        
        batch_results = []
        text_embeddings = []
        
        # Get embeddings for all texts
        for text in texts:
            text_emb = self.get_hierarchical_embedding(text)
            text_embeddings.append(text_emb)
        
        # Calculate similarities with cached definition embeddings
        for text_emb in text_embeddings:
            scores = {}
            for label, def_embeddings in self.definition_embeddings.items():
                # Get max similarity across all definitions for this label
                label_scores = [
                    self.compute_similarity_score(text_emb, def_emb)
                    for def_emb in def_embeddings
                ]
                scores[label] = max(label_scores)
            batch_results.append(scores)
        
        return batch_results

    def train(self, train_df: pd.DataFrame, epochs=3, learning_rate=2e-5):
        """Train the model with batch processing"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(range(0, len(train_df), self.batch_size), 
                              desc=f"Epoch {epoch+1}/{epochs}")
            
            for start_idx in progress_bar:
                end_idx = min(start_idx + self.batch_size, len(train_df))
                batch_df = train_df.iloc[start_idx:end_idx]
                
                # Process batch
                batch_loss = 0
                optimizer.zero_grad()
                
                for _, row in batch_df.iterrows():
                    # Get embeddings
                    text_emb = self.get_hierarchical_embedding(
                        f"{row['attribute_name']}: {row['description']}"
                    )
                    
                    # Get embeddings for correct label definitions
                    label = row['original_label'].upper()
                    if not self.definition_embeddings.get(label):
                        self.cache_definition_embeddings()
                    
                    target_embeddings = self.definition_embeddings[label]
                    similarities = []
                    
                    # Calculate similarities with all definitions
                    for def_emb in target_embeddings:
                        sim = self.compute_similarity_score(text_emb, def_emb)
                        similarities.append(sim)
                    
                    # Use max similarity for loss
                    prediction = torch.tensor(similarities).to(self.device)
                    target = torch.ones(1).to(self.device)
                    loss = torch.nn.functional.cross_entropy(
                        prediction.unsqueeze(0),
                        target.long()
                    )
                    batch_loss += loss
                
                # Backward pass for batch
                batch_loss = batch_loss / len(batch_df)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
            
            avg_loss = total_loss / len(progress_bar)
            logger.info(f"Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}")

    def process_csv(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """Process CSV file with batch prediction"""
        df = pd.read_csv(input_path)
        results = []
        
        # Process in batches
        for start_idx in tqdm(range(0, len(df), self.batch_size), desc="Processing"):
            end_idx = min(start_idx + self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Prepare batch texts
            batch_texts = [
                f"{row['attribute_name']}: {row['description']}"
                for _, row in batch_df.iterrows()
            ]
            
            # Get predictions for batch
            batch_scores = self.predict_batch(batch_texts)
            
            # Process results
            for idx, scores in enumerate(batch_scores):
                row = batch_df.iloc[idx]
                predicted_label = max(scores.items(), key=lambda x: x[1])[0]
                
                results.append({
                    'attribute_name': row['attribute_name'],
                    'description': row['description'],
                    'original_label': row['original_label'].upper(),
                    'predicted_label': predicted_label,
                    **{f'{label.lower()}_score': score 
                       for label, score in scores.items()}
                })
        
        output_df = pd.DataFrame(results)
        if output_path:
            output_df.to_csv(output_path, index=False)
        
        # Log accuracy metrics
        accuracy = (output_df['predicted_label'] == output_df['original_label']).mean()
        logger.info(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Per-label accuracy
        for label in output_df['original_label'].unique():
            mask = output_df['original_label'] == label
            label_acc = (output_df[mask]['predicted_label'] == label).mean()
            logger.info(f"{label} Accuracy: {label_acc:.4f}")
        
        return output_df

    def save_model(self, path: str):
        """Save model and cached embeddings"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'definition_embeddings': self.definition_embeddings
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model and cached embeddings"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.definition_embeddings = checkpoint['definition_embeddings']
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    try:
        # Initialize classifier
        classifier = HierarchicalCrossEncoder()
        
        # Example usage
        input_file = "your_input.csv"  # Replace with your input file
        output_file = "predictions.csv"
        model_file = "models/hierarchical_model.pt"
        
        # Load data
        train_df = pd.read_csv(input_file)
        
        # Train model
        classifier.train(train_df, epochs=3)
        
        # Save model
        classifier.save_model(model_file)
        
        # Make predictions
        results = classifier.process_csv(input_file, output_file)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
