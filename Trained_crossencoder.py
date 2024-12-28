import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)

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
        
        # Load your provided definitions here
        self.label_definitions = {
            'sensitive_pii': """[Your sensitive PII definition]""",
            'confidential': """[Your confidential definition]""",
            'licensed': """[Your licensed data definition]""",
            'non_sensitive_pii': """[Your non-sensitive PII definition]"""
        }

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
                # Use last hidden state
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

    def compute_similarity_score(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute similarity between embeddings"""
        # Ensure same dimensionality
        if emb1.size(-1) != emb2.size(-1):
            # Pad smaller embedding
            max_dim = max(emb1.size(-1), emb2.size(-1))
            if emb1.size(-1) < max_dim:
                emb1 = torch.nn.functional.pad(emb1, (0, max_dim - emb1.size(-1)))
            else:
                emb2 = torch.nn.functional.pad(emb2, (0, max_dim - emb2.size(-1)))
        
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        return similarity.item()

    def predict(self, attr_name: str, description: str) -> Dict[str, float]:
        """Predict sensitivity scores"""
        self.model.eval()
        combined_text = f"{attr_name}: {description}"
        text_embedding = self.get_hierarchical_embedding(combined_text)
        
        scores = {}
        for label, definition in self.label_definitions.items():
            def_embedding = self.get_hierarchical_embedding(definition)
            score = self.compute_similarity_score(text_embedding, def_embedding)
            scores[label] = score
            
        return scores

    def train(self, train_df: pd.DataFrame, epochs=3, learning_rate=2e-5):
        """Train the model"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_df.iterrows(), total=len(train_df))
            
            for idx, row in progress_bar:
                # Get embeddings
                text_emb = self.get_hierarchical_embedding(
                    f"{row['attribute_name']}: {row['description']}"
                )
                
                # Get embeddings for all definitions
                def_embeddings = {}
                for label, definition in self.label_definitions.items():
                    def_embeddings[label] = self.get_hierarchical_embedding(definition)
                
                # Calculate loss
                target = torch.zeros(len(self.label_definitions)).to(self.device)
                target[list(self.label_definitions.keys()).index(row['label'])] = 1
                
                # Compute similarities
                similarities = []
                for def_emb in def_embeddings.values():
                    sim = self.compute_similarity_score(text_emb, def_emb)
                    similarities.append(sim)
                
                predictions = torch.tensor(similarities).to(self.device)
                loss = torch.nn.functional.cross_entropy(predictions.unsqueeze(0), target.unsqueeze(0))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_df)
            logging.info(f"Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}")

    def process_csv(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """Process CSV file"""
        df = pd.read_csv(input_path)
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            scores = self.predict(row['attribute_name'], row['description'])
            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
            
            results.append({
                'attribute_name': row['attribute_name'],
                'description': row['description'],
                'predicted_label': predicted_label,
                'sensitive_pii_score': scores['sensitive_pii'],
                'confidential_score': scores['confidential'],
                'licensed_score': scores['licensed'],
                'non_sensitive_pii_score': scores['non_sensitive_pii']
            })
        
        output_df = pd.DataFrame(results)
        if output_path:
            output_df.to_csv(output_path, index=False)
        
        return output_df

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

# Example usage:
"""
# Initialize
classifier = HierarchicalCrossEncoder()

# Training data
train_df = pd.DataFrame({
    'attribute_name': ['customer_ssn', 'trading_strategy'],
    'description': ['Social security number', 'Trading algorithm'],
    'label': ['sensitive_pii', 'confidential']
})

# Train
classifier.train(train_df)

# Predict
test_df = pd.read_csv('test.csv')
results = classifier.process_csv('test.csv', 'predictions.csv')
"""
