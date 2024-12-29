from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
import json

# Configure logging
log_filename = f'sensitivity_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    attribute_name: str
    description: str
    label: str
    score: float
    top_hypothesis: str
    all_scores: Dict[str, float]

class SensitivityDataset(Dataset):
    def __init__(self, 
                 texts: List[str], 
                 labels: List[str], 
                 tokenizer, 
                 hypotheses: Dict[str, List[str]], 
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hypotheses = hypotheses
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Get positive and negative hypotheses
        positive_hyp = np.random.choice(self.hypotheses[label])
        neg_labels = [l for l in self.hypotheses.keys() if l != label]
        negative_label = np.random.choice(neg_labels)
        negative_hyp = np.random.choice(self.hypotheses[negative_label])
        
        # Tokenize positive pair
        pos_encoding = self.tokenizer(
            text,
            positive_hyp,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        # Tokenize negative pair
        neg_encoding = self.tokenizer(
            text,
            negative_hyp,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        
        return {
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(1)  # 1 for positive pair
        }

class CSVSensitivityClassifier:
    def __init__(self, model_name="microsoft/deberta-v3-large-mnli", batch_size=32):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = 512
        
        logger.info(f"Model loaded on {self.device}")
        
        # Load hypotheses
        self.hypotheses = {
            'SENSITIVE_PII': [
                "This data reveals racial or ethnic origin of a person",
                "This data contains political opinions or religious beliefs",
                # ... [rest of sensitive_pii hypotheses]
            ],
            'CONFIDENTIAL': [
                "This is non-public business information",
                "This data contains internal proprietary information",
                # ... [rest of confidential hypotheses]
            ],
            'LICENSED': [
                "This data is sourced from third-party providers",
                "This data is obtained under commercial agreement",
                # ... [rest of licensed hypotheses]
            ],
            'NON_SENSITIVE_PII': [
                "This data contains basic contact information",
                "This data includes standard personal identifiers",
                # ... [rest of non_sensitive_pii hypotheses]
            ]
        }
        
        # Cache for hypothesis embeddings
        self.hypothesis_cache = {}
        
    def cache_hypothesis_embeddings(self):
        """Pre-compute and cache embeddings for all hypotheses"""
        logger.info("Caching hypothesis embeddings...")
        self.model.eval()
        
        try:
            with torch.no_grad():
                for label, hypotheses in self.hypotheses.items():
                    self.hypothesis_cache[label] = []
                    for hypothesis in tqdm(hypotheses, desc=f"Caching {label} hypotheses"):
                        encoded = self.tokenizer(
                            hypothesis,
                            max_length=self.max_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors="pt"
                        ).to(self.device)
                        outputs = self.model(**encoded)
                        self.hypothesis_cache[label].append({
                            'text': hypothesis,
                            'embedding': outputs.logits
                        })
                        
            logger.info("Hypothesis embeddings cached successfully")
            
        except Exception as e:
            logger.error(f"Error caching hypothesis embeddings: {str(e)}")
            raise
            
    def predict_batch(self, texts: List[str], 
                     attr_names: List[str], 
                     descriptions: List[str]) -> List[PredictionResult]:
        """Predict sensitivity for a batch of texts using cached hypotheses"""
        self.model.eval()
        
        # Ensure hypotheses are cached
        if not self.hypothesis_cache:
            self.cache_hypothesis_embeddings()
            
        batch_results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_attr_names = attr_names[i:i + self.batch_size]
            batch_descriptions = descriptions[i:i + self.batch_size]
            
            with torch.no_grad():
                results = []
                for text, attr_name, description in zip(batch_texts, 
                                                      batch_attr_names, 
                                                      batch_descriptions):
                    scores = {}
                    top_hyp = {}
                    
                    # Get text encoding once
                    text_encoded = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Compare with cached hypotheses
                    for label in self.hypotheses.keys():
                        label_scores = []
                        for cache_item in self.hypothesis_cache[label]:
                            inputs = self.tokenizer(
                                text,
                                cache_item['text'],
                                max_length=self.max_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors="pt"
                            ).to(self.device)
                            
                            outputs = self.model(**inputs)
                            score = torch.softmax(outputs.logits, dim=1)[:, -1].item()
                            label_scores.append((score, cache_item['text']))
                        
                        best_score, best_hyp = max(label_scores, key=lambda x: x[0])
                        scores[label] = best_score
                        top_hyp[label] = best_hyp
                    
                    # Get best label
                    final_label = max(scores.items(), key=lambda x: x[1])[0]
                    
                    results.append(PredictionResult(
                        attribute_name=attr_name,
                        description=description,
                        label=final_label,
                        score=scores[final_label],
                        top_hypothesis=top_hyp[final_label],
                        all_scores=scores
                    ))
                
                batch_results.extend(results)
        
        return batch_results
    
    def train(self, train_data: pd.DataFrame, epochs=3, learning_rate=2e-5):
        """Train the model with improved batch processing"""
        logger.info("Starting training...")
        self.model.train()
        
        # Prepare training texts
        train_texts = [
            f"{row['attribute_name']}: {row['description']}"
            for _, row in train_data.iterrows()
        ]
        train_labels = [label.upper() for label in train_data['label'].tolist()]
        
        # Create dataset and dataloader
        train_dataset = SensitivityDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.hypotheses
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, 
                              desc=f"Epoch {epoch+1}/{epochs}",
                              position=0,
                              leave=True)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Process positive pairs
                pos_outputs = self.model(
                    input_ids=batch['pos_input_ids'].to(self.device),
                    attention_mask=batch['pos_attention_mask'].to(self.device),
                    labels=batch['label'].to(self.device)
                )
                
                # Process negative pairs
                neg_outputs = self.model(
                    input_ids=batch['neg_input_ids'].to(self.device),
                    attention_mask=batch['neg_attention_mask'].to(self.device),
                    labels=torch.zeros_like(batch['label']).to(self.device)
                )
                
                # Combined loss
                loss = pos_outputs.loss + neg_outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Cache hypothesis embeddings after training
        self.cache_hypothesis_embeddings()
    
    def process_csv(self, input_data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """Process CSV with batch predictions"""
        try:
            logger.info("Starting CSV processing...")
            
            # Validate columns
            required_columns = ['attribute_name', 'description']
            if not all(col in input_data.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
            # Prepare texts
            texts = [
                f"{row['attribute_name']}: {row['description']}"
                for _, row in input_data.iterrows()
            ]
            
            # Get predictions in batches
            results = self.predict_batch(
                texts,
                input_data['attribute_name'].tolist(),
                input_data['description'].tolist()
            )
            
            # Create DataFrame
            results_df = pd.DataFrame([
                {
                    'attribute_name': r.attribute_name,
                    'description': r.description,
                    'predicted_label': r.label,
                    'confidence_score': r.score,
                    'matching_hypothesis': r.top_hypothesis,
                    **{f'{label.lower()}_score': r.all_scores[label] 
                       for label in self.hypotheses.keys()},
                    'original_label': input_data.iloc[i]['label'].upper() 
                    if 'label' in input_data.columns else None
                }
                for i, r in enumerate(results)
            ])
            
            # Calculate metrics if original labels exist
            if 'original_label' in results_df.columns:
                accuracy = (results_df['predicted_label'] == 
                          results_df['original_label']).mean()
                logger.info(f"\nOverall Accuracy: {accuracy:.4f}")
                
                # Per-label metrics
                for label in results_df['original_label'].unique():
                    mask = results_df['original_label'] == label
                    label_acc = (results_df[mask]['predicted_label'] == label).mean()
                    label_count = mask.sum()
                    logger.info(f"{label} Accuracy: {label_acc:.4f} "
                              f"(Count: {label_count})")
            
            # Save if path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def save_model(self, path: str):
        """Save model and cached embeddings"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model and tokenizer
            model_path = os.path.join(os.path.dirname(path), 'model')
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Save cache
            cache_path = os.path.join(os.path.dirname(path), 'hypothesis_cache.pt')
            torch.save(self.hypothesis_cache, cache_path)
            
            logger.info(f"Model and cache saved to {os.path.dirname(path)}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load model and cached embeddings"""
        try:
            # Load model and tokenizer
            model_path = os.path.join(os.path.dirname(path), 'model')
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)
            
            # Load cache
            cache_path = os.path.join(os.path.dirname(path), 'hypothesis_cache.pt')
            self.hypothesis_cache = torch.load(cache_path)
            
            logger.info(f"Model and cache loaded from {os.path.dirname(path)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def process_sensitivity_data(
    input_csv: str,
    output_csv: str = None,
    model_path: str = None,
    train: bool = True,
    epochs: int = 3,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    End-to-end pipeline for sensitivity prediction
    """
    try:
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Initialize classifier
        classifier = CSVSensitivityClassifier(batch_size=batch_size)
        
        if train:
            logger.info("Starting training pipeline")
            classifier.train(df, epochs=epochs)
            if model_path:
                classifier.save_model(model_path)
        elif model_path:
            logger.info(f"Loading pre-trained model from {model_path}")
            classifier.load_model(model_path)
        else:
            raise ValueError("Either train must be True or model_path must be provided")
        
        # Process data
        results_df = classifier.process_csv(df, output_csv)
        
        # Log final summary
        if 'original_label' in df.columns:
            logger.info("\nFinal Results Summary:")
            accuracy = (results_df['predicted_label'] == results_df['original_label'].str.upper()).mean()
            logger.info(f"Overall Accuracy: {accuracy:.4f}")
            
            # Confusion matrix
            labels = sorted(df['label'].unique())
            confusion_matrix = pd.DataFrame(
                0, 
                index=labels, 
                columns=labels
            )
            
            for true_label in labels:
                for pred_label in labels:
                    count = ((results_df['original_label'].str.upper() == true_label.upper()) & 
                            (results_df['predicted_label'] == pred_label.upper())).sum()
                    confusion_matrix.loc[true_label, pred_label] = count
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_matrix))
            
            # Per-label metrics
            logger.info("\nPer-Label Metrics:")
            for label in labels:
                mask = results_df['original_label'].str.upper() == label.upper()
                precision = (results_df[mask]['predicted_label'] == label.upper()).mean()
                recall = (results_df[results_df['predicted_label'] == label.upper()]['original_label'].str.upper() == label.upper()).mean()
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                logger.info(f"\n{label}:")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        input_file = "your_input.csv"  # Replace with your input file
        output_file = "predictions.csv"
        model_file = "models/deberta_sensitivity_model.pt"
        
        # Create directories
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Process data
        results_df = process_sensitivity_data(
            input_csv=input_file,
            output_csv=output_file,
            model_path=model_file,
            train=True,  # Set to False if using pre-trained model
            epochs=3,
            batch_size=32
        )
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
