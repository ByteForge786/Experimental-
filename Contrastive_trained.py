def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
    """Optimized batch prediction"""
    batch_size = self.batch_size
    results = []
    
    # Process in batches
    for start_idx in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[start_idx:start_idx + batch_size]
        
        # Batch encode all texts at once
        texts = [f"{row['attribute_name']}: {row['description']}" 
                for _, row in batch_df.iterrows()]
        
        encoded = self.tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings for entire batch
        with torch.no_grad():
            batch_embeddings = self.model.encode(
                encoded['input_ids'],
                encoded['attention_mask']
            )
            
            # Calculate similarities for batch
            batch_results = []
            for text_emb in batch_embeddings:
                scores = {}
                for label, def_embeddings in self.definition_embeddings.items():
                    label_scores = [
                        torch.nn.functional.cosine_similarity(
                            text_emb.unsqueeze(0), def_emb
                        ).item()
                        for def_emb in def_embeddings
                    ]
                    scores[label] = max(label_scores)
                batch_results.append(scores)
            
            results.extend(batch_results)


import torch
from torch import nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import sys
from sklearn.metrics import classification_report
import os

# Configure logging
log_filename = f'sensitivity_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SensitivityDefinitions:
    """Individual definitions for each sensitivity label"""
    DEFINITIONS = {
        'SENSITIVE_PII': [
            "This data reveals racial or ethnic origin of a person",
            "This data contains political opinions or religious beliefs",
            "This data includes health, genetic, or biometric information",
            "This data contains unique identification numbers like SSN, passport, or licenses",
            "This data includes financial or credit record information",
            "This data reveals precise geolocation of a person",
            "This data contains login credentials or passwords",
            "This data includes information about trade union membership",
            "This data contains information about sexual orientation or activities",
            "This data includes criminal history or victim information",
            "This data contains Japan-specific identifiers like My Number",
            "This data includes Korean PCI or PII information",
            "This data reveals veteran status of a person",
            "This data contains biometric identifiers",
            "This data includes private messages or communication content"
        ],
        'CONFIDENTIAL': [
            "This is non-public business information",
            "This data contains internal proprietary information",
            "This data includes trading strategies or positions",
            "This data contains research opinions or strategic plans",
            "This data includes intellectual property or product development plans",
            "This data contains client orders or positions",
            "This data includes non-public client information",
            "This data contains system access codes or passwords",
            "This data includes proprietary models or programs",
            "This data contains non-public contract information",
            "This data includes information from regulators or exchanges",
            "This data contains syndicate member information",
            "This data includes non-public ownership information",
            "This data contains internal business metadata",
            "This data includes private company investments"
        ],
        'LICENSED': [
            "This data is sourced from third-party providers",
            "This data is obtained under commercial agreement",
            "This data has vendor-specific usage restrictions",
            "This data is provided through licensing agreements",
            "This data requires vendor attribution",
            "This data has commercial usage limitations",
            "This data is obtained through paid subscriptions",
            "This data includes third-party proprietary content",
            "This data requires vendor-specific handling",
            "This data has commercial distribution restrictions"
        ],
        'NON_SENSITIVE_PII': [
            "This data contains basic contact information",
            "This data includes standard personal identifiers",
            "This data contains publicly available personal information",
            "This data includes basic demographic information",
            "This data contains business contact details",
            "This data includes professional information",
            "This data contains general identification information",
            "This data includes standard contact methods",
            "This data contains regular personal details",
            "This data includes common personal attributes"
        ]
    }

class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with individual definitions"""
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.definitions = SensitivityDefinitions.DEFINITIONS
        self.labels = list(self.definitions.keys())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"{row['attribute_name']}: {row['description']}"
        
        # Get positive label and randomly select one of its definitions
        pos_label = row['original_label'].upper()
        pos_definition = np.random.choice(self.definitions[pos_label])
        
        # Get negative label and randomly select one of its definitions
        neg_labels = [l for l in self.labels if l != pos_label]
        neg_label = np.random.choice(neg_labels)
        neg_definition = np.random.choice(self.definitions[neg_label])
        
        # Tokenize
        anchor = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        positive = self.tokenizer(
            pos_definition,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        negative = self.tokenizer(
            neg_definition,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'anchor_ids': anchor['input_ids'].squeeze(),
            'anchor_mask': anchor['attention_mask'].squeeze(),
            'positive_ids': positive['input_ids'].squeeze(),
            'positive_mask': positive['attention_mask'].squeeze(),
            'negative_ids': negative['input_ids'].squeeze(),
            'negative_mask': negative['attention_mask'].squeeze(),
        }

class ContrastiveSensitivityModel(nn.Module):
    def __init__(self, model_name: str = "microsoft/mpnet-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Project to lower dimension for efficiency
        self.projection = nn.Linear(self.encoder.config.hidden_size, 256)
        
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Project and normalize
        projected = self.projection(embeddings)
        return torch.nn.functional.normalize(projected, p=2, dim=1)
    
    def forward(self, anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask):
        anchor_emb = self.encode(anchor_ids, anchor_mask)
        positive_emb = self.encode(positive_ids, positive_mask)
        negative_emb = self.encode(negative_ids, negative_mask)
        return anchor_emb, positive_emb, negative_emb

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum((anchor - positive) ** 2, dim=1)
        distance_negative = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class SensitivityPredictor:
    def __init__(self, 
                 model_name: str = "microsoft/mpnet-base", 
                 batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ContrastiveSensitivityModel(model_name).to(self.device)
        self.batch_size = batch_size
        self.definitions = SensitivityDefinitions.DEFINITIONS
        
        # Cache for definition embeddings (storing list of embeddings per label)
        self.definition_embeddings = {}
        
        logger.info(f"Model initialized on {self.device}")
    
    def cache_definition_embeddings(self):
        """Cache embeddings for each individual definition"""
        logger.info("Caching individual definition embeddings...")
        self.model.eval()
        
        try:
            with torch.no_grad():
                for label, definitions in self.definitions.items():
                    self.definition_embeddings[label] = []
                    
                    for definition in tqdm(definitions, desc=f"Caching {label} definitions"):
                        def_encoded = self.tokenizer(
                            definition,
                            max_length=512,
                            truncation=True,
                            padding='max_length',
                            return_tensors='pt'
                        ).to(self.device)
                        
                        def_emb = self.model.encode(
                            def_encoded['input_ids'],
                            def_encoded['attention_mask']
                        )
                        self.definition_embeddings[label].append(def_emb)
                    
            logger.info("Individual definition embeddings cached successfully")
            
        except Exception as e:
            logger.error(f"Error caching definition embeddings: {str(e)}")
            raise
    
    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using cached individual definition embeddings"""
        try:
            logger.info("Starting prediction...")
            self.model.eval()
            
            # Ensure definitions are cached
            if not self.definition_embeddings:
                self.cache_definition_embeddings()
            
            results = []
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
                with torch.no_grad():
                    # Encode input text
                    text = f"{row['attribute_name']}: {row['description']}"
                    text_encoded = self.tokenizer(
                        text,
                        max_length=512,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    ).to(self.device)
                    
                    text_emb = self.model.encode(
                        text_encoded['input_ids'],
                        text_encoded['attention_mask']
                    )
                    
                    # Calculate similarity with all cached definition embeddings
                    scores = {}
                    best_definitions = {}
                    
                    for label, def_embeddings in self.definition_embeddings.items():
                        # Get similarities with all definitions for this label
                        label_scores = [
                            torch.nn.functional.cosine_similarity(text_emb, def_emb).item()
                            for def_emb in def_embeddings
                        ]
                        # Use max similarity for the label
                        max_score_idx = np.argmax(label_scores)
                        scores[label] = label_scores[max_score_idx]
                        best_definitions[label] = self.definitions[label][max_score_idx]
                    
                    # Get predicted label
                    predicted_label = max(scores.items(), key=lambda x: x[1])[0]
                    
                    results.append({
                        'attribute_name': row['attribute_name'],
                        'description': row['description'],
                        'original_label': row['original_label'].upper(),
                        'predicted_label': predicted_label,
                        'matching_definition': best_definitions[predicted_label],
                        **{f'{label.lower()}_score': score for label, score in scores.items()}
                    })
            
            # Create output DataFrame
            output_df = pd.DataFrame(results)
            
            # Calculate and log metrics
            accuracy = (output_df['predicted_label'] == output_df['original_label']).mean()
            logger.info(f"Overall Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(
                output_df['original_label'],
                output_df['predicted_label']
            ))
            
            return output_df
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def train(self, train_df: pd.DataFrame, epochs: int = 3, learning_rate: float = 2e-5):
        """Train the model using contrastive learning"""
        try:
            logger.info("Starting training...")
            
            train_dataset = ContrastiveDataset(train_df, self.tokenizer)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            criterion = ContrastiveLoss()
            
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                
                for batch in progress_bar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    anchor_emb, positive_emb, negative_emb = self.model(
                        batch['anchor_ids'],
                        batch['anchor_mask'],
                        batch['positive_ids'],
                        batch['positive_mask'],
                        batch['negative_ids'],
                        batch['negative_mask']
                    )
                    
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Cache definition embeddings after training
            self.cache_definition_embeddings()
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self, path: str):
        """Save model and cached embeddings"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'definition_embeddings': self.definition_embeddings
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load model and cached embeddings"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.definition_embeddings = checkpoint['definition_embeddings']
            logger.info(f"Model loaded from {path}")
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
    End-to-end pipeline for sensitivity prediction using contrastive learning
    """
    try:
        # Load data
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Validate required columns
        required_columns = ['attribute_name', 'description', 'original_label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Initialize model
        predictor = SensitivityPredictor(batch_size=batch_size)
        
        # Train or load model
        if train:
            logger.info("Starting training pipeline")
            predictor.train(df, epochs=epochs)
            if model_path:
                predictor.save_model(model_path)
        elif model_path:
            logger.info(f"Loading pre-trained model from {model_path}")
            predictor.load_model(model_path)
        else:
            raise ValueError("Either train must be True or model_path must be provided")
        
        # Make predictions
        results_df = predictor.predict(df)
        
        # Save results if output path provided
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            logger.info(f"Results saved to {output_csv}")
        
        # Log final metrics
        logger.info("\nFinal Results:")
        accuracy = (results_df['predicted_label'] == results_df['original_label']).mean()
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        
        # Detailed metrics per label
        for label in results_df['original_label'].unique():
            label_mask = results_df['original_label'] == label
            label_accuracy = (
                results_df[label_mask]['predicted_label'] == 
                results_df[label_mask]['original_label']
            ).mean()
            label_count = label_mask.sum()
            logger.info(f"{label} Accuracy: {label_accuracy:.4f} (Count: {label_count})")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        input_file = "your_input.csv"  # Replace with your input file path
        output_file = "predictions.csv"  # Replace with desired output path
        model_file = "models/sensitivity_model.pt"  # Replace with desired model path
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        
        # Process the data
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
        sys.exit(1)
