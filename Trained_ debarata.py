from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class PredictionResult:
    attribute_name: str
    description: str
    label: str
    score: float
    top_hypothesis: str

class SensitivityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, hypotheses):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hypotheses = hypotheses
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Get the hypotheses for this label
        label_hypotheses = self.hypotheses[label]
        
        # Tokenize with positive hypothesis
        encodings = self.tokenizer(
            text,
            label_hypotheses[0],  # Use first hypothesis of correct label
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'label': torch.tensor(1)  # 1 for entailment
        }

class CSVSensitivityClassifier:
    def __init__(self, model_name="microsoft/deberta-v3-large-mnli", batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        
        logging.info(f"Model loaded on {self.device}")

        # Organizational definition-based hypotheses
        self.hypotheses = {
            'sensitive_pii': [
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
            'confidential': [
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
            'licensed': [
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
            'non_sensitive_pii': [
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

    def train(self, train_data: pd.DataFrame, epochs=3, learning_rate=2e-5, validation_data=None):
        """
        Fine-tune the model on training data
        train_data: DataFrame with columns [attribute_name, description, label]
        """
        self.model.train()
        
        # Prepare training texts
        train_texts = [
            f"{row['attribute_name']}: {row['description']}"
            for _, row in train_data.iterrows()
        ]
        train_labels = train_data['label'].tolist()
        
        # Create dataset
        train_dataset = SensitivityDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.hypotheses
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Prepare optimizer and scheduler
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
            self.model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
            
            # Validation if provided
            if validation_data is not None:
                self.evaluate(validation_data)

    def evaluate(self, eval_data: pd.DataFrame):
        """Evaluate the model on validation data"""
        self.model.eval()
        predictions = self.process_csv(eval_data)
        
        # Calculate accuracy if 'label' column exists
        if 'label' in eval_data.columns:
            accuracy = np.mean(predictions['predicted_label'] == eval_data['label'])
            logging.info(f"Validation Accuracy: {accuracy:.4f}")
        
        return predictions

    def process_csv(self, input_data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Process a DataFrame containing attribute_name and description columns
        """
        try:
            # Validate columns
            required_columns = ['attribute_name', 'description']
            if not all(col in input_data.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
            # Create combined texts for processing
            texts = []
            for _, row in input_data.iterrows():
                # Clean and combine attribute name and description
                attr_name = str(row['attribute_name']).strip()
                desc = str(row['description']).strip()
                combined_text = f"{attr_name}: {desc}"
                texts.append(combined_text)
            
            # Get predictions
            results = self.predict_batch(texts, input_data['attribute_name'].tolist(), input_data['description'].tolist())
            
            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    'attribute_name': r.attribute_name,
                    'description': r.description,
                    'predicted_label': r.label,
                    'confidence_score': r.score,
                    'matching_hypothesis': r.top_hypothesis
                }
                for r in results
            ])
            
            # Save if output path provided
            if output_path:
                results_df.to_csv(output_path, index=False)
                logging.info(f"Results saved to {output_path}")
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise

    def predict_batch(self, texts: List[str], attr_names: List[str], descriptions: List[str]) -> List[PredictionResult]:
        self.model.eval()
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_attr_names = attr_names[i:i + self.batch_size]
            batch_descriptions = descriptions[i:i + self.batch_size]
            batch_results = self._process_batch(batch_texts, batch_attr_names, batch_descriptions)
            results.extend(batch_results)
            
        return results

    def _process_batch(self, texts: List[str], attr_names: List[str], descriptions: List[str]) -> List[PredictionResult]:
        batch_results = []
        
        with torch.no_grad():
            for text, attr_name, description in zip(texts, attr_names, descriptions):
                max_score = -1
                final_label = ""
                final_hypothesis = ""
                
                for label, hypothesis_list in self.hypotheses.items():
                    # Process all hypotheses for this label
                    inputs = self.tokenizer(
                        [text] * len(hypothesis_list),
                        hypothesis_list,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    
                    # Get entailment scores
                    entailment_scores = torch.softmax(outputs.logits, dim=1)[:, -1]
                    
                    # Get max score for this label
                    label_max_score = entailment_scores.max().item()
                    max_hypothesis_idx = entailment_scores.argmax().item()
                    
                    if label_max_score > max_score:
                        max_score = label_max_score
                        final_label = label
                        final_hypothesis = hypothesis_list[max_hypothesis_idx]
                
                batch_results.append(PredictionResult(
                    attribute_name=attr_name,
                    description=description,
                    label=final_label,
                    score=max_score,
                    top_hypothesis=final_hypothesis
                ))
                
        return batch_results

    def save_model(self, path: str):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logging.info(f"Model loaded from {path}")

# Example usage:
"""
# Initialize classifier
classifier = CSVSensitivityClassifier()

# Training data
train_df = pd.DataFrame({
    'attribute_name': ['customer_ssn', 'trading_strategy', 'bloomberg_feed'],
    'description': [
        'Social security number for identification',
        'Proprietary trading algorithm',
        'Market data from Bloomberg'
    ],
    'label': ['sensitive_pii', 'confidential', 'licensed']
})

# Train the model
classifier.train(train_df, epochs=3)

# Save the model
classifier.save_model('path/to/save/model')

# Later, load the model
classifier.load_model('path/to/save/model')

# Make predictions
test_df = pd.DataFrame({
    'attribute_name': ['user_email', 'trade_book'],
    'description': [
        'Customer email address',
        'Trading book positions'
    ]
})

results = classifier.process_csv(test_df)
print(results)
"""
