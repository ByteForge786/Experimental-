from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import logging

@dataclass
class PredictionResult:
    attribute_name: str
    description: str
    label: str
    score: float
    top_hypothesis: str

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

    def process_csv(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Process a CSV file containing attribute_name and description columns
        """
        try:
            # Read CSV
            df = pd.read_csv(input_path)
            
            # Validate columns
            required_columns = ['attribute_name', 'description']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Create combined texts for processing
            texts = []
            for _, row in df.iterrows():
                # Clean and combine attribute name and description
                attr_name = str(row['attribute_name']).strip()
                desc = str(row['description']).strip()
                combined_text = f"{attr_name}: {desc}"
                texts.append(combined_text)
            
            # Get predictions
            results = self.predict_batch(texts, df['attribute_name'].tolist(), df['description'].tolist())
            
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
            logging.error(f"Error processing CSV: {str(e)}")
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

# Example usage:
"""
# Initialize classifier
classifier = CSVSensitivityClassifier()

# Process CSV file
# Input CSV should have columns: attribute_name, description
results_df = classifier.process_csv(
    input_path='attributes.csv',
    output_path='sensitivity_predictions.csv'
)

# Results DataFrame will have columns:
# - attribute_name
# - description
# - predicted_label
# - confidence_score
# - matching_hypothesis
"""
