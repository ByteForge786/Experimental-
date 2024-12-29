import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Generator
from dataclasses import dataclass
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from functools import partial
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sensitivity_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class SensitivityDefinitions:
    """Class to hold sensitivity label definitions and hypotheses"""
    
    # Core definitions [Same as previous]
    DEFINITIONS = {
        "SENSITIVE_PII": """
        High-risk personal data requiring stringent protection, including:
        - Identity: SSN, passport, licenses, biometric data
        - Protected characteristics: race, ethnicity, religion, political views
        - Health & personal: medical, genetic, sexual orientation
        - Financial: credit records, financial accounts
        - Special categories: union membership, criminal history
        - Regional specifics: Japan My Number, Korean PCI/PII
        - Digital security: credentials, geolocation
        - Private communications content
        """,
        
        "CONFIDENTIAL": """
        Non-public business and operational information including:
        - Client data: orders, positions, personal details
        - Business strategy: research, trading positions, plans
        - Intellectual property: models, programs, development
        - Internal operations: systems, contracts, metadata
        - Third-party private info: regulatory, exchange data
        - Investment details: holdings, ownership structures
        """,
        
        "LICENSED": """
        Third-party sourced data with:
        - Commercial usage agreements
        - Vendor restrictions and requirements
        - Attribution obligations
        - Distribution limitations
        - Subscription-based access
        """,
        
        "NON_SENSITIVE_PII": """
        Basic personal information including:
        - Contact details: email, phone, address
        - Professional info: job title, company
        - General demographics: name, age
        - Public record information
        - Business contact details
        """
    }
    
    # Hypotheses [Same as previous]
    HYPOTHESES = {
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
            "This data contains proprietary models or programs",
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

def create_batch_prompt(attributes: List[Dict[str, str]], definitions: SensitivityDefinitions) -> str:
    """
    Creates a structured prompt for batch processing multiple attributes.
    """
    return f"""As a data sensitivity expert, analyze the following batch of attributes and classify each into the most appropriate sensitivity label. Use both the provided definitions and your knowledge of data privacy and security.

Definitions:
{json.dumps(definitions.DEFINITIONS, indent=2)}

Common characteristics for each label:
{json.dumps(definitions.HYPOTHESES, indent=2)}

Attributes to classify:
{json.dumps(attributes, indent=2)}

For each attribute, provide classification in the following format:
{{
    "predictions": [
        {{
            "attribute_name": "<NAME>",
            "predicted_label": "<LABEL>",
            "reasoning": "<CONCISE_REASONING>"
        }},
        ...
    ]
}}

The label must be one of: SENSITIVE_PII, NON_SENSITIVE_PII, CONFIDENTIAL, or LICENSED.
Ensure predictions array matches the order of input attributes.
"""

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    """
    Yields chunks of the dataframe for batch processing.
    """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]

def parse_batch_response(response: str, expected_count: int) -> List[Tuple[str, str]]:
    """
    Parses the LLM batch response to extract predictions and reasoning.
    """
    try:
        # Clean the response if needed
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3]
            
        response_dict = json.loads(cleaned_response)
        predictions = response_dict.get("predictions", [])
        
        if len(predictions) != expected_count:
            raise ValueError(f"Expected {expected_count} predictions, got {len(predictions)}")
            
        # Validate and extract predictions
        valid_labels = {"SENSITIVE_PII", "NON_SENSITIVE_PII", "CONFIDENTIAL", "LICENSED"}
        result = []
        
        for pred in predictions:
            label = pred["predicted_label"].upper()
            if label not in valid_labels:
                raise ValueError(f"Invalid label: {label}")
            result.append((label, pred["reasoning"]))
            
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise ValueError(f"Invalid LLM response format: {response}")

def predict_sensitivity_labels_batch(
    input_csv: str,
    llm_response_func,
    definitions: SensitivityDefinitions,
    batch_size: int = 50,
    max_workers: int = 4,
    checkpoint_interval: int = 1000
) -> pd.DataFrame:
    """
    Predicts sensitivity labels for attributes in batches with parallel processing.
    """
    logger.info(f"Starting batch prediction process for {input_csv}")
    
    try:
        # Read input CSV
        df = pd.read_csv(input_csv)
        logger.info(f"Successfully loaded input CSV with {len(df)} rows")
        
        # Initialize results storage
        all_predictions = []
        all_reasonings = []
        
        # Create a partial function for processing batches
        def process_batch(batch_df: pd.DataFrame) -> List[Tuple[str, str]]:
            attributes = batch_df[['attribute_name', 'description']].to_dict('records')
            prompt = create_batch_prompt(attributes, definitions)
            response = llm_response_func(prompt)
            return parse_batch_response(response, len(attributes))
        
        # Process in batches with parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            checkpoint_data = []
            
            # Submit batch jobs
            for chunk in tqdm(chunk_dataframe(df, batch_size), total=len(df)//batch_size + 1):
                futures.append(executor.submit(process_batch, chunk))
                
                # Save checkpoint if needed
                if len(all_predictions) >= checkpoint_interval:
                    temp_df = df.iloc[:len(all_predictions)].copy()
                    temp_df['predicted_label'] = all_predictions
                    temp_df['reasoning'] = all_reasonings
                    checkpoint_path = f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    temp_df.to_csv(checkpoint_path, index=False)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    batch_results = future.result()
                    for label, reasoning in batch_results:
                        all_predictions.append(label)
                        all_reasonings.append(reasoning)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    all_predictions.extend(["ERROR"] * batch_size)
                    all_reasonings.extend([f"Error: {str(e)}"] * batch_size)
        
        # Trim results to match dataframe length
        all_predictions = all_predictions[:len(df)]
        all_reasonings = all_reasonings[:len(df)]
        
        # Add predictions to DataFrame
        df['predicted_label'] = all_predictions
        df['reasoning'] = all_reasonings
        
        # Save final results
        output_path = f'predicted_sensitivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Fatal error in prediction process: {e}")
        raise

def calculate_accuracy_metrics(predicted_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    [Same as previous implementation]
    """
    logger.info("Calculating accuracy metrics")
    
    try:
        metrics = {
            'per_label': {},
            'confusion_matrix': {},
            'overall': {}
        }
        
        all_labels = predicted_df['original_label'].unique()
        
        for label in all_labels:
            label_mask = predicted_df['original_label'] == label
            correct_predictions = (
                predicted_df[label_mask]['predicted_label'] == 
                predicted_df[label_mask]['original_label']
            ).sum()
            total_predictions = label_mask.sum()
            
            metrics['per_label'][label] = {
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
                'total_cases': total_predictions,
                'correct_predictions': correct_predictions
            }
        
        # Calculate confusion matrix
        for true_label in all_labels:
            metrics['confusion_matrix'][true_label] = {}
            true_label_mask = predicted_df['original_label'] == true_label
            
            for pred_label in all_labels:
                pred_count = (predicted_df[true_label_mask]['predicted_label'] == pred_label).sum()
                metrics['confusion_matrix'][true_label][pred_label] = pred_count
        
        # Calculate overall metrics
        total_correct = (predicted_df['predicted_label'] == predicted_df['original_label']).sum()
        total_cases = len(predicted_df)
        
        metrics['overall'] = {
            'accuracy': total_correct / total_cases if total_cases > 0 else 0,
            'total_cases': total_cases,
            'correct_predictions': total_correct
        }
        
        logger.info("=== Accuracy Metrics ===")
        logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
        for label, stats in metrics['per_label'].items():
            logger.info(f"{label} Accuracy: {stats['accuracy']:.2%} ({stats['correct_predictions']}/{stats['total_cases']})")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating accuracy metrics: {e}")
        raise

def main(
    input_csv: str,
    llm_response_func,
    batch_size: int = 50,
    max_workers: int = 4,
    checkpoint_interval: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Main function to run the end-to-end pipeline with batch processing.
    """
    logger.info("Starting sensitivity prediction pipeline with batch processing")
    
    try:
        # Initialize definitions
        definitions = SensitivityDefinitions()
        
        # Run batch prediction
        predicted_df = predict_sensitivity_labels_batch(
            input_csv,
            llm_response_func,
            definitions,
            batch_size=batch_size,
            max_workers=max_workers,
            checkpoint_interval=checkpoint_interval
        )
        
        # Calculate accuracy
        accuracy_metrics = calculate_accuracy_metrics(predicted_df)
        
        return predicted_df, accuracy_metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# Example usage:
if __name__ == "__main__":
    def mock_llm_response(prompt: str) -> str:
        # Simulate batch response
        attributes = json.loads(prompt.split("Attributes to classify:")[1].split("For each")[0])
        predictions = [
            {
                "attribute_name": attr["attribute_name"],
                "predicted_label": "NON_SENSITIVE_PII",
                "reasoning": "Mock reasoning"
            }
            for attr in attributes
        ]
        return json.dumps({"predictions": predictions})
    
    try:
        # Run the pipeline with batch processing
        results_df, accuracy = main(
            "input.csv",
            mock_llm_response,
            batch_size=50,
            max_workers=4,
            checkpoint_interval=1000
        )
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
