import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Generator
from dataclasses import dataclass
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

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
    """Creates a structured prompt for batch processing multiple attributes."""
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

def validate_llm_response(response: str) -> str:
    """Validates and cleans the LLM response before parsing."""
    if not response:
        raise ValueError("Empty response received from LLM")
        
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        logger.error(f"Invalid JSON structure in response: {response}")
        raise ValueError("Response does not contain valid JSON structure")
        
    json_str = response[start_idx:end_idx + 1]
    
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logger.error(f"JSON validation failed: {str(e)}")
        logger.error(f"Invalid JSON: {json_str}")
        raise

def parse_batch_response(response: str, expected_count: int) -> List[Tuple[str, str]]:
    """Parses the LLM batch response with enhanced error handling."""
    try:
        logger.debug(f"Raw response received: {response}")
        
        cleaned_response = response.strip()
        
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
        cleaned_response = cleaned_response.replace('\n', ' ').replace('\r', '')
        cleaned_response = re.sub(r'}\s*{', '},{', cleaned_response)
        
        try:
            response_dict = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {str(e)}")
            logger.error(f"Problematic JSON: {cleaned_response}")
            pattern = r'"predicted_label":\s*"([^"]+)".*?"reasoning":\s*"([^"]+)"'
            matches = re.findall(pattern, cleaned_response)
            if matches:
                logger.warning("Used regex fallback for parsing")
                return [(label.upper(), reasoning) for label, reasoning in matches[:expected_count]]
            raise

        predictions = response_dict.get("predictions", [])
        
        if len(predictions) != expected_count:
            logger.warning(f"Prediction count mismatch. Expected: {expected_count}, Got: {len(predictions)}")
            if len(predictions) < expected_count:
                predictions.extend([{
                    "predicted_label": "ERROR",
                    "reasoning": "Missing prediction"
                }] * (expected_count - len(predictions)))
            predictions = predictions[:expected_count]
            
        valid_labels = {"SENSITIVE_PII", "NON_SENSITIVE_PII", "CONFIDENTIAL", "LICENSED"}
        result = []
        
        for pred in predictions:
            label = pred.get("predicted_label", "ERROR").upper()
            reasoning = pred.get("reasoning", "No reasoning provided")
            
            if label not in valid_labels:
                logger.warning(f"Invalid label found: {label}")
                label = "ERROR"
                
            result.append((label, reasoning))
            
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in parse_batch_response: {str(e)}")
        logger.error(f"Problematic response: {response}")
        return [("ERROR", f"Parsing error: {str(e)}")] * expected_count

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
    """Yields chunks of the dataframe for batch processing."""
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]

def predict_sensitivity_labels_batch(
    input_csv: str,
    batch_size: int = 10,
    max_workers: int = 4,
    checkpoint_interval: int = 1000
) -> pd.DataFrame:
    """Predicts sensitivity labels for attributes in batches with parallel processing."""
    logger.info(f"Starting batch prediction process for {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Successfully loaded input CSV with {len(df)} rows")
        
        all_predictions = []
        all_reasonings = []
        
        def process_batch(batch_df: pd.DataFrame) -> List[Tuple[str, str]]:
            try:
                attributes = batch_df[['attribute_name', 'description']].to_dict('records')
                prompt = create_batch_prompt(attributes, definitions)
                response = chinou_response(prompt)
                
                if '"predictions": [' in response and not response.strip().endswith('}'):
                    logger.warning("Incomplete response detected - retrying with smaller batch")
                    if len(batch_df) > 5:
                        mid = len(batch_df) // 2
                        first_half = process_batch(batch_df.iloc[:mid])
                        second_half = process_batch(batch_df.iloc[mid:])
                        return first_half + second_half
                    else:
                        results = []
                        for _, row in batch_df.iterrows():
                            single_result = process_batch(pd.DataFrame([row]))
                            results.extend(single_result)
                        return results
                
                return parse_batch_response(response, len(attributes))
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                return [("ERROR", f"Processing error: {str(e)}")] * len(batch_df)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for chunk in tqdm(chunk_dataframe(df, batch_size), total=len(df)//batch_size + 1):
                futures.append(executor.submit(process_batch, chunk))
                
                if len(all_predictions) >= checkpoint_interval:
                    temp_df = df.iloc[:len(all_predictions)].copy()
                    temp_df['predicted_label'] = all_predictions
                    temp_df['reasoning'] = all_reasonings
                    checkpoint_path = f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    temp_df.to_csv(checkpoint_path, index=False)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    batch_results = future.result()
                    all_predictions.extend([label for label, _ in batch_results])
                    all_reasonings.extend([reasoning for _, reasoning in batch_results])
                except Exception as e:
                    logger.error(f"Error processing future: {e}")
        
        all_predictions = all_predictions[:len(df)]
        all_reasonings = all_reasonings[:len(df)]
        
        df['predicted_label'] = all_predictions
        df['reasoning'] = all_reasonings
        
        output_path = f'predicted_sensitivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Fatal error in prediction process: {e}")
        raise

def calculate_accuracy_metrics(predicted_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculates accuracy metrics for predictions."""
    logger.info("Calculating accuracy metrics")
    
    try:
        metrics = {
            'per_label': {},
            'confusion_matrix': {},
            'overall': {}
        }
        
        all_labels = predicted_df['label'].unique()
        
        for label in all_labels:
            label_mask = predicted_df['label'] == label
            correct_predictions = (
                predicted_df[label_mask]['predicted_label'] == 
                predicted_df[label_mask]['label']
            ).sum()
            total_predictions = label_mask.sum()
            
            metrics['per_label'][label] = {
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
                'total_cases': total_predictions,
                'correct_predictions': correct_predictions
            }
        
        for true_label in all_labels:
            metrics['confusion_matrix'][true_label] = {}
            true_label_mask = predicted_df['label'] == true_label
            
            for pred_label in all_labels:
                pred_count = (predicted_df[true_label_mask]['predicted_label'] == pred_label).sum()
                metrics['confusion_matrix'][true_label][pred_label] = pred_count
        
        total_correct = (predicted_df['predicted_label'] == predicted_df['label']).sum()
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
    batch_size: int = 10,
    max_workers: int = 4,
    checkpoint_interval: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Main function to run the end-to-end pipeline with batch processing."""
    logger.info("Starting sensitivity prediction pipeline with batch processing")
    
    try:
        # Initialize definitions
        definitions = SensitivityDefinitions()
        
        # Run batch prediction
        predicted_df = predict_sensitivity_labels_batch(
            input_csv,
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

if __name__ == "__main__":
    try:
        # Run the pipeline with batch processing
        results_df, accuracy = main(
            "input.csv",  # Replace with your input CSV path
            batch_size=10,
            max_workers=4,
            checkpoint_interval=1000
        )
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
