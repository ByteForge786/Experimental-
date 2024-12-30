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
    """Class to hold sensitivity label definitions"""
    
    DEFINITIONS = {
        "Sensitive PII": "High-risk personal data requiring stringent protection",
        "Non-Sensitive PII": "Basic personal information like contact details",
        "Confidential Information": "Non-public business and operational information",
        "Licensed Data": "Third-party sourced data with usage agreements"
    }

def create_batch_prompt(attributes: List[Dict[str, str]]) -> str:
    """Creates a simplified prompt for batch processing."""
    return f"""Analyze these attributes and classify each as either: Sensitive PII, Non-Sensitive PII, Confidential Information, or Licensed Data.
Provide ONLY predictions in this exact format, nothing else:
{{
    "predictions": [
        {{
            "predicted_label": "<LABEL>",
            "reasoning": "<ONE_LINE_REASON>"
        }}
    ]
}}

Attributes to classify:
{json.dumps(attributes, indent=2)}"""

def validate_and_parse_response(response: str, batch_size: int) -> List[Tuple[str, str]]:
    """Validates and parses LLM response with enhanced error handling."""
    try:
        # Clean the response
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Try JSON parsing first
        try:
            response_dict = json.loads(cleaned_response)
            predictions = response_dict.get("predictions", [])
            
            if not predictions:
                raise ValueError("No predictions found")
                
            valid_labels = {
                "Sensitive PII", 
                "Non-Sensitive PII", 
                "Confidential Information", 
                "Licensed Data"
            }
            
            result = []
            for pred in predictions:
                label = pred.get("predicted_label", "ERROR")
                reasoning = pred.get("reasoning", "No reasoning provided")
                
                if label not in valid_labels:
                    logger.warning(f"Invalid label found: {label}")
                    label = "ERROR"
                    
                result.append((label, reasoning))
            
            # Handle size mismatch
            if len(result) < batch_size:
                result.extend([("ERROR", "Missing prediction")] * (batch_size - len(result)))
            return result[:batch_size]
            
        except json.JSONDecodeError:
            # Fallback to regex parsing
            pattern = r'"predicted_label":\s*"([^"]+)".*?"reasoning":\s*"([^"]+)"'
            matches = re.findall(pattern, cleaned_response)
            
            if matches:
                logger.warning("Used regex fallback for parsing")
                result = [(label, reasoning) for label, reasoning in matches]
                
                if len(result) < batch_size:
                    result.extend([("ERROR", "Missing prediction")] * (batch_size - len(result)))
                return result[:batch_size]
            
            raise ValueError("Failed to parse response with regex")
            
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return [("ERROR", "Parsing error")] * batch_size

def process_batch(batch_df: pd.DataFrame, retry_count: int = 3) -> List[Tuple[str, str]]:
    """Process a batch with retries and size reduction."""
    batch_size = len(batch_df)
    
    for attempt in range(retry_count):
        try:
            attributes = batch_df[['attribute_name', 'description']].to_dict('records')
            prompt = create_batch_prompt(attributes)
            response = chinou_response(prompt)
            
            # Validate response structure
            if not response or '"predictions":' not in response:
                raise ValueError("Invalid response structure")
                
            results = validate_and_parse_response(response, batch_size)
            
            # Verify results
            if len(results) == batch_size and all(label != "ERROR" for label, _ in results):
                return results
                
            # If we got some results but with errors, and batch size > 1
            if batch_size > 1:
                logger.warning(f"Retrying with smaller batch size. Attempt {attempt + 1}")
                mid = batch_size // 2
                first_half = process_batch(batch_df.iloc[:mid], retry_count=1)
                second_half = process_batch(batch_df.iloc[mid:], retry_count=1)
                return first_half + second_half
                
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            if attempt == retry_count - 1:  # Last attempt
                return [("ERROR", "Processing failed")] * batch_size
            
    return [("ERROR", "Max retries reached")] * batch_size

def predict_sensitivity_labels_batch(
    input_csv: str,
    batch_size: int = 10,
    max_workers: int = 8,
    checkpoint_interval: int = 1000
) -> pd.DataFrame:
    """Optimized batch prediction function."""
    logger.info(f"Starting batch prediction for {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
        logger.info(f"Loaded {len(df)} rows")
        
        all_predictions = []
        all_reasonings = []
        processed_count = 0
        
        # Process in chunks for better memory management
        chunks = [df[i:i + checkpoint_interval] for i in range(0, len(df), checkpoint_interval)]
        
        for chunk_idx, chunk_df in enumerate(chunks):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for start_idx in range(0, len(chunk_df), batch_size):
                    batch = chunk_df.iloc[start_idx:start_idx + batch_size]
                    futures.append(executor.submit(process_batch, batch))
                
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        batch_results = future.result()
                        all_predictions.extend([label for label, _ in batch_results])
                        all_reasonings.extend([reasoning for _, reasoning in batch_results])
                        processed_count += len(batch_results)
                    except Exception as e:
                        logger.error(f"Future processing error: {str(e)}")
            
            # Save checkpoint
            if chunk_idx < len(chunks) - 1:  # Don't checkpoint on last chunk
                temp_df = df.iloc[:processed_count].copy()
                temp_df['predicted_label'] = all_predictions
                temp_df['reasoning'] = all_reasonings
                temp_df.to_csv(f'checkpoint_{processed_count}.csv', index=False)
                logger.info(f"Saved checkpoint at {processed_count} records")
        
        df['predicted_label'] = all_predictions[:len(df)]
        df['reasoning'] = all_reasonings[:len(df)]
        
        return df
        
    except Exception as e:
        logger.error(f"Prediction process failed: {str(e)}")
        raise

def calculate_accuracy_metrics(predicted_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate accuracy metrics per label."""
    metrics = {}
    
    try:
        for label in predicted_df['label'].unique():
            mask = predicted_df['label'] == label
            correct = (predicted_df[mask]['predicted_label'] == predicted_df[mask]['label']).sum()
            total = mask.sum()
            accuracy = (correct / total) if total > 0 else 0
            metrics[label] = {
                'accuracy': accuracy,
                'total_cases': total,
                'correct_predictions': correct
            }
            
        # Overall accuracy
        total_correct = (predicted_df['predicted_label'] == predicted_df['label']).sum()
        metrics['overall'] = {
            'accuracy': total_correct / len(predicted_df),
            'total_cases': len(predicted_df),
            'correct_predictions': total_correct
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        
    return metrics

def main(
    input_csv: str,
    batch_size: int = 10,
    max_workers: int = 8,
    checkpoint_interval: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Main pipeline function."""
    try:
        predicted_df = predict_sensitivity_labels_batch(
            input_csv,
            batch_size=batch_size,
            max_workers=max_workers,
            checkpoint_interval=checkpoint_interval
        )
        
        metrics = calculate_accuracy_metrics(predicted_df)
        
        # Save final results
        predicted_df.to_csv(f'final_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        
        return predicted_df, metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results_df, accuracy = main(
            "input.csv",
            batch_size=10,
            max_workers=8,
            checkpoint_interval=1000
        )
        
        logger.info("\nAccuracy Metrics:")
        for label, metrics in accuracy.items():
            logger.info(f"{label}: {metrics['accuracy']:.2%} ({metrics['correct_predictions']}/{metrics['total_cases']})")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)
