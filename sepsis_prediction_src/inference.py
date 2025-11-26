"""
Inference Script for Sepsis Prediction

Load trained model and make predictions on new clinical notes
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer

from model import MeditronForMultiLabelClassification


class SepsisPredictor:
    """Predictor for sepsis risk assessment"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run on ('cuda' or 'cpu')
            threshold: Decision threshold for binary prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.label_names = ['h6', 'h12', 'h24', 'h48']
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Load model
        # Note: For inference, we need to reconstruct the model and load weights
        base_model_path = "/home/ubuntu/nahiyan_workspace/s3_bucket/models/meditron-7b"
        
        self.model = MeditronForMultiLabelClassification(
            model_path=base_model_path,
            num_labels=4
        )
        
        # Load LoRA adapters
        from peft import PeftModel
        self.model.base_model = PeftModel.from_pretrained(
            self.model.base_model.base_model,
            model_path
        )
        
        # Load classification head
        classifier_path = Path(model_path) / "classifier.pt"
        if classifier_path.exists():
            self.model.load_classifier(model_path)
        
        self.model.eval()
        self.model.to(self.device)
        
        print("✅ Model loaded successfully!")
    
    def predict(
        self,
        clinical_note: str,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict sepsis risk from clinical note
        
        Args:
            clinical_note: Patient clinical note text
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            clinical_note,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            logits = outputs['logits']
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Convert to binary predictions
        predictions = (probabilities >= self.threshold).astype(int)
        
        # Format output
        result = {
            'predictions': {
                label: int(pred)
                for label, pred in zip(self.label_names, predictions)
            }
        }
        
        if return_probabilities:
            result['probabilities'] = {
                label: float(prob)
                for label, prob in zip(self.label_names, probabilities)
            }
        
        # Risk assessment
        result['risk_assessment'] = self._assess_risk(predictions, probabilities)
        
        return result
    
    def _assess_risk(self, predictions: list, probabilities: list) -> Dict:
        """Assess overall risk level"""
        # Check if any positive predictions
        any_positive = any(predictions)
        
        if not any_positive:
            risk_level = "LOW"
            recommendation = "Continue routine monitoring"
        elif predictions[0] == 1:  # h6 positive
            risk_level = "CRITICAL"
            recommendation = "IMMEDIATE intervention required - High risk within 6 hours"
        elif predictions[1] == 1:  # h12 positive
            risk_level = "HIGH"
            recommendation = "Urgent attention needed - High risk within 12 hours"
        elif predictions[2] == 1:  # h24 positive
            risk_level = "MODERATE"
            recommendation = "Increased monitoring recommended - Risk within 24 hours"
        else:  # h48 positive
            risk_level = "ELEVATED"
            recommendation = "Close monitoring advised - Risk within 48 hours"
        
        # Earliest positive time horizon
        earliest_positive = None
        for label, pred in zip(self.label_names, predictions):
            if pred == 1:
                earliest_positive = label
                break
        
        return {
            'risk_level': risk_level,
            'earliest_positive_horizon': earliest_positive,
            'recommendation': recommendation,
            'max_probability': float(max(probabilities))
        }
    
    def predict_batch(
        self,
        clinical_notes: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Predict on multiple clinical notes
        
        Args:
            clinical_notes: List of clinical note texts
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(clinical_notes), batch_size):
            batch = clinical_notes[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                logits = outputs['logits']
                probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Process each sample
            for probs in probabilities:
                preds = (probs >= self.threshold).astype(int)
                
                result = {
                    'predictions': {
                        label: int(pred)
                        for label, pred in zip(self.label_names, preds)
                    },
                    'probabilities': {
                        label: float(prob)
                        for label, prob in zip(self.label_names, probs)
                    },
                    'risk_assessment': self._assess_risk(preds, probs)
                }
                
                results.append(result)
        
        return results
    
    def predict_from_jsonl(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 8
    ):
        """
        Predict on data from JSONL file
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to save predictions
            batch_size: Batch size for processing
        """
        print(f"Loading data from {input_file}...")
        
        # Load data
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"Processing {len(data)} samples...")
        
        # Extract texts
        texts = [record['input'] for record in data]
        
        # Predict
        predictions = self.predict_batch(texts, batch_size)
        
        # Combine with original data
        for record, pred in zip(data, predictions):
            record['model_prediction'] = pred
        
        # Save results
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
        
        print(f"✅ Predictions saved to {output_file}")
        
        # Print summary statistics
        self._print_prediction_summary(predictions)
    
    def _print_prediction_summary(self, predictions: List[Dict]):
        """Print summary of predictions"""
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        
        total = len(predictions)
        
        # Count predictions per label
        label_counts = {label: 0 for label in self.label_names}
        risk_levels = {'LOW': 0, 'ELEVATED': 0, 'MODERATE': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for pred in predictions:
            for label, value in pred['predictions'].items():
                label_counts[label] += value
            
            risk_level = pred['risk_assessment']['risk_level']
            risk_levels[risk_level] += 1
        
        print("\nPositive Predictions per Label:")
        print("-"*60)
        for label, count in label_counts.items():
            pct = (count / total) * 100
            print(f"{label:5s}: {count:5d} ({pct:5.2f}%)")
        
        print("\nRisk Level Distribution:")
        print("-"*60)
        for level, count in risk_levels.items():
            pct = (count / total) * 100
            print(f"{level:10s}: {count:5d} ({pct:5.2f}%)")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Sepsis prediction inference")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input JSONL file or single clinical note'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for predictions'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Single clinical note text for prediction'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SepsisPredictor(
        model_path=args.model_path,
        device=args.device,
        threshold=args.threshold
    )
    
    # Run prediction
    if args.text:
        # Single text prediction
        print("\n" + "="*60)
        print("PREDICTING ON SINGLE TEXT")
        print("="*60)
        print(f"\nClinical Note:\n{args.text[:200]}...\n")
        
        result = predictor.predict(args.text)
        
        print("\nPrediction Results:")
        print("-"*60)
        print(json.dumps(result, indent=2))
        
    elif args.input:
        # JSONL file prediction
        if not args.output:
            args.output = args.input.replace('.jsonl', '_predictions.jsonl')
        
        predictor.predict_from_jsonl(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size
        )
    
    else:
        print("Please provide either --text or --input argument")


if __name__ == "__main__":
    main()
