"""
Model Architecture: Meditron-7B with QLoRA and Classification Head

This module sets up the model for multi-label classification:
1. Load Meditron-7B with 4-bit quantization (QLoRA)
2. Add LoRA adapters
3. Add classification head for 4 binary outputs
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional, Dict
import warnings


class MeditronForMultiLabelClassification(nn.Module):
    """
    Meditron-7B with classification head for multi-label sepsis prediction
    
    Architecture:
    - Base: Meditron-7B (4-bit quantized)
    - LoRA: Applied to attention and MLP layers
    - Classification Head: Maps hidden states to 4 binary outputs
    """
    
    def __init__(
        self,
        model_path: str,
        num_labels: int = 4,
        lora_config: Optional[Dict] = None,
        quantization_config: Optional[Dict] = None,
        device_map: str = "auto"
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.model_path = model_path
        
        # Setup quantization config
        if quantization_config is None:
            quantization_config = {
                "enabled": True,
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            }
        
        # Check if quantization is enabled
        use_quantization = quantization_config.get("enabled", True)
        
        if use_quantization and (quantization_config.get("load_in_4bit") or quantization_config.get("load_in_8bit")):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_config.get("load_in_4bit", False),
                load_in_8bit=quantization_config.get("load_in_8bit", False),
                bnb_4bit_compute_dtype=quantization_config.get("bnb_4bit_compute_dtype", torch.bfloat16),
                bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True)
            )
            quant_type = "4-bit" if quantization_config.get("load_in_4bit") else "8-bit"
            print(f"Loading Meditron-7B with {quant_type} quantization (QLoRA)...")
        else:
            bnb_config = None
            print("Loading Meditron-7B in 16-bit (Regular LoRA)...")
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Prepare for k-bit training (only if quantized)
        if use_quantization and bnb_config is not None:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            print("Model prepared for k-bit training")
        
        # Setup LoRA config
        if lora_config is None:
            lora_config = {
                "r": 64,
                "lora_alpha": 128,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        print("Adding LoRA adapters...")
        self.base_model = get_peft_model(self.base_model, peft_config)
        
        # Print trainable parameters
        self.base_model.print_trainable_parameters()
        
        # Get hidden size from config
        self.hidden_size = self.base_model.config.hidden_size
        
        # Classification head
        print(f"Adding classification head (hidden_size={self.hidden_size} -> {num_labels} labels)...")
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        
        # Initialize classification head weights
        self._init_classifier_weights()
        
        # Move classifier to same device and dtype as base model
        if use_quantization:
            # For quantized models, move to CUDA with bfloat16
            self.classifier = self.classifier.to(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=torch.bfloat16
            )
        else:
            # For non-quantized models with device_map="auto", match base model
            # Use bfloat16 to match the base model dtype
            target_device = next(self.base_model.parameters()).device
            self.classifier = self.classifier.to(device=target_device, dtype=torch.bfloat16)
        
        print(f"Model setup complete!")
        print(f"  Classifier device: {next(self.classifier.parameters()).device}")
        print(f"  Classifier dtype: {next(self.classifier.parameters()).dtype}")
    
    def _init_classifier_weights(self):
        """Initialize classification head with proper weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, num_labels] (optional)
            
        Returns:
            Dictionary with:
            - logits: Raw predictions [batch_size, num_labels]
            - loss: Loss value (if labels provided)
            - hidden_states: Last hidden states (optional)
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state
        # Use the hidden state of the last token for classification
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Get last non-padding token representation for each sequence
        # Find the last non-padding token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        # Extract the representation at the last token position
        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]  # [batch_size, hidden_size]
        
        # Ensure classifier is on the same device and dtype as hidden states
        classifier_device = next(self.classifier.parameters()).device
        classifier_dtype = next(self.classifier.parameters()).dtype
        
        if classifier_device != last_hidden_states.device or classifier_dtype != last_hidden_states.dtype:
            self.classifier = self.classifier.to(
                device=last_hidden_states.device,
                dtype=last_hidden_states.dtype
            )
        
        # Pass through classification head
        logits = self.classifier(last_hidden_states)  # [batch_size, num_labels]
        
        output_dict = {
            "logits": logits,
            "hidden_states": last_hidden_states
        }
        
        # Compute loss if labels provided (loss will be computed by trainer)
        if labels is not None:
            output_dict["labels"] = labels
        
        return output_dict
    
    def save_pretrained(self, save_directory: str):
        """Save model and adapters"""
        # Save LoRA adapters
        self.base_model.save_pretrained(save_directory)
        
        # Save classification head
        classifier_path = f"{save_directory}/classifier.pt"
        torch.save(self.classifier.state_dict(), classifier_path)
        
        print(f"Model saved to {save_directory}")
    
    def load_classifier(self, save_directory: str):
        """Load saved classification head"""
        classifier_path = f"{save_directory}/classifier.pt"
        self.classifier.load_state_dict(torch.load(classifier_path))
        print(f"Classifier loaded from {classifier_path}")


def load_model_and_tokenizer(
    model_path: str,
    num_labels: int = 4,
    lora_config: Optional[Dict] = None,
    quantization_config: Optional[Dict] = None
) -> tuple:
    """
    Load model and tokenizer
    
    Args:
        model_path: Path to Meditron-7B model
        num_labels: Number of output labels
        lora_config: LoRA configuration dict
        quantization_config: Quantization configuration dict
        
    Returns:
        (model, tokenizer)
    """
    print("="*60)
    print("LOADING MODEL AND TOKENIZER")
    print("="*60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load model
    print("\n")
    model = MeditronForMultiLabelClassification(
        model_path=model_path,
        num_labels=num_labels,
        lora_config=lora_config,
        quantization_config=quantization_config
    )
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Classifier parameters
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"Classification head parameters: {classifier_params:,}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test model loading
    import sys
    
    model_path = "/home/ubuntu/nahiyan_workspace/s3_bucket/models/meditron-7b"
    
    print("Testing model loading...")
    
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            num_labels=4
        )
        
        print("\n" + "="*60)
        print("Testing forward pass...")
        print("="*60)
        
        # Create dummy input
        dummy_text = "Patient presents with fever and elevated white blood cell count."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Logits: {outputs['logits']}")
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs['logits'])
        print(f"Probabilities: {probs}")
        
        print("\n✅ Model loading and forward pass test PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
