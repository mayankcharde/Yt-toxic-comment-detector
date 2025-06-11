from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_toxicity(comment):
    try:
        # Tokenize and get prediction
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = F.sigmoid(outputs.logits).squeeze().tolist()
        
        # Handle both single value and list outputs
        toxicity_score = prob[0] if isinstance(prob, list) else prob
        
        # Determine toxicity level
        if toxicity_score > 0.7:
            level = "Severely Toxic"
        elif toxicity_score > 0.5:
            level = "Toxic"
        else:
            level = "Non-Toxic"
        
        return {
            "weighted_score": float(toxicity_score),
            "level": level
        }
    except Exception as e:
        print(f"Error processing comment: {e}")
        return {
            "weighted_score": 0.0,
            "level": "Error"
        }
