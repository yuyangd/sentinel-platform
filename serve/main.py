import ray
from ray import serve
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

@serve.deployment(
    num_replicas=1, 
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0}
)
@serve.ingress(app)
class SentinelModel:
    def __init__(self):
        print("Initializing Sentinel Model (TinyBERT)...")
        # 1. Load the Model & Tokenizer
        # In the future, this path will be your S3 bucket or local checkpoint path
        model_name = "prajjwal1/bert-tiny" 
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Optimize for inference (turn off gradient calculation)
        self.model.eval() 
        print("Model Initialized!")

    @app.post("/predict")
    def predict(self, data: dict):
        # 2. Parse Input
        text = data.get("text", "")
        if not text:
            return {"error": "No text provided"}

        # 3. Preprocess (Tokenize)
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )

        # 4. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities (Softmax)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the predicted class (0 or 1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class_id].item()

        # 5. Map to labels (MRPC dataset is "Not Equivalent" vs "Equivalent")
        labels = ["Not Equivalent", "Equivalent"]
        prediction_label = labels[predicted_class_id]

        return {
            "model": "prajjwal1/bert-tiny",
            "text": text,
            "prediction": prediction_label,
            "confidence": round(confidence, 4)
        }

# Bind the deployment to the app
deployment = SentinelModel.bind()
