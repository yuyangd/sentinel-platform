import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Define the input structure explicitly
class FactCheckRequest(BaseModel):
    query: str
    facts: dict

@serve.deployment(
    num_replicas=1, 
    ray_actor_options={"num_cpus": 0.5}
)
@serve.ingress(app)
class FactCheckGuardrail:
    def __init__(self):
        print("Initializing Fact-Check Guardrail...")
        # We use a model specifically trained for NLI (Truth checking)
        self.model_name = "cross-encoder/nli-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

    def linearize_facts(self, facts: dict) -> str:
            """Converts JSON facts into natural language statements."""
            statements = []
            for key, value in facts.items():
                # Handle Booleans explicitly
                if isinstance(value, bool):
                    if value:
                        statements.append(f"The property has a {key}.")
                    else:
                        statements.append(f"The property does not have a {key}.")
                # Handle Numbers/Strings
                else:
                    statements.append(f"The number of {key} is {value}.")
            
            return " ".join(statements)

    @app.post("/check")
    def check_facts(self, request: FactCheckRequest):
        # 1. Prepare the "Truth" (Premise)
        premise = self.linearize_facts(request.facts)
        
        # 2. Prepare the "Claim" (Hypothesis)
        hypothesis = request.query

        # 3. Tokenize as a pair
        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )

        # 4. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The model outputs 3 scores: [Contradiction, Entailment, Neutral]
            # (Note: specific models vary on index order, standard for this one is usually C/E/N)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # For this specific model: Label 0 = Contradiction, Label 1 = Entailment
            contradiction_score = probs[0][0].item()
            entailment_score = probs[0][1].item()

        # 5. Business Logic (The Guardrail)
        # If contradiction is higher than entailment, we BLOCK.
        is_safe = entailment_score > contradiction_score
        
        return {
            "status": "PASS" if is_safe else "FAIL",
            "reason": "Fact Contradiction" if not is_safe else "Verified",
            "scores": {
                "contradiction": round(contradiction_score, 4),
                "entailment": round(entailment_score, 4)
            },
            "inputs_debug": f"Comparing '{premise}' vs '{hypothesis}'"
        }

deployment = FactCheckGuardrail.bind()
