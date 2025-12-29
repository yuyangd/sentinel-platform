import ray
import ray.train.huggingface.transformers
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import os

# --- 1. The Training Logic (Runs inside the Worker Pods) ---
def train_func(config):
    # Imports must be inside the function for distributed workers
    import numpy as np
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )

    # A. Load Data
    print("Loading dataset...")
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # B. Prepare Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # C. Define Metrics (Accuracy/F1)
    metric = evaluate.load("glue", "mrpc")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # D. Hugging Face Training Config
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=config.get("epochs", 1), 
        weight_decay=0.01,
        push_to_hub=False,
        # --- FIX 1: Use Native HF MLflow integration ---
        report_to="mlflow",  
        disable_tqdm=True, 
        no_cuda=True,      
        use_cpu=True       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # Optimization: Use a small shard of data (1/10th)
        train_dataset=tokenized_datasets["train"].shard(index=0, num_shards=10), 
        eval_dataset=tokenized_datasets["validation"].shard(index=0, num_shards=10),
        compute_metrics=compute_metrics,
    )

    # E. Ray Integration Callback
    # This keeps Ray informed of progress (for autoscaling/dashboard), 
    # but MLflow logging is now handled by 'report_to="mlflow"' above.
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    
    # F. Prepare & Train
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    print("Starting training...")
    trainer.train()

# --- 2. The Orchestration Logic (Runs on the Head Node Driver) ---
if __name__ == "__main__":
    # Define MLflow Settings
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.default.svc.cluster.local:5000")
    experiment_name = "sentinel-bert-finetune"

    # --- FIX 2: Pass Env Vars to Workers via runtime_env ---
    # This ensures every worker knows WHERE to log, without needing a special callback.
    runtime_env = {
        "env_vars": {
            "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": experiment_name,
        }
    }

    # Initialize connection to Ray Cluster with the env vars
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env=runtime_env)

    print(f"Tracking URI: {mlflow_tracking_uri}")

    # Define the Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"epochs": 2}, 
        
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False), 
        
        run_config=RunConfig(
            name=experiment_name,
            storage_path="/tmp/ray_results", 
            # --- FIX 3: Removed the incompatible MLflowLoggerCallback ---
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    # Execute
    print(f"Submitting Distributed Training Job...")
    result = trainer.fit()
    
    print(f"Training Complete!")
    print(f"Metrics: {result.metrics}")
    print(f"Checkpoint Path: {result.checkpoint}")