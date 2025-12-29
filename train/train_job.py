import ray
import ray.train.huggingface.transformers
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import mlflow
import os

# --- 1. The Training Logic (Runs inside the Worker Pods) ---
def train_func(config):
    # Imports must be inside the function for distributed workers
    import transformers
    import evaluate
    import numpy as np
    from datasets import load_dataset
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
        # Robust handling if logits is a tuple (common in HF)
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
        report_to="none",  # IMPORTANT: Disable HF internal logging so Ray handles it
        disable_tqdm=True, # Cleaner logs
        no_cuda=True,      # Force CPU training for local labs
        use_cpu=True       # Explicitly request CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # Optimization: Use a small shard of data (1/10th) so it finishes quickly on a laptop
        train_dataset=tokenized_datasets["train"].shard(index=0, num_shards=10), 
        eval_dataset=tokenized_datasets["validation"].shard(index=0, num_shards=10),
        compute_metrics=compute_metrics,
    )

    # E. Ray Integration Callback
    # This magic callback bridges HuggingFace Trainer -> Ray Train -> MLflow
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    
    # F. Prepare & Train
    # prepare_trainer handles moving model to GPU/CPU and wrapping with DDP (Distributed Data Parallel)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    print("Starting training...")
    trainer.train()

# --- 2. The Orchestration Logic (Runs on the Head Node Driver) ---
if __name__ == "__main__":
    # Initialize connection to Ray Cluster
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    # Configure MLflow URI
    # When running inside K8s, we access the service via internal DNS
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.default.svc.cluster.local:5000")
    experiment_name = "sentinel-bert-finetune"
    
    print(f"Tracking URI: {mlflow_tracking_uri}")

    # Define the Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"epochs": 2}, # Run 2 epochs to see a chart
        
        # Scaling Config: This dictates the cluster resources
        # Ensure your Kind cluster has enough CPU!
        # num_workers=2 requires at least 2 CPUs available after Head node overhead
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False), 
        
        run_config=RunConfig(
            name=experiment_name,
            storage_path="/tmp/ray_results", # Local storage on head node (ephemeral)
            callbacks=[
                # Ray's native MLflow integration
                ray.train.callbacks.MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name=experiment_name,
                    save_artifact=True
                )
            ],
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    # Execute
    print(f"Submitting Distributed Training Job...")
    result = trainer.fit()
    
    print(f"Training Complete!")
    print(f"Metrics: {result.metrics}")
    print(f"Checkpoint Path: {result.checkpoint}")