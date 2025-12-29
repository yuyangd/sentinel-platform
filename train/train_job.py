import ray
import ray.train.huggingface.transformers
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import os

# --- 1. The Training Logic ---
def train_func(config):
    import numpy as np
    from datasets import load_dataset
    import evaluate
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )

    # 1. Recover variables from config
    # Since we can't set env vars in ray.init, we pass them via config
    experiment_name = config.get("mlflow_experiment_name", "sentinel-default")
    
    # Set the env var locally for this worker process so HF picks it up
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    
    # Note: MLFLOW_TRACKING_URI is already set by your RayJob YAML, 
    # so we don't need to touch it here.

    # 2. Load Data
    print("Loading dataset...")
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Prepare Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # 4. Metrics
    metric = evaluate.load("glue", "mrpc")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 5. HF Config
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
        report_to="mlflow",  # Native HF integration
        disable_tqdm=True, 
        no_cuda=True,      
        use_cpu=True       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shard(index=0, num_shards=10), 
        eval_dataset=tokenized_datasets["validation"].shard(index=0, num_shards=10),
        compute_metrics=compute_metrics,
    )

    # 6. Ray Callback
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    
    # 7. Train
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)
    print("Starting training...")
    trainer.train()

# --- 2. The Orchestration Logic ---
if __name__ == "__main__":
    # --- FIX: Initialize without runtime_env ---
    # We assume the RayJob YAML has already set up the environment (pip packages, basic env vars).
    if ray.is_initialized():
        ray.shutdown()
    ray.init() 

    # Dynamic variables we want to control from code
    experiment_name = "sentinel-bert-finetune"

    # Define the Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        # Pass the experiment name dynamically to workers
        train_loop_config={
            "epochs": 2,
            "mlflow_experiment_name": experiment_name
        }, 
        
        scaling_config=ScalingConfig(
            num_workers=2, 
            use_gpu=False,
            # Tell Ray each worker only "costs" 0.5 CPU.
            # This allows 2 workers to run even if you only have 1 CPU free.
            resources_per_worker={"CPU": 0.5}
        ), 
        
        run_config=RunConfig(
            name=experiment_name,
            storage_path="/tmp/ray_results", 
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    print(f"Submitting Distributed Training Job: {experiment_name}")
    result = trainer.fit()
    
    print(f"Training Complete!")
    print(f"Metrics: {result.metrics}")