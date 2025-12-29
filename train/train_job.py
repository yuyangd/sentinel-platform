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

    # 1. Recover variables
    experiment_name = config.get("mlflow_experiment_name", "sentinel-default")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    
    # 2. Load Data
    print("Loading dataset...")
    dataset = load_dataset("glue", "mrpc")
    
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Prepare Model
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
        eval_strategy="epoch",
        save_strategy="no",  # Do not save the checkpoint
        load_best_model_at_end=False,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=config.get("epochs", 1), 
        weight_decay=0.01,
        push_to_hub=False,
        report_to="mlflow", 
        disable_tqdm=True,
        dataloader_num_workers=0, # Keep at 0 for memory safety
        use_cpu=True       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"].shard(index=0, num_shards=10), 
        eval_dataset=tokenized_datasets["validation"].shard(index=0, num_shards=10),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, 
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
    if ray.is_initialized():
        ray.shutdown()
    ray.init() 

    experiment_name = "sentinel-bert-tiny"

    # Define the Ray Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "epochs": 2,
            "mlflow_experiment_name": experiment_name
        }, 
        
        scaling_config=ScalingConfig(
            num_workers=1, 
            use_gpu=False,
            resources_per_worker={"CPU": 1}
        ), 
        
        run_config=RunConfig(
            name=experiment_name,
            storage_path="/tmp/ray_results", 
            checkpoint_config=CheckpointConfig(
                num_to_keep=0,
                checkpoint_frequency=0
            ),
        ),
    )

    print(f"Submitting Distributed Training Job: {experiment_name}")
    result = trainer.fit()
    
    print(f"Training Complete!")
    print(f"Metrics: {result.metrics}")
