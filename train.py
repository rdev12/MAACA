import os
import wandb
import transformers
import numpy as np
from dataclasses import dataclass
from transformers import Trainer, HfArgumentParser, TrainingArguments, EarlyStoppingCallback
from dataset import VideoEntityDataset, DataCollatorForVideoClassfication
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.utils.load import load_preprocess, load_model_from_config

transformers.logging.set_verbosity_info()

@dataclass
class DataArguments:
    video_dir: str = "data/video/"
    audio_dir: str = "data/audio/"
    train_csv_dir: str = "data/data_train.csv"
    val_csv_dir: str = "data/data_val.csv"
    intervals_path: str = "data/predicted_time_intervals.csv"
    device: str = "cuda:0"

@dataclass
class ModelArguments:
    config_path: str = "src/configs/config.yaml"
    pretraining: bool = False
    audio_transform_hidden_dim: int = 768
    video_transform_hidden_dim: int = 768
    audio_transform_num_layers: int = 2
    video_transform_num_layers: int = 2
    audio_output_seq_len: int = 128
    video_output_seq_len: int = 128
    audio_transform_output_dim: int = 768
    video_transform_output_dim: int = 768
    fusion_output_dim: int = 768
    linear_layer_hidden_dim: int = 64
    add_pooling: bool = False
    max_txt_len: int = 128

@dataclass
class TrainingArgumentsExtra(TrainingArguments):
    run_name: str = "maaca"
    output_dir: str = "output/"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    num_train_epochs: int = 15
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_steps: int = 100
    evaluation_strategy: str = "steps"
    report_to: str = "none"
    logging_steps: int = 10
    disable_tqdm: bool = False
    metric_for_best_model: str = "eval_loss"
    load_best_model_at_end: bool = True
    save_total_limit: int = 1
    early_stopping_patience: int = 2

def compute_metrics(p):
    label_ids = p.label_ids
    predictions = p.predictions[1]
    stats = {}
    for pred, labels in zip(predictions, label_ids):
        entity = "complaint" if pred.shape[1] == 2 else "aspect"
        pred = np.argmax(pred, axis=1)
        labels = np.expand_dims(labels, axis=1)
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred, average='micro')
        precision = precision_score(y_true=labels, y_pred=pred, average='micro')
        f1 = f1_score(y_true=labels, y_pred=pred, average='micro')

        stat = {f"{entity}_accuracy": accuracy,
                f"{entity}_precision": precision,
                f"{entity}_recall": recall,
                f"{entity}_f1": f1}
        stats.update(stat)

    return stats

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsExtra))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.label_names = ["complaint", "aspect"]
    training_args.output_dir = os.path.join("output", training_args.run_name)
    training_args.no_cuda = True if data_args.device == "cpu" else False

    vis_processor, txt_processor, audio_processor = load_preprocess(model_args.config_path)

    train_data = VideoEntityDataset(
        data_args.video_dir,
        data_args.audio_dir,
        data_args.train_csv_dir,
        data_args.intervals_path,
        txt_processor["train"],
        vis_processor["train"],
        audio_processor
    )

    val_data = VideoEntityDataset(
        data_args.video_dir,
        data_args.audio_dir,
        data_args.val_csv_dir,
        data_args.intervals_path,
        txt_processor["eval"],
        vis_processor["eval"],
        audio_processor
    )

    model = load_model_from_config(config_file=model_args.config_path, cmd_config=model_args, device=data_args.device)

    print(f"MODEL:\n{model}")
    print(f"MODEL DEVICE: {model.device}")
    print(f"MODEL ARGS: {model_args}")

    if training_args.report_to == "wandb":
        wandb.init(name=model_args.run_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForVideoClassfication(),
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    trainer.save_model(training_args.output_dir)
