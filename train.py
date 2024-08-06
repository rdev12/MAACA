import transformers
import os
import pydub
import torch
import wandb
import numpy as np
import pandas as pd
from typing import Any
from omegaconf import OmegaConf
from dataclasses import dataclass
from torch.utils.data import Dataset
from pytorchvideo.data.video import VideoPathHandler
from transformers import WhisperFeatureExtractor, Trainer, HfArgumentParser, TrainingArguments, EarlyStoppingCallback
from src.model.alpro_qa_audio2 import MAACA
from src.processors.alpro_processors import AlproVideoEvalProcessor, AlproVideoTrainProcessor
from src.processors.blip_processors import BlipCaptionProcessor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

transformers.logging.set_verbosity_info()

def load_model_from_config(config_path, cmd_config, is_eval=False, device="cpu", checkpoint=None):

    model_cfg = OmegaConf.load(config_path).model
    model = MAACA.from_config(model_cfg, cmd_config)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)

def load_preprocess2(config_path):

    config = OmegaConf.load(config_path).preprocess

    vis_processors = dict()
    txt_processors = dict()

    vis_proc_cfg = config.get("vis_processor")
    txt_proc_cfg = config.get("text_processor")

    if vis_proc_cfg is not None:
        vis_train_cfg = vis_proc_cfg.get("train")
        vis_eval_cfg = vis_proc_cfg.get("eval")
    else:
        vis_train_cfg = None
        vis_eval_cfg = None

    vis_processors["train"] = AlproVideoTrainProcessor.from_config(vis_train_cfg)
    vis_processors["eval"] = AlproVideoEvalProcessor.from_config(vis_eval_cfg)

    if txt_proc_cfg is not None:
        txt_train_cfg = txt_proc_cfg.get("train")
        txt_eval_cfg = txt_proc_cfg.get("eval")
    else:
        txt_train_cfg = None
        txt_eval_cfg = None

    txt_processors["train"] = BlipCaptionProcessor.from_config(txt_train_cfg)
    txt_processors["eval"] = BlipCaptionProcessor.from_config(txt_eval_cfg)

    return vis_processors, txt_processors

@dataclass
class DataArguments:
    video_dir: str = "data/video/"
    audio_dir: str = "data/audio/"
    train_csv_dir: str = "data/data_train_sample.csv"
    val_csv_dir: str = "data/data_val_sample.csv"
    intervals_path: str = "data/predicted_time_intervals.csv"
    device: str = "cuda:0"

@dataclass
class ModelArguments:
    run_name: str = "maaca"
    config_path: str = "src/configs/config.yaml"
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

parser = HfArgumentParser((ModelArguments, DataArguments))
model_args, data_args = parser.parse_args_into_dataclasses()

training_args = TrainingArguments(
    output_dir = os.path.join("src/output", model_args.run_name),
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 15,
    save_strategy = "steps",
    save_steps = 1000,
    eval_steps = 100,
    evaluation_strategy = "steps",
    report_to = "none",
    logging_steps = 10,
    disable_tqdm = False,
    metric_for_best_model ="eval_loss",
    load_best_model_at_end = True,
    no_cuda = True if data_args.device == "cpu" else False,
    # label_names = ["aspect"] if model_args.num_classes == 7 else ["complaint"]
    label_names = ["complaint", "aspect"],
    save_total_limit = 1
)

df_intervals = pd.read_csv(data_args.intervals_path, index_col="index")

class VideoEntityDataset(Dataset):
    label2id = {'camera':0, 'os':1, 'design':2, 'battery':3, 'price':4, 'speaker':5, 'storage':6}
    id2label = {0:'camera', 1:'os', 2:'design', 3:'battery', 4:'price', 5:'speaker', 6:'storage'}

    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        csv_file: str,
        txt_processor,
        vis_processor,
        audio_processor
    ) -> None:
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.df = pd.read_csv(csv_file)
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self._video_path_handler = VideoPathHandler()

    def get_time_range_in_sec(self, time):
        time = time.replace(" ", "")
        start, end = time.split("-")
        start = int(start.split(":")[0])*60 + int(start.split(":")[1])
        end = int(end.split(":")[0])*60 + int(end.split(":")[1])

        return [start, end]

    def return_audio_tensor(self, file_path):
        audio_segment = pydub.AudioSegment.from_mp3(file_path)

        input_audio = audio_segment.get_array_of_samples()
        input_audio = audio_processor(input_audio, sampling_rate=16000, return_tensors="pt")
        input_features = input_audio["input_features"]

        return input_features.squeeze(dim=0).cpu()
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        tsmp = self.get_time_range_in_sec(row["time stamp"])
        video_dir = os.path.join(self.video_dir, "tmp_{}.mp4".format(row["video index"]))
        audio_path = os.path.join(self.audio_dir, "audio_{}.mp3".format(row["index"]))

        video = self._video_path_handler.video_from_path(video_dir)

        window_start = max(0, df_intervals.iloc[index]["window_0_start"])
        window_end = df_intervals.iloc[index]["window_0_end"]

        clip = video.get_clip(tsmp[0] + window_start, tsmp[0] + window_end)
        audio = self.return_audio_tensor(audio_path)

        video = self.vis_processor(clip["video"])

        aspect = VideoEntityDataset.label2id[row["aspect"]]
        complaint = int(row["complaint label"])
        
        text = self.txt_processor(row['transcript'])

        item = {"video": video, "audio": audio, "text_input": text, "aspect":aspect, "complaint":complaint}

        return item

    def __len__(self) -> int:
        return len(self.df)

vis_processor, txt_processor = load_preprocess2("src/configs/config.yaml")
audio_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

train_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.audio_dir,
    data_args.train_csv_dir,
    txt_processor["train"],
    vis_processor["train"],
    audio_processor
)

val_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.audio_dir,
    data_args.val_csv_dir,
    txt_processor["eval"],
    vis_processor["eval"],
    audio_processor
)

model = load_model_from_config(config_file=model_args.config_path, cmd_config=model_args, device=data_args.device)

print(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Max Text Length: {model.max_txt_len}")
print(f"Model:\n{model}")

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

class DataCollatorForVideoClassfication():
    def __call__(self, features, return_tensors="pt"):
        aspect = torch.tensor(
            [feature.pop("aspect") for feature in features]
        )
        complaint = torch.tensor(
            [feature.pop("complaint") for feature in features]
        )
        video = torch.stack(
            [feature.pop("video") for feature in features]
        )
        audio = torch.stack(
            [feature.pop("audio") for feature in features]
        )
        text_input = [feature.pop("text_input") for feature in features]

        return {"video": video, "audio": audio, "text_input": text_input, "aspect":aspect, "complaint":complaint}

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

if (training_args.report_to == "wandb"):
    wandb.init(name=model_args.run_name)
print(model_args)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForVideoClassfication(),
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,   
    callbacks=[early_stopping_callback]  
)

print(f"MODEL DEVICE:{model.device}")
trainer.train()
metrics = trainer.evaluate()
trainer.save_metrics("eval", metrics)
trainer.save_model(training_args.output_dir)