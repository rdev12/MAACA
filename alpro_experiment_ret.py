# %%
from dataclasses import dataclass
import transformers
import os
from typing import Any

from torch.utils.data import Dataset
from pytorchvideo.data.video import VideoPathHandler
import pandas as pd
# from lavis.common.dist_utils import get_rank, init_distributed_mode

transformers.logging.set_verbosity_info()

# %%
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# init_distributed_mode(cfg.run_cfg)

# %%
class DataArguments:
    video_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/downloads"
    audio_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/audio"
    train_csv_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/data_train.csv"
    val_csv_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/data_val.csv"
    intervals_path: str = "/home/rishikesh_2001cs85/Video Complaint Identification/CGDETR/predicted_time_intervals.csv"
    # device = "cuda"
    device = "cpu"

data_args = DataArguments()

# %%
df_intervals = pd.read_csv(data_args.intervals_path, index_col="index")
df_intervals.head()

class VideoEntityDataset(Dataset):
    label2id = {'camera':0, 'os':1, 'design':2, 'battery':3, 'price':4, 'speaker':5, 'storage':6}
    id2label = {0:'camera', 1:'os', 2:'design', 3:'battery', 4:'price', 5:'speaker', 6:'storage'}
    # prompt = "Consider the given transcript from the review video and answering the following question.\nCAPTION: {}\nQUESTION: Choose the entity that best represents the topic of the transcript: \na)camera b)os  c)design d)battery e)price f)speaker g)storage \n\n"

    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        df,
        txt_processor,
        vis_processor,
        audio_processor
    ) -> None:
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.df = df
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
        # import pdb; pdb.set_trace()
        video = self._video_path_handler.video_from_path(video_dir)

        window_start = max(0, df_intervals.iloc[index]["window_0_start"])
        window_end = df_intervals.iloc[index]["window_0_end"]

        clip = video.get_clip(tsmp[0] + window_start, tsmp[0] + window_end)
        audio = self.return_audio_tensor(audio_path)

        # print(window_start, window_end)
        # print(clip["video"].shape)
        
        video = self.vis_processor(clip["video"])
        # video = self.vis_processor(video_dir)
        # print("Video shape", video.shape)

        aspect = VideoEntityDataset.label2id[row["aspect"]]
        complaint = int(row["complaint label"])
        
        text = self.txt_processor(row['transcript'])

        item = {"video": video, "audio": audio, "text_input": text, "aspect":aspect, "complaint":complaint}

        return item

    def __len__(self) -> int:
        return len(self.df)

# %%
from lavis.models import load_preprocess2
vis_processor, txt_processor = load_preprocess2("alpro_retrieval", model_type="msrvtt")

# %%
import pydub
from transformers import WhisperFeatureExtractor
audio_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# %%
train_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.audio_dir,
    pd.read_csv(data_args.train_csv_dir),
    txt_processor["train"],
    vis_processor["train"],
    audio_processor
)

val_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.audio_dir,
    pd.read_csv(data_args.val_csv_dir)[:108],
    txt_processor["eval"],
    vis_processor["eval"],
    audio_processor
)

from lavis.models import load_model
device = "cpu"
model = load_model("alpro_retrieval", model_type="msrvtt", device=device)

print(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Max Text Length: {model.max_txt_len}")
print(f"Model:\n{model}")


# %%
# import numpy as np
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# def compute_metrics(p):
#     labels = p.label_ids
#     pred = p.predictions[1]
#     pred = np.argmax(pred, axis=1)
#     labels = np.expand_dims(labels, axis=1)
#     accuracy = accuracy_score(y_true=labels, y_pred=pred)
#     recall = recall_score(y_true=labels, y_pred=pred, average='micro')
#     precision = precision_score(y_true=labels, y_pred=pred, average='micro')
#     f1 = f1_score(y_true=labels, y_pred=pred, average='micro')    
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# %%
import torch
from torch.utils.data import DataLoader

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

# %%
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=1)

training_args = TrainingArguments(
    output_dir="./output/alpro_retrieval_original_4/",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    save_strategy="steps",
    save_steps=900,
    # do_eval=False,
    eval_steps=150,
    evaluation_strategy="steps",
    # gradient_accumulation_steps=4,
    report_to="none",
    no_cuda= True if data_args.device == "cpu" else False,
    label_names=["aspect"],
    # remove_unused_columns=False
    logging_steps=10,
    disable_tqdm=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end = True
)

# %%
# dataloader = DataCollatorForVideoClassfication(train_data, batch_size=1)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForVideoClassfication(),
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[early_stopping_callback]  
)


# %%
print(f"MODEL DEVICE:{model.device}")
trainer.train()
trainer.save_model(training_args.output_dir)



