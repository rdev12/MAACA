# %% [markdown]
# # Alpro Training

# %%
from dataclasses import dataclass
import transformers

transformers.logging.set_verbosity_info()
import torch.distributed as dist

# Initialize the distributed environment
dist.init_process_group(backend='nccl') 

# %%
@dataclass
class DataArguments:
    video_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/downloads"
    train_csv_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/data_train.csv"
    val_csv_dir: str = "/home/rishikesh_2001cs85/Video Complaint Identification/multimodal-complaint-aspect/data_val.csv"
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 128
    num_frames = 1
    sampling_rate = 2
    frames_per_second = 30
    alpha = 4

# %%
data_args = DataArguments()

# %%
import os
from typing import Any

from torch.utils.data import Dataset
from pytorchvideo.data.video import VideoPathHandler
import pandas as pd


class VideoEntityDataset(Dataset):
    label2id = {'camera':0, 'os':1, 'design':2, 'battery':3, 'price':4, 'speaker':5, 'storage':6}
    id2label = {0:'camera', 1:'os', 2:'design', 3:'battery', 4:'price', 5:'speaker', 6:'storage'}
    # prompt = "Consider the given transcript from the review video and answering the following question.\nCAPTION: {}\nQUESTION: Choose the entity that best represents the topic of the transcript: \na)camera b)os  c)design d)battery e)price f)speaker g)storage \n\n"

    def __init__(
        self,
        video_dir: str,
        csv_file: str,
        txt_processor,
        vis_processor
    ) -> None:
        """
        :param narrated_actions_dir: path to dir that contains narrated_actions.csv
            and extracted frames
        """
        self.video_dir = video_dir
        self.df = pd.read_csv(csv_file)
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor
        self._video_path_handler = VideoPathHandler()


    def get_time_range_in_sec(self, time):
        # get rid of unintentional spaces
        time = time.replace(" ", "")

        # get starting and ending times
        start, end = time.split("-")

        # convert to seconds
        start = int(start.split(":")[0])*60 + int(start.split(":")[1])
        end = int(end.split(":")[0])*60 + int(end.split(":")[1])

        return [start, end]

    
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        tsmp = self.get_time_range_in_sec(row["time stamp"])
            
        video_dir = os.path.join(self.video_dir, "tmp_{}.mp4".format(row["video index"]))

        video = self._video_path_handler.video_from_path(video_dir)
        clip = video.get_clip(tsmp[0], tsmp[1])
        video = self.vis_processor(clip["video"])
        # video = self.vis_processor(video_dir)
        # print("Video shape", video.shape)

        # label = VideoEntityDataset.label2id[row["aspect"]]
        label = row["aspect"]
        text = self.txt_processor(row['transcript'])

        # item = {"video": video, "text_input": text, "answers":label, "complaint":int(row["complaint label"])}
        item = {"video": video, "text_input":label}

        return item

    def __len__(self) -> int:
        return len(self.df)


# %%


# %%
# setup device to use
import torch
from lavis.models import load_model_and_preprocess, model_zoo
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model, vis_processor, txt_processor = load_model_and_preprocess("alpro_retrieval", model_type="msrvtt", device="cuda")

# %%
# from transformers import AutoModel
# model = AutoModel.from_pretrained("/home/rishikesh_2001cs85/Video Complaint Identification/LAVIS/output/checkpoint-648")

# %%
# from lavis.models.alpro_models.alpro_multi_task import AlproMultiTask

# %%
# model = AlproMultiTask(model.visual_encoder, model.text_encoder, 768, 7, 2) 

# %%


# %%
train_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.train_csv_dir,
    txt_processor["train"],
    vis_processor["train"]
)

val_data = VideoEntityDataset(
    data_args.video_dir,
    data_args.val_csv_dir,
    txt_processor["eval"],
    vis_processor["eval"]
)

# %%
# samples = train_data[0]

# %%
# samples["video"].shape

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

class DataCollatorForVideoClassfication():
    def __call__(self, features, return_tensors="pt"):
        # answers = torch.tensor(
        #     [feature.pop("answers") for feature in features]
        # )
        # complaint_label = torch.tensor(
        #     [feature.pop("complaint") for feature in features]
        # )

        video = torch.stack(
            [feature.pop("video") for feature in features]
        ) # b, c, t, h, w
        
        text_input = [feature.pop("text_input") for feature in features]
        samples = dict()
        samples["samples"] = {"video": video, "text_input": text_input}
        return samples

# %%
# model = model.to("cuda:2")

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output_alpro_retrieval/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    save_strategy="epoch",
    # eval_steps=50,
    evaluation_strategy="no",
    # gradient_accumulation_steps=4,
    report_to="none",
    # no_cuda=True,
    label_names=["video", "text_input"],
    # remove_unused_columns=False
    logging_steps=1,
    do_eval=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForVideoClassfication(),
    train_dataset=train_data,
    eval_dataset=val_data,
    # callbacks=[EarlyStoppingCallback()],
    # compute_metrics=compute_metrics,
)

# %%
# sum(p.numel() for p in model.parameters())

# %%
print(f"MODEL DEVICE:{model.device}")
trainer.train()
# print(trainer.evaluate())
# print(trainer.predict(val_data))

trainer.save_model()

# %%
# trainer.train()

# %%
print(trainer.evaluate())

# %%



