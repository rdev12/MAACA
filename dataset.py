import pydub
import os
import torch
import pandas as pd
from typing import Any
from torch.utils.data import Dataset
from pytorchvideo.data.video import VideoPathHandler

class VideoEntityDataset(Dataset):
    label2id = {'camera': 0, 'os': 1, 'design': 2, 'battery': 3, 'price': 4, 'speaker': 5, 'storage': 6}
    id2label = {0: 'camera', 1: 'os', 2: 'design', 3: 'battery', 4: 'price', 5: 'speaker', 6: 'storage'}

    def __init__(
            self,
            video_dir: str,
            audio_dir: str,
            data_file_path: str,
            intervals_file_path: str,
            txt_processor,
            vis_processor,
            audio_processor
    ) -> None:
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.df = pd.read_csv(data_file_path)
        self.df_intervals = pd.read_csv(intervals_file_path, index_col="index")
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor
        self.audio_processor = audio_processor
        self._video_path_handler = VideoPathHandler()

    def get_time_range_in_sec(self, time):
        time = time.replace(" ", "")
        start, end = time.split("-")
        start = int(start.split(":")[0]) * 60 + int(start.split(":")[1])
        end = int(end.split(":")[0]) * 60 + int(end.split(":")[1])

        return [start, end]

    def return_audio_tensor(self, file_path):
        audio_segment = pydub.AudioSegment.from_mp3(file_path)

        input_audio = audio_segment.get_array_of_samples()
        input_audio = self.audio_processor(input_audio, sampling_rate=16000, return_tensors="pt")
        input_features = input_audio["input_features"]

        return input_features.squeeze(dim=0).cpu()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        tsmp = self.get_time_range_in_sec(row["time stamp"])
        video_dir = os.path.join(self.video_dir, "tmp_{}.mp4".format(row["video index"]))
        audio_path = os.path.join(self.audio_dir, "audio_{}.mp3".format(row["index"]))

        video = self._video_path_handler.video_from_path(video_dir)

        window_start = max(0, self.df_intervals.iloc[index]["window_0_start"])
        window_end = self.df_intervals.iloc[index]["window_0_end"]

        clip = video.get_clip(tsmp[0] + window_start, tsmp[0] + window_end)
        audio = self.return_audio_tensor(audio_path)

        video = self.vis_processor(clip["video"])

        aspect = VideoEntityDataset.label2id[row["aspect"]]
        complaint = int(row["complaint label"])

        text = self.txt_processor(row['transcript'])

        item = {"video": video, "audio": audio, "text_input": text, "aspect": aspect, "complaint": complaint}

        return item

    def __len__(self) -> int:
        return len(self.df)


class DataCollatorForVideoClassfication:
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