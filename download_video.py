import os
import pandas as pd
import subprocess
from pytube import YouTube
from pytube.cli import on_progress
import argparse

def get_time_range_in_sec(time):
    # get rid of unintentional spaces
    time = time.replace(" ", "")

    # get starting and ending times
    start, end = time.split("-")

    # convert to seconds
    start = int(start.split(":")[0]) * 60 + int(start.split(":")[1])
    end = int(end.split(":")[0]) * 60 + int(end.split(":")[1])

    return [start, end]

def download_and_extract_audio(data_train, data_val, video_dir, audio_dir):
    df = pd.concat([pd.read_csv(data_train, index_col="index"),
                    pd.read_csv(data_val, index_col="index")])

    for index, row in df.iterrows():
        link = row["video link"]
        file_path = os.path.join(video_dir, f"{index}.mp4")
        yt = YouTube(link, on_progress_callback=on_progress)

        if not os.path.exists(file_path):
            try:
                video = yt.streams.filter(res="360p")[0]
                video.download(filename=file_path)
            except Exception as e:
                print(f"Error occurred while downloading video {index}: {e}. Skipping...")
                continue
        else:
            print(f"Video {index} exists. Skipping download...")

        timestamp = get_time_range_in_sec(row["time stamp"])

        # Remove the extension and add ".mp3" to get the desired output filename
        audio_filename = f"audio_{index}.mp3"

        # Build the full output path
        output_path = os.path.join(audio_dir, audio_filename)

        command = [
            "ffmpeg",
            "-i", file_path,
            "-ss", str(timestamp[0]),
            "-to", str(timestamp[1]),
            "-vn", "-acodec", "libmp3lame",
            output_path
        ]
        # Run the command using subprocess
        subprocess.run(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos and extract audio from a combined CSV file.")
    parser.add_argument("--data_train", default="data/data_train.csv", help="Path to the data_train.csv file")
    parser.add_argument("--data_val", default="data/data_val.csv", help="Path to the data_val.csv file")
    parser.add_argument("--video_dir", default="data/video", help="Directory to save downloaded videos")
    parser.add_argument("--audio_dir", default="data/audio", help="Directory to save extracted audio files")

    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.audio_dir, exist_ok=True)

    download_and_extract_audio(args.data_train, args.data_val, args.video_dir, args.audio_dir)