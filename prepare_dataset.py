import os
import subprocess
import pandas as pd
from pytube import YouTube
from pytube.cli import on_progress


def get_time_range_in_sec(time):
    # get rid of unintentional spaces
    time = time.replace(" ", "")
    # get starting and ending times
    start, end = time.split("-")
    # convert to seconds
    start = int(start.split(":")[0]) * 60 + int(start.split(":")[1])
    end = int(end.split(":")[0]) * 60 + int(end.split(":")[1])

    return [start, end]

def download_video(row, path="data/video/"):
    link = row["video link"]
    index = row["video index"]
    yt = YouTube(link, on_progress_callback=on_progress)

    if not os.path.exists(os.path.join(path, f"tmp_{index}.mp4")):
        try:
            video = yt.streams.filter(res="360p")[0]
            path = video.download(filename=os.path.join(path, f"tmp_{index}.mp4"))
        except:
            print(f"Error occurred while downloading video {index}. Please try for this video again later. Skipping...")
    else:
        print(f"Video {index} exists. Skipping download...")


def download_audio(row, video_path="data/video/", audio_path="data/audio/"):
    index = row.name
    video_index = row["video index"]
    mp4_path = os.path.join(video_path, f"tmp_{video_index}.mp4")
    timestamp = get_time_range_in_sec(row["time stamp"])

    # Remove the extension and add ".mp3" to get the desired output filename
    audio_filename = f"audio_{index}.mp3"

    # Build the full output path
    output_path = os.path.join(audio_path, audio_filename)

    command = [
        "ffmpeg",
        "-i", mp4_path,
        "-ss", str(timestamp[0]),
        "-to", str(timestamp[1]),
        "-vn", "-acodec", "libmp3lame",
        output_path
    ]
    # Run the command using subprocess
    subprocess.run(command)


if __name__ == "__main__":

    # df is concatenation of data/data_train.csv and data/data_val.csv
    df = pd.concat([pd.read_csv("data/data_train.csv"), pd.read_csv("data/data_val.csv")])

    # map each row and index to download_video function
    df.apply(lambda row: download_video(row), axis=1)
    df.apply(lambda row: download_audio(row), axis=1)


