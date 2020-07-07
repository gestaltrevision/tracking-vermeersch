import argparse
import os,json
import pandas as pd
from moviepy.editor import *

from tqdm import tqdm
parser = argparse.ArgumentParser()

parser.add_argument("--root_path_data", type=str,
                    help="File with config settings for testing")

parser.add_argument("--root_path_video", type=str,
                    help="File with config settings for testing")


parser.add_argument("--video_folder_root", type=str,
                    help="File with config settings for testing")

def create_video_dataset():
    video_dataset = {
            "video_path": [],
            "label": [],
            "frames": []
            }
    return video_dataset

if __name__ == "__main__":
    args = parser.parse_args()

    root_path_data = args.root_path_data
    root_path_video = args.root_path_video
    participants = os.listdir(root_path)

    for participant in participants:
        #set-up
        video_folder = os.path.join(root_path_data,participant,args.video_folder_root)
        path_video = os.path.join(root_path_video,f"{participant}.mp4")
        video = VideoFileClip(path_video).resize((224,224))
        dataset_path = os.path.join(root_path_data,participant,"FullDataset")
        table_file  = os.path.join(dataset_path,f"video_df_{participant}.csv")
        df = pd.read_csv(table_file)
        if not(os.path.isdir(video_folder)):
            os.makedirs(video_folder)
        
        video_dataset = create_video_dataset()
        for idx in tqdm(df.index):
            #get clip
            clip = video.subclip(df.loc[idx,"start"],df.loc[idx,"end"])
            n_frames  = int(clip.duration * clip.fps)
            label  = df.loc[idx,"target"]
            # Write the result to a file 
            clip_id = df.loc[idx,"id"]
            video_rel_path = os.path.join(participant,"Clips",f"{clip_id}.mp4")
            clip_file= os.path.join(root_path_data,video_rel_path)
            clip.write_videofile(clip_file,audio=False,logger = None)
            #update dataset
            video_dataset["video_path"].append(video_rel_path)
            video_dataset["label"].append(label)
            video_dataset["frames"].append(n_frames)

        #save dataset
        video_dataset_file = os.path.join(dataset_path,f"video_dataset_{participant}.txt")
        with open(video_dataset_file,"w") as f:
            json.dump(video_dataset,f)

        print(f"Participant {participant} correctly processed")