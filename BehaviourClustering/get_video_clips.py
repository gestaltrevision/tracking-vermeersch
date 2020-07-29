import argparse
import os,json
import pandas as pd
from moviepy.editor import *
from pathlib import PurePosixPath

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--root_path_data", type=str,
                    help="Root path for dataset")

parser.add_argument("--root_path_video", type=str,
                    help="Path containing videos")

parser.add_argument("--metadata_path", type=str,
                    help="Path containing mapping from participants to video filenames")

parser.add_argument("--video_folder_root", type=str,
                    help="Type of clips (eg. ArtworkClips)")

def create_video_dataset():
    video_dataset = {
            "video_path": [],
            "label": [],
            "frames": []
            }
    return video_dataset

def get_file_stem(path):
    base=os.path.basename(path)
    return os.path.splitext(base)[0]

def read_metadata(df_path):
  #read df
  df = pd.read_csv(df_path,sep=" ",header= None)
  df.columns = ["video_path","frames","label"]
  return df

if __name__ == "__main__":
    args = parser.parse_args()

    root_path_data = args.root_path_data
    root_path_video = args.root_path_video
    metadata_path  = args.metadata_path
    processed_participants = [participant for participant in os.listdir(root_path_data) 
                                if os.path.isdir(os.path.join(root_path_data,participant,args.video_folder_root))]
    #read metadata
    with open(metadata_path,"r") as f:
        metadata = json.load(f)
    
    test_txt = r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\FinalDatasets\Datasets\Frames_Dataset_v1\Test.txt"
    val_txt = r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\FinalDatasets\Datasets\Frames_Dataset_v1\Val.txt"
    text_df = read_metadata(test_txt)
    val_df = read_metadata(val_txt)
    text_df["participant"]  = text_df["video_path"].apply(lambda participant: participant.split("_")[0])
    val_df["participant"]  = val_df["video_path"].apply(lambda participant: participant.split("_")[0])

    participant_train = list(set(text_df.participant.values))
    participant_val = list(set(val_df.participant.values))
    participant_testing = participant_train + participant_val

    for idx in metadata.keys():

        meta  = list(metadata[idx].values())
        participant, filename = meta
        
        if (participant in participant_testing):
            #set-up
            video_folder = os.path.join(root_path_data,participant,args.video_folder_root)
            path_video = os.path.join(root_path_video,filename)
        
            video = VideoFileClip(path_video).resize((224,224))

            dataset_path = os.path.join(root_path_data,participant,"FullDataset")
            table_file  = os.path.join(dataset_path,f"video_df_{participant}.csv")
            try:
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
                    video_rel_path = PurePosixPath(participant,args.video_folder_root,f"{clip_id}.mp4")
                    clip_file= os.path.join(root_path_data,video_rel_path)
                    clip.write_videofile(clip_file,audio=False,logger = None)

                    #update dataset
                    video_dataset["video_path"].append(str(video_rel_path))
                    video_dataset["label"].append(label)
                    video_dataset["frames"].append(n_frames)

                #save dataset
                video_dataset_file = os.path.join(dataset_path,f"{args.video_folder_root}_dataset_{participant}.txt")
                #old version
                os.remove(video_dataset_file)

                with open(video_dataset_file,"w") as f:
                    json.dump(video_dataset,f)

                print(f"Participant {participant} correctly processed")
            except:
                print(f"Participant {participant} was not labeled")
    print("Finished")