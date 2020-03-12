##Active learning

from IPython.display import Video,clear_output,display

def oracle_annotations(video_clip):
    
    clear_output(wait=True)
    display(Video(video_clip,embed=True))
    print("Type class of video...")
    y_new = np.array([int(input())], dtype=int)
    
    return y_new

class Oracle(object):
    """docstring for ClassName."""
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg


def get_metadata(query_idx,data):
    start_time=data.loc[query_idx]["start time"].values
    participant=data.loc[query_idx]["participant"].values
    return start_time,participant

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip,ffmpeg_resize

def get_unlabeled_clip(start_time,participant,query_idx,main_dir,window_duration=2):
    labeling_folder=os.path.join(main_dir,"AnnotatedData","Accelerometer_Data","labeling_folder")
    data_dir=os.path.join(main_dir,"Data")
  

    clip_path=os.path.join(labeling_folder,"clip{0}_{1}.mp4".format(participant,query_idx))
    
    participant_folder=find_participant(participant,main_dir)
    video_path=os.path.join(data_dir,participant_folder,"segments","1","fullstream_resized.mp4")
    assert os.path.isfile(video_path),"Resized version does not exit yet"

    ffmpeg_extract_subclip(video_path,start_time,start_time+window_duration, targetname=clip_path)

    return clip_path
    