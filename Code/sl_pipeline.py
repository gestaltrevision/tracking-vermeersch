from tw_classification import *

import time
import argparse
from tqdm import tqdm
from statistics import StatisticsError
parser = argparse.ArgumentParser()

parser.add_argument("--main_dir", type=str,
                    help="Base directory of project")

parser.add_argument("--video_name", type=str,
                    help="Folder containing all clips of same artwork")




if __name__ == "__main__":
    #Sample usage for sampling
    #python sl_pipeline.py --main_dir "C:\Users\jeuux\Desktop\Carrera\MoAI\TFM" --video_name "2vre55a"
    args = parser.parse_args()
    data_path=os.path.join(args.main_dir,"AnnotatedData")
    utilities_path=os.path.join(args.main_dir,"Utilities")
    
    classes=['Y3A6090', 'Y3A6092', 'Y3A6096', 'Y3A6098', 'Y3A6101']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Loading Datasets with frames from video
    sl_data_dir=os.path.join(data_path,"Sl_Dataset",args.video_name)

    sl_size=11
    data_transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.34839347, 0.32749262, 0.37307963], [0.11140174, 0.10704455, 0.11354395])
                                ])

    sl_dataset= SlidingWindowData(sl_data_dir,args.video_name,transform=data_transform)
                            

    dataset_loader = DataLoader(sl_dataset,batch_size=sl_size, shuffle=False, num_workers=4)

    #Loading trained network
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 5) ##important to change #of classes
    weights_path=os.path.join(utilities_path,"Weights","model_class_init.pth")
    model_ft.load_state_dict(torch.load(weights_path, map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()

    #Classifying frames from Sliding Windows
    table_dict={}
    time_vect=[]
    pic_class=[]
    sl_vector=[]
    threshold=0.9
    skip=3
    fps=25.014717418571774
    # video = cv2.VideoCapture(video_path)
    # fps=video.get(cv2.CAP_PROP_FPS)
    # video.release()


    print("Starting to classify")
    for sl, images in tqdm(enumerate(dataset_loader)):
        images = images.to(device)
        preds=images_to_predictions(model_ft,images,threshold,device)
        try :
            av_pred=mode(preds)
        except StatisticsError:
            av_pred=-1
        #Fill row of table
        #class
        pic_class.append("Rest" if av_pred==-1 else classes[av_pred])
        sl_vector.append(sl)
        #sl to time_vect
        seconds=(sl_size*skip)/fps
        time_vect.append(strftime("%M:%S", gmtime(seconds*sl)))

    table_dict["class"]=pic_class
    table_dict["sl"]=sl_vector
    table_dict["time"]=time_vect

    #Saving results
    df=pd.DataFrame(table_dict)
    results_folder=os.path.join(data_path,"Sl_Annotations")
    table_path=os.path.join(results_folder,"{0}_classification.csv".format(args.video_name))
    df.to_csv(table_path)

    print("Video Successfully classified ")
