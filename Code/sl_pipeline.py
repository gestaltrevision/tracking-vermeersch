from tw_classification import *

classes=['Y3A6090', 'Y3A6092', 'Y3A6096', 'Y3A6098', 'Y3A6101']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


sl_data_dir="/content/drive/My Drive/sl/slTest"
sl_size=11
data_transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.34839347, 0.32749262, 0.37307963], [0.11140174, 0.10704455, 0.11354395])
                            ])

sl_dataset= SlidingWindowData(sl_data_dir,"p1_y3A6090",transform=data_transform)
                        

dataset_loader = DataLoader(sl_dataset,batch_size=sl_size, shuffle=False, num_workers=4)
                                        
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5) ##important to change #of classes
weights_path="/content/drive/My Drive/model_class_init.pth"

model_ft.load_state_dict(torch.load(weights_path))

model_ft = model_ft.to(device)
model_ft.eval()

table_dict={}
time=[]
pic_class=[]
sl_vector=[]
for sl, images in enumerate(dataset_loader):
    images = images.to(device)
    preds=images_to_predictions(model_ft,images,threshold,device)
    av_pred=mode(preds)
    #Fill row of table
    #class
    pic_class.append("Rest" if av_pred==-1 else classes[av_pred])
    sl_vector.append(sl)
    #sl to time
    seconds=(sl_size*skip)/fps
    time.append(strftime("%M:%S", gmtime(seconds*sl)))

table_dict["class"]=pic_class
table_dict["sl"]=sl_vector
table_dict["time"]=time

