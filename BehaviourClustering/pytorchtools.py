import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def to_numpy(v):
    if is_list_or_tuple(v):
        return np.stack([to_numpy(sub_v) for sub_v in v], axis=1)
    try:
        return v.cpu().detach().numpy()

    except AttributeError:
        return v


def set_requires_grad(model, requires_grad):
  for param in model.parameters():
    param.requires_grad = requires_grad
    
def load_pretrained_model(model_name, model_dir, model, device):
  try:
    checkpoint_file = next(file for file in os.listdir(model_dir) if (f"{model_name}_best" in file))
  except:
    checkpoint_file = next(file for file in os.listdir(model_dir) if (f"{model_name}" in file))
  checkpoint_path=os.path.join(model_dir,checkpoint_file)
  model.load_state_dict(torch.load(checkpoint_path))
  return model.to(device)

def get_samples_from_class(data_loader,label,n_samples = 5):
  #get sample batch
  samples, labels = next(iter(data_loader))
  #filter samples
  label_idx = (labels == label).nonzero()
  try:
    random_idx = np.random.choice(np.arange(len(label_idx)), size = n_samples,replace = False)
    label_idx  = label_idx[random_idx]
  except:
    print("Less samples in class than n_samples")
  samples_filtered = to_numpy(samples[label_idx])

  return samples_filtered

def plot_samples_from_class(data_loader,label,title,n_samples = 4):
  #get samples
  samples_filtered  = get_samples_from_class(data_loader,label,n_samples)
  if(len(samples_filtered) < n_samples):
    n_samples = len(samples_filtered) + 1
  n_components = samples_filtered.shape[-1]
  #plot samples
  _, axs = plt.subplots(n_components, n_samples, sharex=True, sharey=False)

  for sample_id in range(len(samples_filtered)):
    sample = samples_filtered[sample_id]
    axs = create_subplot_sample(sample,axs,sample_id,n_components)
  plt.title(title)
  plt.show()

def create_subplot_sample(sample,axs,sample_id,n_components):
    for component in range(n_components):
        axs[component,sample_id].plot(sample[0,:,component])
    return axs

    
# if __name__ == "__main__":
#     import os 
#     import numpy as np
#     import torch
#     import torch.nn as nn
#     import torch.nn.functional as F

#     from torch.utils.data import Dataset,DataLoader
#     from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
#     from BehaviourDatasets import TSDataset,TSDatasetSiamese
#     from tqdm.notebook import tqdm
#     from sklearn.preprocessing import RobustScaler,StandardScaler
#     from sklearn.preprocessing import LabelEncoder

#     scaler=StandardScaler()
#     encoder=LabelEncoder
#     folder = r"C:\Users\jeuux\Desktop\Carrera\MoAI\TFM\AnnotatedData\Accelerometer_Data\Datasets\HAR_Dataset_raw"
#     data_types=[True,True,True] #dont select gaze
#     level="AG"
#     n_components=9
#     batch_size=128
#     #creating train and valid datasets
#     train_dataset= TSDataset(folder,scaler,"Train",level,data_types)
#     validation_dataset= TSDataset(folder,scaler,"Val",level,data_types)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
#     val_loader= DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
#     data_loaders=[train_loader,val_loader]

#     plot_samples_from_class(train_loader,0,n_samples=3)
#     pass