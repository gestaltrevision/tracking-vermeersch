import torch
def prepare_batch_rnn(batch,device):
  samples,labels=batch
  samples = samples.to(device).float()
  labels =torch.flatten(labels).to(device).long()

  return samples,labels

def prepare_batch_cnn(batch,device):
  samples,labels=batch
  samples=_prepare_sample_cnn(samples,device)
  labels =torch.flatten(labels).to(device).long()
  return samples,labels

def _prepare_sample_cnn(samples,device):
    samples=torch.transpose(samples,1,2)
    return samples.to(device).float()

def prepare_batch_embeddings(batch):
  samples,labels = batch
  samples = torch.transpose(samples,1,2).float()
  labels = torch.flatten(labels).long()
  return samples,labels

def prepare_batch_siamese_cnn(batch,device):
  samples1,samples2, labels=batch
  samples1= _prepare_sample_cnn(samples1,device)
  samples2= _prepare_sample_cnn(samples2,device)
  labels =torch.flatten(labels).to(device).long()
  return [samples1,samples2],labels


