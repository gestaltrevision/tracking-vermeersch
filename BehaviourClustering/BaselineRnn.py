import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.8)
                            
        self.relu=nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out=F.softmax(out,dim=1)
        return out

class Trainer(object):
  def __init__(self,model,criterion,optimizer,device):
    self.model=model
    self.criterion=criterion
    self.optimizer=optimizer
    self.device=device

  def train(self,num_epochs,train_loader):
    # Train the model
    self.model.train()
    total_step = len(train_loader)
    loss_arr=[]
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            # samples = samples.reshape(-1, sequence_length, input_size).to(self.device)
            samples=samples.to(self.device)
            labels=torch.flatten(labels).to(self.device)
            # Forward pass
            outputs = self.model(samples.float())
            loss = self.criterion(outputs, labels.long())
            #save loss
            loss_arr.append(loss.item())
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            #Gradient Flow to debug
            plot_grad_flow(self.model.named_parameters())
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            #Get eval metrics?
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            if(np.isnan(loss.item())):
              break
    return loss_arr
  # def plot_summary(self):
  #       plt.plot(self.loss_arr)
  #       plt.xlabel("Iterations")
  #       plt.ylabel("Loss")
  #       plt.show()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters):
  '''Plots the gradients flowing through different layers in the net during training.
  Can be used for checking for possible gradient vanishing / exploding problems.
  
  Usage: Plug this function in Trainer class after loss.backwards() as 
  "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
  ave_grads = []
  max_grads= []
  layers = []
  for n, p in named_parameters:
      if(p.requires_grad) and ("bias" not in n):
          layers.append(n)
          ave_grads.append(p.grad.abs().mean())
          max_grads.append(p.grad.abs().max())
  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  # plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")