import torch
import math
import sys
import numpy as np
import torch.nn as nn
from lstm_cell import LSTM
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision as T
import argparse
import os
import glob
import tqdm
import pickle
from torch.autograd import Variable
parser = argparse.ArgumentParser(description='sequential MNIST parameters')
parser.add_argument('--p-detach', type=float, default=0.25, help='probability of detaching each timestep')
parser.add_argument('--permute', type=int, default=1, help='pMNIST or normal MNIST')
parser.add_argument('--save-dir', type=str, default='h_detach_0.25_mnist_0.0001', help='save directory')
parser.add_argument('--lstm-size', type=int, default=100, help='width of LSTM')
parser.add_argument('--seed', type=int, default=400, help='seed value')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--clipval', type=float, default=1., help='gradient clipping value')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--anneal-p', type=int, default=40, help='number of epochs before total number of epochs for setting p-detach to 0')
parser.add_argument('--loadsaved',type=int,default=1)


args = parser.parse_args()
log_dir = args.save_dir

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

# if os.path.isdir(log_dir):
# 	if len(glob.glob(log_dir+'events.*'))>0:
# 		print ('TensorBoard file exists by this name. Please delete it manually using \nrm -f {} \nor choose another save_dir.'.format(glob.glob(log_dir+'events.*')[0]))
# 		exit(0)

writer = SummaryWriter(log_dir=log_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
tensor = torch.FloatTensor

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000))
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000, 60000))

trainset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
testset = T.datasets.MNIST(root='./MNIST', train=False, download=True, transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler=valid_sampler, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=2)

n_epochs = args.n_epochs
batch_size = args.batch_size
hid_size = args.lstm_size
lr = args.lr

T = 784
inp_size = 1
out_size = 10
train_size = 60000
test_size = 10000
clipval = float(args.clipval) if args.clipval>0 else float('inf')

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size):
		super().__init__()
		#self.lstm = h_detach(inp_size, hid_size,args.p_detach)
		self.lstm=LSTM(inp_size,hid_size)
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		x1,(h,c) = self.lstm(x, state)
		#x1=Variable(x1,requires_grad=True)
		x2 = self.fc1(x1)
		
		return x2,h,c

def test_model(model, loader, criterion, order):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i, data in enumerate(loader, 1):
			test_x, test_y = data
			test_x = test_x.view(-1, 784, 1)
			test_x, test_y = test_x.to(device), test_y.to(device)
			test_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)

			for j in order:
				 outputs,h, c = model(test_x[j], (h, c))

			loss += criterion(outputs, test_y).item()
			preds = torch.argmax(outputs, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	loss /= 100.0
	return loss, accuracy

def train_model(model, epochs, criterion, optimizer):
	acc=[]
	lossstats=[]
	best_acc = 0.0
	ctr = 0	
	global lr
	if args.permute==1:
		order = np.random.permutation(T)
	else:
		order = np.arange(T)

	test_acc = 0
	start_epoch=0
	ctr=0
	if args.loadsaved==1:
		with open(log_dir+'/accstats.pickle','rb') as f:
			acc=pickle.load(f)
		with open(log_dir+'/lossstats.pickle','rb') as f:
			losslist=pickle.load(f)
		start_epoch=len(acc)-1
		best_acc=0
		for i in acc:
			if i[0]>best_acc:
				best_acc=i[0]
		ctr=len(losslist)-1
		
	for epoch in range(start_epoch,epochs):
		if epoch>epochs-args.anneal_p:
			args.p_detach=-1
		print('epoch ' + str(epoch + 1))
		epoch_loss = 0.
		iter_ctr = 0.
		for data in tqdm.tqdm(trainloader):
			iter_ctr+=1.
		# for z, data in enumerate(trainloader, 0):
			inp_x, inp_y = data
			inp_x = inp_x.view(-1, 28*28, 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			h = torch.zeros(batch_size, hid_size).to(device)
			c = torch.zeros(batch_size, hid_size).to(device)
			sq_len = T
			loss = 0

			for i in order:
				if args.p_detach >0:
					val = np.random.random(size=1)[0]
					if val <= args.p_detach:
						h = h.detach()
				output, h, c = model(inp_x[i].contiguous(), (h, c))
			#print('-------------------------')
			loss += criterion(output, inp_y)

			model.zero_grad()
			#print(type(output))
			loss.backward()
			#print(grads['x1'])

			norms = nn.utils.clip_grad_norm_(model.parameters(), clipval)

			optimizer.step()


			loss_val = loss.item()
			#print(loss_val)
			epoch_loss += loss_val

			# print(z, loss_val)
			# writer.add_scalar('/hdetach:loss', loss_val, ctr)
			ctr += 1

		v_loss, v_accuracy = test_model(model, validloader, criterion, order)
		if best_acc < v_accuracy:
			best_acc = v_accuracy
			print('best validation accuracy ' + str(best_acc))
			print('Saving best model..')
			state = {
	        'net': model,
	        'hid_size': hid_size,
	        'epoch':epoch,
	    	'ctr':ctr,
	    	'best_acc':best_acc
	    	}
			with open(log_dir + '/best_model.pt', 'wb') as f:
				torch.save(state, f)
			_, test_acc = test_model(model, testloader, criterion, order)
		print('epoch_loss: {}, val accuracy: {} '.format(epoch_loss/(iter_ctr), v_accuracy))
		lossstats.append((ctr,epoch_loss/iter_ctr))
		acc.append((epoch,v_accuracy))
		with open(log_dir+'/lossstats.pickle','wb') as f:
			pickle.dump(lossstats,f)
		with open(log_dir+'/accstats.pickle','wb') as f:
			pickle.dump(acc,f)
		writer.add_scalar('/hdetach:val_acc', v_accuracy, epoch)
		writer.add_scalar('/hdetach:epoch_loss', epoch_loss/(iter_ctr), epoch)

		#print(model.lstm.weights)
	print('best val accuracy: {} '.format( best_acc))
	writer.add_scalar('/hdetach:best_val_acc', best_acc, 0)
	print('test accuracy: {} '.format( test_acc))
	writer.add_scalar('/hdetach:test_acc', test_acc, 0)

device = torch.device('cuda')

net = Net(inp_size, hid_size, out_size).to(device)
if args.loadsaved==1:
	modelstate=torch.load(log_dir+'/best_model.pt')
	net.load_state_dict(modelstate['net'].state_dict())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train_model(net, n_epochs, criterion, optimizer)
writer.close()
