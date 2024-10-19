# import all modules and our model layers
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys,os 
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
#from Adam import Adam
from tensorflow.keras.optimizers import Adam
torch.manual_seed(3407)
np.random.seed(0)
from IPython.display import clear_output
from tqdm import tqdm



# Check command-line arguments
if len(sys.argv) < 3:
    print("Usage: python3 script_name.py model_path train_data_path [sdf_data_path]")
    sys.exit(1)

model_path = sys.argv[1]
train_data_path = sys.argv[2]
sdf_data_path = sys.argv[3] if len(sys.argv) > 3 else None

# Verify training data path exists
if not os.path.exists(train_data_path):
    print(f"Error: Training data path '{train_data_path}' does not exist.")
    sys.exit(1)

# Verify SDF data path exists if provided
if sdf_data_path and not os.path.exists(sdf_data_path):
    print(f"Error: SDF data path '{sdf_data_path}' does not exist.")
    sys.exit(1)
    

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, num, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(num+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)

        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 10, self.width * 4) 

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = x
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = x
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = x
        x = x1 + x2
        x = F.gelu(x)     
        
        
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = x
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



# Load data
print("Loading data...")
data_train = np.load(train_data_path)
ntrain = data_train.shape[0]
T_in = 5
T_out = 10
train_a = data_train[:, :, :, :T_in]
train_u = data_train[:, :, :, T_in:T_in + T_out]

# Torch-lize data
train_a = torch.Tensor(train_a)
train_u = torch.Tensor(train_u)

# Normalize data
a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
y_normalizer = GaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)
inputNum = 5
# Load and process SDF data if provided
if sdf_data_path:
    sdf_train = np.load(sdf_data_path)
    sdf_train = torch.from_numpy(sdf_train).unsqueeze(-1).float()
    sdf_a_normalizer = GaussianNormalizer(sdf_train)
    sdf_train = sdf_a_normalizer.encode(sdf_train)
    train_a = torch.cat((train_a, sdf_train), -1)
    inputNum = 6

# Prepare data loader
batch_size = 100
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, pin_memory=True)

# Initialize model
modes = 32
width = 48
model = FNO2d(inputNum, modes, modes, width)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, verbose=True)


# Load existing model if model path exists
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epochs_done = checkpoint['epoch']
else:
    print(f"Training new model. Saving to {model_path}...")
    epochs_done = 0

# Training loop
epochs = 100

for ep in tqdm(range(epochs_done, epochs)):
    model.train() # we let it train!
    t1 = default_timer() # ok we get a timer
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device) # we put them to device at first     
        optimizer.zero_grad() #initialize the grad of optimizer
        out = model(x) 
        mse = F.mse_loss(out, y, reduction='mean')
        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        relative_error_train = torch.mean(torch.sqrt(torch.mean(torch.square(out - y), axis = (1,2)) / torch.mean(torch.square(y), axis = (1,2))))
        relative_error_train.backward()
        optimizer.step()
    scheduler.step()
    t2 = default_timer()
    print(f'{ep}, {t2-t1:.2f}')
    #finally we just saved our data
    if ep%5 == 0:
      torch.save({
            'epoch': ep+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, model_path+'.pt')
    print(f'saved epoch {ep} successfully!')
    
    