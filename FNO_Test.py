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
import time
torch.manual_seed(3407)
np.random.seed(0)
import os





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
    def __init__(self, modes1, modes2, width):
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

        self.p = nn.Linear(8, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
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


# 检查命令行参数
if len(sys.argv) < 5:
    print("Usage: python3 script_name.py model_path train_data_path [train_sdf_data_path] test_data_path [test_sdf_data_path]")
    sys.exit(1)

# 从命令行获取参数
model_path = sys.argv[1]
train_data_path = sys.argv[2]
train_sdf_data_path = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].endswith('.npy') else None
test_data_path = sys.argv[3] if train_sdf_data_path is None else sys.argv[4]
test_sdf_data_path = sys.argv[4] if train_sdf_data_path is None else (sys.argv[5] if len(sys.argv) > 4 else None)

# 确保模型路径存在
if not os.path.exists(model_path):
    print(f"Error: Model path '{model_path}' does not exist.")
    sys.exit(1)

# 确保训练数据路径存在
if not os.path.exists(train_data_path):
    print(f"Error: Training data path '{train_data_path}' does not exist.")
    sys.exit(1)

# 确保测试数据路径存在
if not os.path.exists(test_data_path):
    print(f"Error: Test data path '{test_data_path}' does not exist.")
    sys.exit(1)

# 如果提供了训练 SDF 数据路径，确保其存在
if train_sdf_data_path and not os.path.exists(train_sdf_data_path):
    print(f"Error: Train SDF data path '{train_sdf_data_path}' does not exist.")
    sys.exit(1)

# 如果提供了测试 SDF 数据路径，确保其存在
if test_sdf_data_path and not os.path.exists(test_sdf_data_path):
    print(f"Error: Test SDF data path '{test_sdf_data_path}' does not exist.")
    sys.exit(1)

# 加载训练数据
print("Loading training data for initialization...")
data_train = np.load(train_data_path)
ntrain = data_train.shape[0]
T_in = 5
T_out = 10

train_a = data_train[:, :, :, :T_in]
train_u = data_train[:, :, :, T_in:]

# 将训练数据转换为 Tensor
train_a = torch.Tensor(train_a)
train_u = torch.Tensor(train_u)

# 数据归一化
a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
y_normalizer = GaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

# 如果提供了训练 SDF 数据路径，则加载和处理 SDF 数据
if train_sdf_data_path:
    sdf_train = np.load(train_sdf_data_path)
    sdf_train = torch.from_numpy(sdf_train).unsqueeze(-1).float()
    sdf_a_normalizer = GaussianNormalizer(sdf_train)
    sdf_train = sdf_a_normalizer.encode(sdf_train)
    train_a = torch.cat((train_a, sdf_train), -1)

# 加载测试数据
print("Loading test data...")
data_test = np.load(test_data_path)
ntest = data_test.shape[0]

test_a = data_test[:, :, :, :T_in]
test_u = data_test[:, :, :, T_in:]

# 将测试数据转换为 Tensor
test_a = torch.Tensor(test_a)
test_u = torch.Tensor(test_u)

# 使用训练数据的归一化器对测试数据进行归一化
test_a = a_normalizer.encode(test_a)
# test_u = y_normalizer.encode(test_u)

# 如果提供了测试 SDF 数据路径，则加载和处理 SDF 数据
if test_sdf_data_path:
    sdf_test = np.load(test_sdf_data_path)
    sdf_test = torch.from_numpy(sdf_test).unsqueeze(-1).float()
    sdf_test = sdf_a_normalizer.encode(sdf_test)
    test_a = torch.cat((test_a, sdf_test), -1)

# 初始化模型和设备
modes = 32
width = 48
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FNO2d(modes, modes, width).to(device)

# 加载模型
print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 准备测试数据加载器
batch_size = 16
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, pin_memory=True)

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        start_time = time.time()
        tmpout = model(x).view(x.shape[0], patch_size, patch_size, T_out)
        end_time = time.time()
        timeList.append(end_time-start_time)
        out = tmpout
        i=0
        while i < 16:
            i = i + 1    
            x = torch.cat((tmpout[:,:,:,5:10],x[:,:,:,5:6]), dim=3)
            start_time = time.time()
            tmpout = model(x).view(x.shape[0], patch_size, patch_size, T_out)
            end_time = time.time()
            timeList.append(end_time-start_time)
            out = torch.cat((out,tmpout),dim=3)
            # print(out.shape)
        out = y_normalizer.decode(out)
        for homme in range(total_frame):
            train_step_rela = torch.mean(torch.sqrt(torch.mean(torch.square(out[:,:,:,homme] - y[:,:,:,homme]), axis = (1,2)) / torch.mean(torch.square(y[:,:,:,homme]), axis = (1,2))))
            tttList.append(train_step_rela)
        totaltttList.append(tttList)
        out = out[:,:,:,:total_frame]
        relative_error_test = torch.mean(torch.sqrt(torch.mean(torch.square(out - y[:,:,:,:total_frame]), axis = (1,2)) / torch.mean(torch.square(y[:,:,:,:total_frame]), axis = (1,2))))

# Function to reshape and save tensors
def reshape_and_save(tensor, filename):
    # Reshape the 16 (64x64) patches into one (256x256) image
    tensor_reshaped = tensor.view(4, 4, 64, 64, -1)  # Split into 4x4 grid
    tensor_reshaped = tensor_reshaped.permute(0, 2, 1, 3, 4).contiguous()  # Swap axes
    tensor_reshaped = tensor_reshaped.view(1, 256, 256, -1)  # Reshape to (1, 256, 256, num)
    # Convert to numpy and save as npy
    tensor_np = tensor_reshaped.cpu().numpy()  # Move to CPU and convert to numpy
    np.save(filename, tensor_np)

    # Notify the user
    print(f"Tensor saved as {filename}")

# Assuming y and out are your tensors (torch.Size([16, 64, 64, num]))
# Call the function to reshape and save them
reshape_and_save(y, "reshaped_GroundTruth.npy")
reshape_and_save(out, "reshaped_Prediction.npy")

