import torch
import torch.nn as nn

## My first 2/3 layers cnn
class my_first_CNN(nn.Module):
    def __init__(self,num_channels=16,bias=True):
        super(my_first_CNN,self).__init__()
        #nn.LeakyReLU(),
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d
        #self.activation = nn.Tanh()
        self.bias = bias
        self.kernel_size = 5
        self.pad = int((self.kernel_size - 1) / 2)
        #nn.BatchNorm2d(num_features=num_channels),

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=bias,
            ),
            #nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            #nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            #nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )

        self.out = nn.Conv2d(
                in_channels=num_channels*3+1,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
        )
        
    # The initial network is set to identity
    def init_weights(self) :
        with torch.no_grad():           
            nn.init.dirac_(self.conv1[0].weight)
            nn.init.dirac_(self.conv2[0].weight)
            nn.init.dirac_(self.conv3[0].weight)
            nn.init.dirac_(self.out.weight)

    # Applying the neural net
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat((x,x1,x2,x3),dim=1)
        #x = self.slope * self.out(x4) + self.bias
        x = self.out(x4)
        return x


## My first 2/3 layers cnn
class my_first_CNN2(nn.Module):
    def __init__(self,num_channels=16,bias=True):
        super(my_first_CNN2,self).__init__()
        #nn.BatchNorm2d(num_features=num_channels),
        #nn.LeakyReLU(),
        self.activation = nn.ReLU()
        #self.activation = nn.Tanh()
        self.bias = bias
        self.kernel_size = 5
        self.pad = int((self.kernel_size - 1) / 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=bias
            ),
            self.activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias
            ),
            self.activation
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias
            ),
            self.activation
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias
            ),
            self.activation
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            self.activation
        )

        self.out = nn.Conv2d(
                in_channels=num_channels*5+1,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = torch.cat((x,x1,x2,x3,x4,x5),dim=1)
        x = self.out(x6)
        return x

## My first 2/3 layers cnn
class one_layer_CNN(nn.Module):
    def __init__(self,num_channels=16,bias=True):
        super(one_layer_CNN,self).__init__()
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d
        self.bias = bias
        self.kernel_size = 5
        self.pad = int((self.kernel_size - 1) / 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=bias,
            ),
            self.activation,
        )
        self.out = nn.Conv2d(
                in_channels=num_channels,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
        )
        
    # The initial network is set to identity
    def init_weights(self) :
        with torch.no_grad():           
            nn.init.dirac_(self.conv1[0].weight)
            nn.init.dirac_(self.out.weight)

    # Applying the neural net
    def forward(self,x):
        x1 = self.conv1(x)
        x = self.out(x1)
        return x

## My first 2/3 layers cnn
class my_CNN_batchnorm(nn.Module):
    def __init__(self,num_channels=16,bias=True):
        super(my_CNN_batchnorm,self).__init__()
        self.activation = nn.ReLU()
        self.bias = bias
        self.kernel_size = 5
        self.pad = int((self.kernel_size - 1) / 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
            ),
            nn.BatchNorm2d(num_features=num_channels),
            self.activation,
        )

        self.out = nn.Conv2d(
                in_channels=num_channels*3+1,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                bias=self.bias,
        )
        
    # The initial network is set to identity
    def init_weights(self) :
        with torch.no_grad():           
            nn.init.dirac_(self.conv1[0].weight)
            nn.init.dirac_(self.conv2[0].weight)
            nn.init.dirac_(self.conv3[0].weight)
            nn.init.dirac_(self.out.weight)

    # Applying the neural net
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = torch.cat((x,x1,x2,x3),dim=1)
        x = self.out(x4)
        return x