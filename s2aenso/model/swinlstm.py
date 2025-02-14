'''LSTM models with 2D inputs. 

@Author  :   Jannik Thuemmel and Jakob Schlör 
@Time    :   2022/09/11 14:47:32
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch as th
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange


class SwinLSTM(nn.Module):
    def __init__(self, x_channels: int, h_channels: int, kernel_size: int = 7, activation_fn = nn.Tanh()):
        '''
        :param x_channels: Input channels   
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        conv_channels = x_channels + h_channels
        self.spatial_mixing = nn.Conv2d(conv_channels, conv_channels, kernel_size, padding='same', groups= conv_channels)
        self.channel_mixing = nn.Conv2d(conv_channels, 4 * h_channels, 1)

        self.separate_gates = Rearrange('b (gates c) h w -> gates b c h w', gates = 4, c = h_channels)
        
        self.activation_fn = activation_fn
        self.gate_norm = nn.GroupNorm(num_channels = conv_channels, num_groups = 4, affine = False)
        self.output_norm = nn.GroupNorm(num_channels = h_channels, num_groups = 1, affine = True)

        nn.init.dirac_(self.spatial_mixing.weight)
        nn.init.dirac_(self.channel_mixing.weight)
        nn.init.ones_(self.channel_mixing.bias[:conv_channels]) * 2

    def forward(self, x, h, c, context = None):
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        z = th.cat((x, h), dim = 1) if x is not None else h
        z = self.spatial_mixing(z)
        z = self.gate_norm(z)
        z = self.channel_mixing(z)
        if context is not None:
            a, b = einops.rearrange(context, 'b (split c) -> split b c () ()', split = 2)
            z = z * (1 + a) + b

        i, f, o, g = self.separate_gates(z) 
        c = th.sigmoid(f) * c + th.sigmoid(i) * th.tanh(g)
        h = th.sigmoid(o) * self.activation_fn(self.output_norm(c))
        return h, c


class MultiHeadConvTranspose3d(nn.Module):
    '''Multihead ConvTranspose3d layer'''
    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: tuple, stride: tuple,
                 num_heads: int = 2):
        '''
        input_dim: input dimension
        output_dim: output dimension
        num_heads: number of heads
        '''
        super().__init__()
        self.heads = nn.ModuleList([
            nn.ConvTranspose3d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: th.Tensor):
        '''
        x: input tensor (batch_size, *, input_dim)
        return: output tensor (batch_size, num_heads, *, output_dim)
        '''
        return th.stack([head(x) for head in self.heads], dim = 1)

class Cutout(nn.Module):

    def __init__(self, cutout: float = 0.0, dim = [2,3]):
        super().__init__()
        self.cutout = cutout
        self.dim = dim + [0]

    def forward(self, x):
        if self.training and self.cutout > 0:
            shape = [1] * len(x.shape)
            for i in self.dim:
                shape[i] = x.shape[i]
            mask = th.rand(*shape, device=x.device) > self.cutout
            return x * mask
        else:
            return x
    
class SwinLSTMNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_channels: int,
                 patch_size: tuple = (4,4),
                 num_layers: int = 2,
                 num_conditions: int = -1,
                 k_conv: int = 7,
                 num_tails: int = 16,
                 cutout: float = 0.0,
                 step_strided_conv: bool = False,
                 ) -> None:
        '''
        ElNet: The SwinLSTMNet model.
        Args:
            input_dim: The number of input channels
            num_channels: The dimension of the latent space.
            patch_size: The size of the latent patches.
            num_layers: The number of LSTM layers in the model.
            k_conv: The kernel size of the convolutional layers.
            expansion_factor: The expansion factor of the processing blocks.
            activation_fn: The activation function of the processing blocks.'''
        
        super().__init__()
        #define model attributes
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_film = num_conditions > 0
        self.num_tails = num_tails
        self.cutout = Cutout(cutout, dim=[3,4])
        self.step_strided_conv = step_strided_conv
        #define model layers
        self.processor = nn.ModuleDict()
        if step_strided_conv is False:
            self.processor['to_latent'] = nn.Conv3d(input_dim, num_channels, 
                                                    kernel_size=(1, *patch_size), 
                                                    stride=(1, *patch_size))
            self.processor['to_data'] = MultiHeadConvTranspose3d(
                num_channels, output_dim, kernel_size=(1, *patch_size), stride=(1, *patch_size),
                num_heads=self.num_tails 
            )
        else:
            self.processor['to_latent'] = nn.Sequential(
                nn.Conv3d(input_dim, num_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
                nn.ReLU(),
                nn.Conv3d(num_channels, num_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                nn.ReLU(),
            )
            self.processor['to_data'] = nn.Sequential(
                nn.ConvTranspose3d(num_channels, num_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1)),
                nn.ReLU(),
                nn.ConvTranspose3d(num_channels, 2*input_dim, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), output_padding=(0, 1, 1)),
            )


        # TODO: Group norm
        for i in range(num_layers):
            self.processor[f'encoder_lstm_{i}'] = SwinLSTM(num_channels, num_channels, k_conv)
            if i == 0:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(0, num_channels, k_conv)
            else:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(num_channels, num_channels, k_conv)
        
                
        if self.use_film:
            self.processor['embedding'] = nn.Embedding(
                num_embeddings = num_conditions, embedding_dim = 8 * num_channels) 
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)


    def forward(self, x, horizon = 1, context = None):
        '''
        Forward pass of the model.
        Args:
            x: Input tensor.
            horizon: The number of steps to predict.
            context: Auxiliary conditioning tensor.'''
        batch, _, history, height, width = x.shape

        if self.step_strided_conv is False:
            h = [th.zeros((batch, self.num_channels, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
            c = [th.zeros((batch, self.num_channels, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        else:
            h = [None for _ in range(self.num_layers)]
            c = [None for _ in range(self.num_layers)]

        #encoder
        patches = self.processor['to_latent'](x)
        patches = self.cutout(patches)

        for t in range(history):
            z = patches[:, :, t]

            if self.use_film:
                context = context.long()  # Convert to LongTensor
                u = self.processor['embedding'](context)[:, t] 
            else:
                u = None

            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'encoder_lstm_{i}'](z, h[i], c[i], context=u) 
                z += h[i]
        #decoder
        z_out = []
        for t in range(horizon):
            z = None
            u = self.processor['embedding'](context)[:,history+t] if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'decoder_lstm_{i}'](z, h[i], c[i], context=u)    
                z = h[i] if z is None else z + h[i]
            z_out.append(z)
        z_out = th.stack(z_out, dim = 2)

        # To data space
        x_pred = self.processor['to_data'](z_out)
        if self.num_tails == 2:
            mean, sigma = x_pred.split(1, dim = 1)
            x_pred = th.cat([mean, nn.functional.softplus(sigma)], dim = 1) 
            
        return x_pred