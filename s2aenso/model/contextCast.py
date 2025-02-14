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
    def __init__(self, h_channels: int, kernel_size: int = 7, activation_fn = nn.Tanh()):
        '''
        :param x_channels: Input channels   
        :param h_channels: Latent state channels
        :param kernel_size: Convolution kernel size
        :param activation_fn: Output activation function
        '''
        super().__init__()
        self.spatial_mixing = nn.Conv2d(h_channels, h_channels, kernel_size, padding='same', groups= h_channels)
        self.channel_mixing = nn.Conv2d(h_channels, 4 * h_channels, 1)

        self.separate_gates = Rearrange('b (gates c) h w -> gates b c h w', gates = 4, c = h_channels)
        
        self.activation_fn = activation_fn
        self.gate_norm = nn.GroupNorm(num_channels = h_channels, num_groups = 4, affine = False)
        self.output_norm = nn.GroupNorm(num_channels = h_channels, num_groups = 1, affine = True)

        nn.init.dirac_(self.spatial_mixing.weight)
        nn.init.dirac_(self.channel_mixing.weight)
        nn.init.ones_(self.channel_mixing.bias[:h_channels]) * 2

    def forward(self, x, h, c, context = None):
        '''
        LSTM forward pass
        :param x: Input
        :param h: Hidden state
        :param c: Cell state
        '''
        z = x + h if x is not None else h
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


class TailEnsemble(nn.Module):

    def __init__(self, input_dim, output_dim, patch_size: tuple, num_tails: int, tail_dim: int = 1,
                 activation: nn.Module = nn.GELU):
        super().__init__()
        self.tail_dim = tail_dim
        patch_dim = output_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.tails = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(input_dim, input_dim, kernel_size = 1),
                activation(),
                nn.Conv3d(input_dim, patch_dim, kernel_size = 1),
                nn.ConvTranspose3d(patch_dim, output_dim, kernel_size = patch_size, stride = patch_size, groups = output_dim)
                ) for _ in range(num_tails)
            ])

    def forward(self, x: th.Tensor):
        '''
        Args:
            x: input tensor (batch_size, *input_shape)
        Returns:
            output tensor (batch_size, num_tails, *output_shape)
        '''
        return th.stack([tail(x) for tail in self.tails], dim = self.tail_dim)

    
class ContextCast(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_channels: int,
                 output_dim: int,
                 patch_size: tuple = (4,4),
                 num_layers: int = 2,
                 num_conditions: int = -1,
                 k_conv: int = 7,
                 num_tails: int = 16,
                 ) -> None:
        '''
        Context cast model.
        Args:
            input_dim: The number of input channels
            num_channels: The dimension of the latent space.
            patch_size: The size of the latent patches.
            num_layers: The number of LSTM layers in the model.
            residual_forecast: Whether to use residual connections in the forecast.
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
        #define model layers
        self.processor = nn.ModuleDict()
        self.processor['to_latent'] = nn.Conv3d(input_dim, num_channels, 
                                                kernel_size=(1 , *patch_size), 
                                                stride = (1, *patch_size))
        for i in range(num_layers):
            self.processor[f'encoder_lstm_{i}'] = SwinLSTM(num_channels, k_conv)
            if i == 0:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(num_channels, k_conv)
            else:
                self.processor[f'decoder_lstm_{i}'] = SwinLSTM(num_channels, k_conv)
        
        
        self.processor['to_data'] = TailEnsemble(num_channels, output_dim, 
                                                 (1 , *patch_size), num_tails)
                
        if self.use_film:
            self.processor['embedding'] = nn.Embedding(
                num_embeddings = num_conditions, embedding_dim = 8 * num_channels) 
            nn.init.uniform_(self.processor['embedding'].weight, 0.05, 0.05)

    def forward(self, x: th.Tensor, context: th.Tensor):
        '''
        Forward pass of the model.
        Args:
            x: Input tensor.
            horizon: The number of steps to predict.
            context: Monthly conditioning input.'''
        batch, _, history, height, width = x.shape
        horizon = context.shape[1] - history
        h = [th.zeros((batch, self.num_channels, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        c = [th.zeros((batch, self.num_channels, height // self.patch_size[0], width // self.patch_size[1]), device = x.device) for _ in range(self.num_layers)]
        #encoder
        patches = self.processor['to_latent'](x)
        for t in range(history):
            z = patches[:, :, t]
            u = self.processor['embedding'](context)[:,t] if self.use_film else None
            for i in range(self.num_layers):
                h[i], c[i] = self.processor[f'encoder_lstm_{i}'](z, h[i], c[i], context=u) 
                z += h[i]
        #decoder
        z_out = []
        for t in range(horizon):
            z = None
            u = self.processor['embedding'](context)[:, history+t] if self.use_film else None
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