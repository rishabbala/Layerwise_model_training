import torch
import torch.nn as nn
from collections import OrderedDict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

## https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class mySequential(nn.Sequential):

    def forward(self, inputs, **kwargs):
        for module in self._modules.values():
            inputs = module(inputs, **kwargs)
        return inputs


class ViT(nn.Module):

    def __init__(self, feature_dim=768, mlp_dim=3072, num_blocks=12, num_heads=12, output_size=10, stochastic_depth=0.1):
        
        super().__init__()
        self.patch_size = 4
        self.num_patches = 8 * 8
        self.patch_dim = 3 * self.patch_size * self.patch_size
        self.feature_dim = feature_dim
        self.mlp_dim = mlp_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.patch_embedding = nn.Sequential(OrderedDict([
            ('gen_patches', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)),
            ('linear', nn.Linear(in_features=self.patch_dim, out_features=self.feature_dim))
        ]))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.feature_dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        self.drop_path_rate = [x.item() for x in torch.linspace(0, stochastic_depth, num_blocks)]

        # self.dropout = nn.Dropout(0.1)

        od = OrderedDict()
        for i in range(num_blocks):
            od['block'+str(i)] = TransformerBlock(self.feature_dim, self.mlp_dim, self.num_heads, drop_path_rate=self.drop_path_rate[i])

        self.transformer = mySequential(od)
        self.mlp = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(self.feature_dim)),
            ('linear', nn.Linear(in_features=self.feature_dim, out_features=output_size))
        ]))


    def forward(self, x, training=False):
        x = self.patch_embedding(x)

        class_token = repeat(self.class_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embedding

        # x = self.dropout(x)

        x = self.transformer(x, training=training)
        
        f = x[:, 0]
        f = self.mlp(f)

        return f



class MultiHeadAttention(nn.Module):

    def __init__(self, feature_dim, num_heads, dim_per_head=64):

        super().__init__()

        self.num_heads = num_heads
        self.norm = dim_per_head ** -0.5
        self.feature_dim = feature_dim

        self.to_qkv = nn.Linear(self.feature_dim, 3 * self.feature_dim, bias=False) 
        self.attention = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(in_features=self.feature_dim, out_features=feature_dim)


    def forward(self, x):
        # Obtained from timm: github.com:rwightman/pytorch-image-models
        B, N, C = x.shape

        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, self.num_heads, num_patches, dim_per_head)

        q, k, v = qkv[0], qkv[1], qkv[2]

        dot_prod = torch.matmul(q, k.transpose(-1, -2)) * self.norm
        attention = self.attention(dot_prod)
        # attention = self.attention_dropout(attention)

        out = torch.matmul(attention, v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out


class TransformerBlock(nn.Module):

    def __init__(self, feature_dim, mlp_dim, num_heads, drop_path_rate=0.1):

        super().__init__()

        self.norm0 = nn.LayerNorm(normalized_shape=feature_dim)
        self.MHA = MultiHeadAttention(feature_dim=feature_dim, num_heads=num_heads, dim_per_head=64)
        self.norm1 = nn.LayerNorm(normalized_shape=feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Linear(in_features=mlp_dim, out_features=feature_dim),
        )

        self.drop_path = DropPath(drop_path_rate)


    def forward(self, x, **kwargs):

        training = kwargs['training']

        y = self.norm0(x)
        y = self.MHA(y)

        y = self.drop_path(y, training) + x
        # y = y + x

        z = self.norm1(y)
        z = self.mlp(z)

        z = self.drop_path(z, training) + y
        # z = z + y

        return z


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training):
        return drop_path(x, self.drop_prob, training)
