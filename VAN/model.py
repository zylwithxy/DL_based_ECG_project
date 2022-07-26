import torch
import torch.nn as nn
import torch.nn.functional as F
from van_params import hyperparams_config
import math
import sys
sys.path.insert(0, "..")
# from Metrics import cal_sum_param

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class DWConv(nn.Module):
    """
    Depthwise convolution
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # Every filter only considers one channel
        # self.dwconv = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, (1, 3), (1, 1), (0, 1), bias=True, groups=dim)

    def forward(self, x):
        """
        # x: shape(C, 1, L) omitting the batch size
        x: shape(10, 12, L) omitting the batch size
        """
        x = self.dwconv(x) # x: (10, 12, L)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features # --> out_features = out_features if out_features else in_features
        hidden_features = hidden_features or in_features # --> hidden_features = hidden_features if hidden_features else in_features
        # self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1) # basis change
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # 2D
            # fan_out = m.kernel_size[0] * m.out_channels # 1D
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        # x: Tensor shape(in_features, l) 1D
        # x: Tensor shape(in_features, 1, l) 2D
        x: Tensor shape(in_features, 12, l) 2D shape
        """
        x = self.fc1(x) # x shape:(hidden_features, 12, l)
        x = self.dwconv(x) # x shape:(hidden_features, 12, l)
        x = self.act(x) # x shape:(hidden_features, 12, l)
        x = self.drop(x)
        x = self.fc2(x) # x shape:(out_features, 12, l)
        x = self.drop(x)
        return x

# LKA attention
class LKA(nn.Module):
    def __init__(self, dim):
        """
        (standard convolution)
        K * K ---------------------> (K/d, K/d) + (2*d -1, 2*d -1) + (1, 1)
        """
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, (1, 5), padding= (0,2), groups=dim) # d= 3
        self.conv_spatial = nn.Conv2d(dim, dim, (1,7), stride= (1,1), padding=(0, 9), groups=dim, dilation=(1,3))
        # 2 * (7-1) + 7= 19 = 2 * 9 + 1
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        """
        # x: Tensor shape(in_features, l) 1D
        # x: Tensor shape(in_features, 1, l) 2D
        x: Tensor shape(in_features, 12, l) 2D
        """
        u = x.clone()        
        attn = self.conv0(x) # attn: Tensor shape(in_features, 12, l)
        attn = self.conv_spatial(attn) # attn: Tensor shape(in_features, 12, l)
        attn = self.conv1(attn) # attn: Tensor shape(in_features, 12, l)

        return u * attn # # Tensor shape: (in_features, 12, l)

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        """
        # x: Tensor shape(in_features, l)
        # x: Tensor shape(in_features, 1, l)
        x: Tensor shape(in_features, 12, l) 2D
        """
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut # x: Tensor shape(in_features, 12, l)
        return x 

class Block(nn.Module): # equals to L in stage in VAN
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        # self.norm1 = nn.BatchNorm1d(dim) # 1D
        self.norm1 = nn.BatchNorm2d(dim) # 1D
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2 = nn.BatchNorm1d(dim) # 1D
        self.norm2 = nn.BatchNorm2d(dim) # 1D
        mlp_hidden_dim = int(dim * mlp_ratio) # Hidden number 
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True) 
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # fan_out = m.kernel_size[0] * m.out_channels
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        # x: (embed_dim, L) 1D
        # x: (embed_dim, 1, L) 2D
        x: (embed_dim, 12, L) 2D
        """
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))) # Residual connection LKA
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))) # Residual connection CFF
        return x # x: (embed_dim, 12, L) 2D

    """
    self.drop_path: Which is similar to dropout(which is used if self.training == True)
       keep_prob = 1 - drop_prob
       shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
       random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
       random_tensor.div_(keep_prob) if keep_prob > 0
       return x * random_tensor
    """

class OverlapPatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding (Downsampling)
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        # patch_size = to_2tuple(patch_size) # patch_size -> (patch_size, patch_size)

        # self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
        #                       padding= patch_size // 2)
        patch_size = (1, patch_size) # x shape (in_chans, 1, L)
        stride = (1, stride)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(0, patch_size[1] // 2))
        # self.norm = nn.BatchNorm1d(embed_dim)
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        # x: (in_chans, L)  1D
        # x: (in_chans, 1, L) 2D
        x: (in_chans, 12, L) 2D
        """
        x = self.proj(x) # x: (embed_dim, 12, L/stride)
        _, _, lead, L = x.shape 
        x = self.norm(x) # x: (embed_dim, 12, L/stride)  
        return x, lead, L

class VAN(nn.Module):
    def __init__(self, *, img_size= 2000, in_chans= 10, num_classes= 9, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        """
        drop_rate: Control the dropout rate in MLP
        drop_path_rate: Control the dropout rate between LKA(MLP) and MLP(LayerNorm)
        """
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            # norm = norm_layer(embed_dims[i] * 12) # New feature map
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm) # norm follows block

        # classification head
        # self.head = nn.Linear(embed_dims[3] * 12, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # hd_size = 12
        # self.head = nn.Linear(756 * hd_size, num_classes) if num_classes > 0 else nn.Identity()

        # LSTM layer
        # self.lstm = nn.LSTM(input_size= embed_dims[3], hidden_size= hd_size, batch_first= True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # fan_out = m.kernel_size[0] * m.out_channels
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # @torch.jit.ignore  用于torch script 中保存 torch script 无法编译的Python代码 
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, lead, L = patch_embed(x) # x shape (B, embed_dim, 12, L/stride)
            for blk in block:
                x = blk(x) # x shape (B, embed_dim, 12, L/stride)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x) # LayerNorm last dim
            if i != self.num_stages - 1:
                # x = x.transpose(1, 2).contiguous()
                x = x.reshape(B, lead, L, -1).permute(0, 3, 1, 2).contiguous()
            
            """      
            else:
                x, _, = self.lstm(x)
                x = x.flatten(1)
            """
        # print(x.shape) # shape: (B, 756, 256) VAN_tiny
        return x.mean(dim=1)
        # return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

if __name__ == '__main__':

    """

    # Cal params
    print(f'The sum params are {cal_sum_param(transformer)}')
    """
    print("Starting main")
    diction = hyperparams_config()
    v = VAN(**diction).cuda()
    with torch.no_grad():
        img = torch.randn(2, 10, 12, 2000).cuda()
        preds = v(img)
        #param_num = cal_sum_param(v) # 3385097
        # print(param_num)
        print(preds.shape)