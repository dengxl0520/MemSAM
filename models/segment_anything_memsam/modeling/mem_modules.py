import torch
import torch.nn as nn
import torch.nn.functional as F
from . import resnet


class BasicConv(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), 
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g

def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self, x_transform=None, method='cat', reverse_order=False):
        super().__init__()

        self.x_transform = x_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x, g):
        num_objects = g.shape[1]

        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1)], 2)
            else:
                g = torch.cat([x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1) + g
        else:
            raise NotImplementedError

        return g


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g) # 1,1,512,h,w
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        # g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
        #     self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g[0], h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        # self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
        #         nn.ReLU()
        #     )

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(128, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        # g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        # image_feat = self.downsample(image_feat)
        g = self.fuser(image_feat, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits

class ForegroundReinforcingModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, size):
        super().__init__()

        self.conv_wxh = nn.Conv2d(in_channels+1, mid_channels, size, padding=(size // 2))  # Convw×h
        self.conv_1x1 = nn.Conv2d(mid_channels, out_channels, 1)  # Conv1×1
        # self.pooling = nn.AdaptiveAvgPool2d((8,8))

    def forward(self, kQ, prev_frame_mask):
        kQ_shape = kQ.shape
        # Concatenate mt⊔1 with k′Q
        prev_frame_mask = torch.nn.functional.interpolate(prev_frame_mask,kQ.shape[-2:])
        concatenated_features = torch.cat((prev_frame_mask, kQ), dim=1)

        # Apply Convw×h to generate local attention feature Fatt
        local_attention_feature = self.conv_wxh(concatenated_features)

        # Apply Conv1×1 to transform the dimensions of Fatt
        local_attention_feature = self.conv_1x1(local_attention_feature)

        # Apply softmax to normalize local attention weights
        alpha = F.softmax(local_attention_feature, dim=1)

        # Incorporate previous frame mask
        kQ = alpha * prev_frame_mask

        return kQ

class MemoryDecoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(256, val_dim+hidden_dim, 512, 256)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(256, hidden_dim)
        else:
            self.hidden_update = None

    def forward(self, imge, hidden_state, memory_readout, h_out=True):
        if self.hidden_update is not None:
            g16 = self.fuser(imge, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(imge, memory_readout)

        if h_out and self.hidden_update is not None:
            hidden_state = self.hidden_update([g16], hidden_state)
        else:
            hidden_state = None
        
        return hidden_state, g16
