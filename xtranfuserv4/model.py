import os
from cognitive_transfuser.model import TransFuser
from collections import deque
import sys
import numpy as np
from torch import torch, cat, add, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import math


def kaiming_init_layer(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx,
                              stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.relu(x)
        return y


class ConvBlock(nn.Module):
    def __init__(self, channel, final=False):  # up,
        super(ConvBlock, self).__init__()
        if final:
            self.conv_block0 = ConvBNRelu(
                channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(
                channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(
                channelx=[channel[1], channel[1]], stridex=1)
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)

    def forward(self, x):
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C //
                             self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C //
                               self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors, seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + sdc
        self.pos_emb = nn.Parameter(torch.zeros(
            1, (self.config.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, sdc_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            sdc_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = sdc_tensor.shape[0] // self.seq_len
        h, w = sdc_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(
            bz, self.config.n_views * self.seq_len, -1, h, w)
        sdc_tensor = sdc_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, sdc_tensor], dim=1).permute(
            0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(
            bz, -1, self.n_embd)  # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1))  # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings +
                      velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 1) * self.seq_len,
                   self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :,
                             :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        sdc_tensor_out = x[:, self.config.n_views*self.seq_len:,
                           :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)

        return image_tensor_out, sdc_tensor_out


class GPT_Seg(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors, seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + sdc
        self.pos_emb = nn.Parameter(torch.zeros(
            1, (self.config.n_views + 2) * seq_len * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, sdc_tensor, seg, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            sdc_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = sdc_tensor.shape[0] // self.config.seq_len
        h, w = sdc_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(
            bz, self.config.n_views * self.config.seq_len, -1, h, w)
        seg_tensor = seg.view(bz, self.config.n_views*self.seq_len, -1, h, w)
        sdc_tensor = sdc_tensor.view(bz, self.config.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat(
            [image_tensor, sdc_tensor, seg_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(
            bz, -1, self.n_embd)  # (B, an * T, C)

        # project velocity to n_embed
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1))  # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings +
                      velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.config.n_views + 2) * self.seq_len,
                   self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :,
                             :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)
        sdc_tensor_out = x[:, self.config.n_views*self.seq_len:self.config.n_views *
                           self.seq_len*2, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)

        return image_tensor_out, sdc_tensor_out


class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c


class SegEncoder(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, num_classes=512, in_channels=23, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        _tmp = self.features.conv1
        self.features.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
                                        kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c


class SdcEncoder(nn.Module):
    """
    Encoder network for Sdc input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=23):
        super().__init__()

        self._model = models.resnet18()
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
                                      kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)

    def forward(self, inputs):
        features = 0
        for sdc_data in inputs:
            sdc_feature = self._model(sdc_data)
            features += sdc_feature

        return features

# x13 model


class x13(nn.Module):
    def __init__(self, config, device):
        super(x13, self).__init__()
        self.config = config
        self.gpu_device = device
        self.seq_len = config.seq_len
        # ------------------------------------------------------------------------------------------------
        # RGB
        self.rgb_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(
            pretrained=True)  # efficientnet_b4
        self.RGB_encoder.classifier = nn.Sequential()
        self.RGB_encoder.avgpool = nn.Sequential()

        # SS
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[4][-1]+config.n_fmap_b3[3][-1], config.n_fmap_b3[3][-1]])
        self.conv2_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[3][-1]+config.n_fmap_b3[2][-1], config.n_fmap_b3[2][-1]])
        self.conv1_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[2][-1]+config.n_fmap_b3[1][-1], config.n_fmap_b3[1][-1]])
        self.conv0_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[1][-1]+config.n_fmap_b3[0][-1], config.n_fmap_b3[0][0]])
        self.final_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[0][0], config.n_class], final=True)
        self.conv0resnet_ss_f = ConvBlock(
            channel=[config.n_fmap_b3[1][-1], config.n_fmap_b3[1][-1] + config.n_fmap_b3[1][-1]])
        # ------------------------------------------------------------------------------------------------
        # red light and stop sign predictor
        self.tls_predictor = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[5][-1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

        self.tls_biasing = nn.Linear(2, config.n_fmap_b3[4][0])
        # ------------------------------------------------------------------------------------------------
        # SDC
        self.cover_area = config.coverage_area
        self.n_class = config.n_class
        self.h, self.w = config.input_resolution, config.input_resolution
        fx = 160
        self.x_matrix = torch.vstack(
            [torch.arange(-self.w/2, self.w/2)]*self.h) / fx
        self.x_matrix = self.x_matrix.to(device)
        # SC
        # self.SC_encoder = models.efficientnet_b1(pretrained=False)
        # self.SC_encoder.features[0][0] = nn.Conv2d(
        #     config.n_class, config.n_fmap_b1[0][0], kernel_size=3, stride=2, padding=1, bias=False)
        # self.SC_encoder.classifier = nn.Sequential()
        # self.SC_encoder.avgpool = nn.Sequential()
        # self.SC_encoder.apply(kaiming_init)
        # ------------------------------------------------------------------------------------------------
        # feature fusion
        self.necks_net = nn.Sequential(  # inputnya dari 2 bottleneck
            nn.Conv2d(config.n_fmap_b3[4][-1]+config.n_fmap_b1[4][-1],
                      config.n_fmap_b3[4][1], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][1], config.n_fmap_b3[4][0])
        )
        # ------------------------------------------------------------------------------------------------
        # wp predictor, input size 5 karena concat dari xy, next route xy, dan velocity
        self.gru = nn.GRUCell(input_size=5, hidden_size=config.n_fmap_b3[4][0])
        self.pred_dwp = nn.Linear(config.n_fmap_b3[4][0], 2)
        # PID Controller
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(
            K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        # ------------------------------------------------------------------------------------------------
        # controller
        # MLP Controller
        self.controller = nn.Sequential(
            nn.Linear(config.n_fmap_b3[4][0], config.n_fmap_b3[3][-1]),
            nn.Linear(config.n_fmap_b3[3][-1], 3),
            nn.ReLU()
        )
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        ).to(self.gpu_device)
        self.avgpool = nn.AdaptiveAvgPool2d(
            (self.config.vert_anchors, self.config.horz_anchors))

        self.image_encoder = ImageCNN(512, normalize=True)
        self.seg_encoder = SegEncoder(
            num_classes=512, in_channels=23, normalize=True)
        self.sdc_encoder = SdcEncoder(num_classes=512, in_channels=23)

        self.transformer1 = GPT_Seg(n_embd=64,
                                    n_head=config.n_head,
                                    block_exp=config.block_exp,
                                    n_layer=config.n_layer,
                                    vert_anchors=config.vert_anchors,
                                    horz_anchors=config.horz_anchors,
                                    seq_len=config.seq_len,
                                    embd_pdrop=config.embd_pdrop,
                                    attn_pdrop=config.attn_pdrop,
                                    resid_pdrop=config.resid_pdrop,
                                    config=config)
        self.transformer2 = GPT(n_embd=128,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=256,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=512,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.modelTransfuser = TransFuser(
            config, device).float().to(device)
        state_dict_transfuser = torch.load(
            os.path.join('cognitive_transfuser/log', 'best_model.pth'))
        self.modelTransfuser.load_state_dict(state_dict_transfuser)

    def forward(self, rgb_f, depth_f, next_route, velo_in):  # , gt_ss):
        # ------------------------------------------------------------------------------------------------
        bz = rgb_f.shape[0] // self.seq_len
        # print(next_route.shape)
        in_rgb = self.rgb_normalizer(rgb_f)  # [i]
        RGB_features0 = self.RGB_encoder.features[0](in_rgb)
        RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
        RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
        RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
        RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
        RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
        RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
        RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
        RGB_features8 = self.RGB_encoder.features[8](RGB_features7)
        # bagian upsampling
        ss_f_3 = self.conv3_ss_f(
            cat([self.up(RGB_features8), RGB_features5], dim=1))
        ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features3], dim=1))
        ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features2], dim=1))
        ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features1], dim=1))
        ss_f = self.final_ss_f(self.up(ss_f_0))
        # ------------------------------------------------------------------------------------------------
        # buat semantic cloud
        # ingat, depth juga sequence, ambil yang terakhir
        top_view_sc = self.gen_top_view_sc(depth_f, ss_f)
        # bagian downsampling
        # SC_features0 = self.SC_encoder.features[0](top_view_sc)
        # SC_features1 = self.SC_encoder.features[1](SC_features0)
        # SC_features2 = self.SC_encoder.features[2](SC_features1)
        # SC_features3 = self.SC_encoder.features[3](SC_features2)
        # SC_features4 = self.SC_encoder.features[4](SC_features3)
        # SC_features5 = self.SC_encoder.features[5](SC_features4)
        # SC_features6 = self.SC_encoder.features[6](SC_features5)
        # SC_features7 = self.SC_encoder.features[7](SC_features6)
        # SC_features8 = self.SC_encoder.features[8](SC_features7)

        # buat encoder waypoint cognitive transfuser(load dari weightnya transfuser)
        # image_features = self.modelTransfuser.image_encoder(in_rgb)
        image_features = self.modelTransfuser.encoder.image_encoder.features.conv1(
            in_rgb)
        image_features = self.modelTransfuser.encoder.image_encoder.features.bn1(
            image_features)
        image_features = self.modelTransfuser.encoder.image_encoder.features.relu(
            image_features)
        image_features = self.modelTransfuser.encoder.image_encoder.features.maxpool(
            image_features)
        # print("top view sc", top_view_sc.shape)
        #
        # sdc_features = self.modelTransfuser.encoder.lidar_encoder._model(top_view_sc)
        sdc_features = self.sdc_encoder._model.conv1(
            top_view_sc)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.bn1(
            sdc_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.relu(
            sdc_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.maxpool(
            sdc_features)
        seg_features = self.seg_encoder.features.conv1(ss_f)
        seg_features = self.seg_encoder.features.bn1(seg_features)
        seg_features = self.seg_encoder.features.relu(seg_features)
        seg_features = self.seg_encoder.features.maxpool(seg_features)
        # print("seg feature", seg_features.shape)
        # print("ss_f_1", ss_f_1.shape)
        ss_resnet_0 = self.conv0resnet_ss_f(ss_f_1)
        # print("ss_resnet_0", ss_resnet_0.shape)
        image_features = self.modelTransfuser.encoder.image_encoder.features.layer1(
            image_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.layer1(
            sdc_features)
        # seg_features = self.seg_encoder.features.layer1(ss_resnet_0)
        # print("image feature", image_features.shape)
        # print("sdc feature", sdc_features.shape)
        # fusion at (B, 64, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        sdc_embd_layer1 = self.avgpool(sdc_features)
        seg_embd_layer1 = self.avgpool(ss_resnet_0)

        image_features_layer1, sdc_features_layer1 = self.modelTransfuser.encoder.transformer1(
            image_embd_layer1, sdc_embd_layer1, seg_embd_layer1, velo_in)
        image_features_layer1 = F.interpolate(
            image_features_layer1, scale_factor=8, mode='bilinear')
        sdc_features_layer1 = F.interpolate(
            sdc_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        sdc_features = sdc_features + sdc_features_layer1

        image_features = self.modelTransfuser.encoder.image_encoder.features.layer2(
            image_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.layer2(
            sdc_features)
        # fusion at (B, 128, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)
        sdc_embd_layer2 = self.avgpool(sdc_features)
        image_features_layer2, sdc_features_layer2 = self.modelTransfuser.encoder.transformer2(
            image_embd_layer2, sdc_embd_layer2, velo_in)
        image_features_layer2 = F.interpolate(
            image_features_layer2, scale_factor=4, mode='bilinear')
        sdc_features_layer2 = F.interpolate(
            sdc_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        sdc_features = sdc_features + sdc_features_layer2

        image_features = self.modelTransfuser.encoder.image_encoder.features.layer3(
            image_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.layer3(
            sdc_features)
        # fusion at (B, 256, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        sdc_embd_layer3 = self.avgpool(sdc_features)
        image_features_layer3, sdc_features_layer3 = self.modelTransfuser.encoder.transformer3(
            image_embd_layer3, sdc_embd_layer3, velo_in)
        image_features_layer3 = F.interpolate(
            image_features_layer3, scale_factor=2, mode='bilinear')
        sdc_features_layer3 = F.interpolate(
            sdc_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3
        sdc_features = sdc_features + sdc_features_layer3

        image_features = self.modelTransfuser.encoder.image_encoder.features.layer4(
            image_features)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.layer4(
            sdc_features)
        # fusion at (B, 512, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)
        sdc_embd_layer4 = self.avgpool(sdc_features)
        image_features_layer4, sdc_features_layer4 = self.modelTransfuser.encoder.transformer4(
            image_embd_layer4, sdc_embd_layer4, velo_in)
        image_features = image_features + image_features_layer4
        sdc_features = sdc_features + sdc_features_layer4
        # print("image feature", image_features.shape)
        # print("sdc feature", sdc_features.shape)
        image_features = self.modelTransfuser.encoder.image_encoder.features.avgpool(
            image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(
            bz, self.seq_len, -1)
        sdc_features = self.modelTransfuser.encoder.lidar_encoder._model.avgpool(
            sdc_features)
        sdc_features = torch.flatten(sdc_features, 1)
        sdc_features = sdc_features.view(bz, self.seq_len, -1)

        fused_features = torch.cat([image_features, sdc_features], dim=1)
        fused_features = torch.sum(fused_features, dim=1)

        # ------------------------------------------------------------------------------------------------
        # red light and stop sign detection
        # print("fused_features", fused_features.shape)
        redl_stops = self.tls_predictor(fused_features)
        red_light = redl_stops[:, 0]
        stop_sign = redl_stops[:, 1]
        # tls_bias = self.tls_biasing(redl_stops)
        # ------------------------------------------------------------------------------------------------
        # waypoint prediction
        # get hidden state dari gabungan kedua bottleneck
        # RGB_features_sum+SC_features8 cat([RGB_features_sum, SC_features8], dim=1)
        # print("RGB_features8", RGB_features8.shape)
        # print("SC_features8", SC_features8.shape)
        hx = self.modelTransfuser.join(fused_features)
        # hx = self.necks_net(cat([RGB_features8, SC_features8], dim=1))
        # print(hx.shape)
        xy = torch.zeros(size=(hx.shape[0], 2)).float().to(self.gpu_device)
        # predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            # print("masuk")
            # print(xy.shape)
            # print(next_route.shape)
            # print( torch.reshape(
            #     velo_in, (velo_in.shape[0], 1)).shape)
            # print((velo_in.shape[0], 1).shape)
            # print((velo_in.shape[0], 1))
            # print(torch.reshape(
            #     velo_in, (velo_in.shape[0], 1)))
            # ins = torch.cat([xy, next_route, torch.reshape(
            #     velo_in, (velo_in.shape[0], 1))], dim=1)
            ins = xy + next_route
            # print("ins", ins.shape)
            hx = self.modelTransfuser.decoder(ins, hx)
            # print("hx", hx.shape)
            d_xy = self.modelTransfuser.output(hx)
            # d_xy = self.modelTransfuser.output(hx+tls_bias)
            # print("d_xy", d_xy.shape)
            xy = xy + d_xy
            # print("xy", xy.shape)
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1)
        # print("pred_wp", pred_wp.shape)
        # ------------------------------------------------------------------------------------------------
        # control decoder
        control_pred = self.controller(hx)
        steer = control_pred[:, 0] * 2 - 1.  # convert from [0,1] to [-1,1]
        throttle = control_pred[:, 1] * self.config.max_throttle
        brake = control_pred[:, 2]  # brake: hard 1.0 or no 0.0

        return ss_f, pred_wp, steer, throttle, brake, red_light, stop_sign, top_view_sc

    def swap_RGB2BGR(self, matrix):
        red = matrix[:, :, 0].copy()
        blue = matrix[:, :, 2].copy()
        matrix[:, :, 0] = blue
        matrix[:, :, 2] = red
        return matrix

    def gen_top_view_sc(self, depth, semseg):
        # proses awal
        depth_in = depth * 1000.0  # normalisasi ke 1 - 1000
        _, label_img = torch.max(semseg, dim=1)  # pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(
            self.h*self.w)] for n in range(depth.shape[0])])).to(self.gpu_device)

        # normalize ke frame
        cloud_data_x = torch.round(
            ((depth_in * self.x_matrix) + (self.cover_area/2)) * (self.w-1) / self.cover_area).ravel()
        cloud_data_z = torch.round(
            (depth_in * -(self.h-1) / self.cover_area) + (self.h-1)).ravel()

        # cari index interest
        bool_xz = torch.logical_and(torch.logical_and(
            cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        # hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
        idx_xz = bool_xz.nonzero().squeeze()

        # stack n x z cls dan plot
        coorx = torch.stack(
            [cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        # tensor harus long supaya bisa digunakan sebagai index
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long()
        # ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        top_view_sc = torch.zeros_like(semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2],
                    coor_clsn[3]] = 1.0  # format axis dari NCHW

        self.show_seg_sdc(semseg, top_view_sc)
        return top_view_sc

    def show_seg_sdc(self, seg, sdc):
        sdc = sdc.cpu().detach().numpy()
        seg = seg.cpu().detach().numpy()

        # buat array untuk nyimpan out gambar
        imgx2 = np.zeros((sdc.shape[2], sdc.shape[3], 3))
        imgx = np.zeros((seg.shape[2], seg.shape[3], 3))

        # print(sdc.shape)
        # ambil tensor output segmentationnya
        pred_sdc = sdc[0]
        pred_seg = seg[0]

        inx2 = np.argmax(pred_sdc, axis=0)
        inx = np.argmax(pred_seg, axis=0)
        # if inx[0].dtype != np.uint8:
        #     # Reshape inx to the desired shape (256, 384)
        #     inx_reshaped = inx.reshape(256, 384)

        #     # Save the reshaped array to a text file
        #     # Open the file in write mode ('w') to overwrite the file
        #     with open('inx_file.txt', 'w') as file:
        #         for row in inx_reshaped:
        #             file.write(' '.join(map(str, row)) + '\n')
        #     # Reshape inx to the desired shape (256, 384)
        #     inx2_reshaped = inx2.reshape(256, 384)

        #     # Save the reshaped array to a text file
        #     # Open the file in write mode ('w') to overwrite the file
        #     with open('inx2_file.txt', 'w') as file:
        #         for row in inx2_reshaped:
        #             file.write(' '.join(map(str, row)) + '\n')
        # entah kenapa deteksi road jadi warna hitam
        cmap = self.config.SEG_CLASSES['colors']
        for i in range(len(self.config.SEG_CLASSES['colors'])):
            cmap_id = self.config.SEG_CLASSES['colors'].index(cmap[i])
            # print(cmap_id, "detected")
            if i+1 < self.config.n_class:
                imgx2[np.where(inx2 == cmap_id)] = cmap[i]
                imgx[np.where(inx == cmap_id)] = cmap[i]

        # GANTI ORDER BGR KE RGB, SWAP!
        imgx2 = self.swap_RGB2BGR(imgx2)
        imgx = self.swap_RGB2BGR(imgx)
        cv2.imshow("seg bro", imgx.astype(np.uint8))
        cv2.imshow("bev bro", imgx2.astype(np.uint8))
        cv2.waitKey(1)

    def mlp_pid_control(self, waypoints, velocity, mlp_steer, mlp_throttle, mlp_brake, redl, stops, ctrl_opt="one_of"):
        assert (waypoints.size(0) == 1)
        waypoints = waypoints[0].data.cpu().numpy()
        red_light = True if redl.data.cpu().numpy() > 0.5 else False
        stop_sign = True if stops.data.cpu().numpy() > 0.5 else False

        waypoints[:, 1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        pid_steer = self.turn_controller.step(angle)
        pid_steer = np.clip(pid_steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(delta)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)
        pid_brake = 0.0

        # final decision
        if ctrl_opt == "one_of":
            # opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
            steer = np.clip(
                self.config.cw_pid[0]*pid_steer + self.config.cw_mlp[0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(
                self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle >= self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                steer = pid_steer
                throttle = pid_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle >= self.config.min_act_thrt):
                pid_brake = 1.0
                steer = mlp_steer
                throttle = mlp_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(
                    self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
        elif ctrl_opt == "both_must":
            # opsi 2: vehicle jalan jika dan hanya jika kedua controller aktif. jika salah satu saja non aktif, maka vehicle berhenti
            steer = np.clip(
                self.config.cw_pid[0]*pid_steer + self.config.cw_mlp[0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(
                self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle < self.config.min_act_thrt) or (mlp_throttle < self.config.min_act_thrt):
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(
                    self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
        elif ctrl_opt == "pid_only":
            # opsi 3: PID only
            steer = pid_steer
            throttle = pid_throttle
            brake = 0.0
            # MLP full off
            mlp_steer = 0.0
            mlp_throttle = 0.0
            mlp_brake = 0.0
            if pid_throttle < self.config.min_act_thrt:
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = pid_brake
        elif ctrl_opt == "mlp_only":
            # opsi 4: MLP only
            steer = mlp_steer
            throttle = mlp_throttle
            brake = 0.0
            # PID full off
            pid_steer = 0.0
            pid_throttle = 0.0
            pid_brake = 0.0
            if mlp_throttle < self.config.min_act_thrt:
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                brake = mlp_brake
        else:
            sys.exit("ERROR, FALSE CONTROL OPTION")

        metadata = {
            'control_option': ctrl_opt,
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'red_light': float(red_light),
            'stop_sign': float(stop_sign),
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1]), float(self.config.cw_pid[2])],
            'pid_steer': float(pid_steer),
            'pid_throttle': float(pid_throttle),
            'pid_brake': float(pid_brake),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1]), float(self.config.cw_mlp[2])],
            'mlp_steer': float(mlp_steer),
            'mlp_throttle': float(mlp_throttle),
            'mlp_brake': float(mlp_brake),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
            'car_pos': None,  # akan direplace di fungsi agent
            'next_point': None,  # akan direplace di fungsi agent
        }
        return steer, throttle, brake, metadata
