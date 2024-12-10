""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import copy
from functools import partial
from itertools import repeat
from functools import reduce
from operator import mul

from sklearn import manifold, datasets
import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
#from torch._six import container_abcs
import collections.abc as container_abcs
from config import cfg

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

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
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self, in_features, hidden_features=64, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act2 = act_layer()
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x_0 = x
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = x + x_0
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''if isinstance(x, tuple):
            x = x[0]'''
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Multi_Heads(nn.Module):
    def __init__(self, inplanes=768, planes=64, heads=16):
        super(Multi_Heads, self ).__init__() 

        #self.num_list = num_list
        #self.down_sample = nn.ModuleDict()
        self.feat_dim = inplanes
        self.dim = planes
        #self.embedding_dim = int(inplanes/4)
        self.heads = heads
        #self.max_pool = max_pool
        #self.gmp = nn.AdaptiveMaxPool2d(1)  
        #self.gap = nn.AdaptiveAvgPool2d(1)  
        #self.mask_module = Mask_Module(inplanes=self.feat_dim, heads=self.heads, with_mask=with_mask, max_pool=max_pool)
        self.classifier_dict = nn.ModuleDict()
        for i in range(self.heads):
            self.classifier_dict[f'step:{i}'] = nn.Linear(self.feat_dim, self.dim, bias=False)
            #self.classifier_dict[f'step:{i}'].bias.requires_grad_(False)
            #self.classifier_dict[f'step:{i}'][0].apply(weights_init_kaiming)
            #self.classifier_dict[f'step:{i}'][1].apply(weights_init_classifier)
            nn.init.kaiming_normal_(self.classifier_dict[f'step:{i}'].weight, a=0, mode='fan_out')
    def forward(self, input, source=None):
    
        #N, C, H, W = input.shape
        #feat_list = self.mask_module(input)
        cls_score_list = []
        for i in range(self.heads):
            #cls_score = self.classifier_dict[f'step:{c_s}'](feat)
            #feat_ = feat_list[i]
            cls_score = self.classifier_dict[f'step:{i}'](input)
            #print(cls_score.size())
            if cfg.MODEL.INS_PMT_FUSE_T != 0.0:
                t = cfg.MODEL.INS_PMT_FUSE_T
                cls_score = F.softmax(cls_score/t, dim=2)
            else:
                cls_score = F.softmax(cls_score, dim=2)
            #print(cls_score.size())
            cls_score_list.append(cls_score)
        output_score = torch.cat(cls_score_list, dim=1) 
        return output_score

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0, modal=2,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0,
                 num_tokens=16, use_prompt=True, num_instance_prompt_tokens=16, size_instance_prompt_bank=64, use_instance_prompt=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)



        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        
        if cfg.MODEL.USE_ME:
            self.modal_embed = nn.Parameter(torch.zeros(modal, 1, embed_dim))
            trunc_normal_(self.modal_embed, std=.02)
            print('modality number is : {}'.format(modal))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        
        self.apply(self._init_weights)

        # initialize modal specific prompts
        self.num_tokens = num_tokens
        self.use_prompt = use_prompt
        self.depth = depth      
        #if cfg.MODEL.USE_PROMPT:
        self._init_prompt(patch_size, self.num_tokens, embed_dim, depth)
        
        # print('patch_size:%d token_num:%d dimension:%d depth:%d' % (patch_size, self.patch_embed.num_x * self.patch_embed.num_y, embed_dim, depth))
        # print('num_tokens:%d' % (self.num_tokens))
        print('num_tokens:%d' % (self.num_tokens), 'use_prompt', self.use_prompt)

        # initialize instance specific prompts
        self.num_instance_prompt_tokens = num_instance_prompt_tokens
        self.use_instance_prompt = use_instance_prompt
        self.size_instance_prompt_bank = size_instance_prompt_bank
        self.use_instance_prompt_generator = cfg.MODEL.USE_INS_PROMPT_GEN
        self.use_naive_prompt = cfg.MODEL.NAIVE_PROMPT
        
        #if cfg.MODEL.USE_INS_PROMPT_GEN:
        self._init_instance_generator(embed_dim, depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        
        #self.prompt_adapter = Adapter(in_features=embed_dim)
        self.prompt_adapters = nn.ModuleList([Adapter(in_features=embed_dim) for i in range(11)])
        #elif cfg.MODEL.USE_INS_PROMPT:
            #self.depth = depth
        if cfg.MODEL.USE_INS_PROMPT and (not cfg.MODEL.USE_INS_PROMPT_GEN):
            self._init_instance_prompt(patch_size, size_instance_prompt_bank, self.num_instance_prompt_tokens, embed_dim, depth)
            print("hello")
            # print('patch_size:%d token_num:%d dimension:%d depth:%d' % (patch_size, self.patch_embed.num_x * self.patch_embed.num_y, embed_dim, depth))
            # print('num_tokens:%d' % (self.num_tokens))

        print('num_instance_prompt_tokens:%d' % (self.num_instance_prompt_tokens), 'use_instance_prompt', self.use_instance_prompt)
        print('use_instance_prompt_generator', cfg.MODEL.USE_INS_PROMPT_GEN)

        self.prompt_embed = nn.Parameter(torch.zeros(1, self.num_instance_prompt_tokens, self.embed_dim))
        trunc_normal_(self.prompt_embed, std=.02)
        # if cfg.MODEL.BUG_REFINE:
        #     self.ins_prompt_embed = nn.Parameter(torch.zeros(depth, 1, self.num_instance_prompt_tokens, self.embed_dim))
        #     trunc_normal_(self.prompt_embed, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id, modality_flag):
        B = x.shape[0]
        # print(x.shape)
        x = self.patch_embed(x)
        # print('1')
        # print(x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)
        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            #x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
            x = x + self.pos_embed
            x = x + self.sie_xishu * self.sie_embed[camera_id]
            # print(self.sie_embed.shape)
            #print(camera_id.shape)
            # print(self.sie_embed[camera_id].shape)
            # print(self.pos_embed.shape)
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed
        # print(x.shape)
        if cfg.MODEL.USE_ME:
            x = x + self.sie_xishu * self.modal_embed[modality_flag]
        # print(x.shape)
        x = self.pos_drop(x)
        
        # print(x.shape)
        if self.use_prompt: 
            if self.use_instance_prompt: # adopt instance & modality specific prompt tuning
                if cfg.TEST.TSNE:
                    x,X_tsne_1,X_tsne_2 = self.forward_deep_all_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                    return x,X_tsne_1,X_tsne_2
                else:
                    x = self.forward_deep_all_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                    return x
            else: # adopt modality specific prompt tuning only
                if cfg.TEST.TSNE:
                    x,X_tsne_1,X_tsne_2 = self.forward_deep_modality_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                    return x,X_tsne_1,X_tsne_2
                else:
                    x = self.forward_deep_modality_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                    return x
        elif self.use_instance_prompt: # adopt instance specific prompt tuning only
            if cfg.TEST.TSNE:
                x,X_tsne_1,X_tsne_2 = self.forward_deep_instance_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                return x,X_tsne_1,X_tsne_2
            elif cfg.MODEL.IPIL:
                x, ins_prompts = self.forward_deep_instance_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                return x, ins_prompts
            else:
                x = self.forward_deep_instance_prompt(x, self.patch_embed.num_x, self.patch_embed.num_y, modality_flag)
                return x
        # print(x.shape)
        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            
            if cfg.TEST.TSNE:
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                X_tsne_2 = tsne.fit_transform(x[:, :1, :].reshape(x.size(0), 1*768).cpu())
                print('tsnetsne')
                return x,None,X_tsne_2
            return x

        else:
            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)

            return x[:, 0]

    def forward(self, x, cam_label=None, view_label=None, modality_flag=None):
        x = self.forward_features(x, cam_label, view_label, modality_flag)
        return x
    
    def forward_deep_modality_prompt(self, inputs, H, W, modality_flag=0):
        B = inputs.shape[0]
        modality_flag = modality_flag.cpu().data.numpy().tolist()
        #print(type(modality_flag))
        #print(modality_flag)
        naive_prompt_flag_m = [0]*len(modality_flag)
        # modality_flag[:4] = [0,0,0,0]
        for i in range(self.depth):
            if i == 0:
                if self.use_naive_prompt:
                    prompts = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[naive_prompt_flag_m]))
                else:
                    prompts = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[modality_flag]))
                # prompts = self.prompt_embeddings[modality_flag])
                # print((prompts[0,0]==prompts[3,0]).all())
                hidden_states = torch.cat((inputs[:, :1, :],
                                prompts,
                                inputs[:, 1:, :]
                                ), dim=1)
                hidden_states = self.blocks[i](hidden_states)
            elif i == (self.depth-2) and self.local_feature:
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    hidden_states[:, -(H*W):, :]
                ), dim=1)
                return hidden_states
                
                
            elif i <= self.deep_prompt_embeddings.shape[1]:
                if self.use_naive_prompt:
                    deep_prompts = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[naive_prompt_flag_m][:,i-1]))
                else:
                    deep_prompts = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[modality_flag][:,i-1]))
                # deep_prompt_emb = self.deep_prompt_embeddings[modality_flag][:,i-1]
                # print((deep_prompt_emb[0,0]==deep_prompt_emb[3,0]).all)
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    deep_prompts,
                    hidden_states[:, (1+self.num_tokens):, :]
                ), dim=1)

                hidden_states = self.blocks[i](hidden_states)


        encoded = self.norm(hidden_states)

        return encoded[:, 0]

    
    def forward_deep_instance_prompt(self, inputs, H, W, modality_flag=0):
        B = inputs.shape[0]
        naive_prompt_flag_i = [1]*len(modality_flag)
        ins_prompts_list = []
        for i in range(self.depth):
            if i == 0:
                if self.use_instance_prompt_generator:
                    if self.use_naive_prompt:
                        prompt_instance = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[naive_prompt_flag_i]))
                    else:
                        # prompt_embeddings = self.prompt_embed.expand(B, -1, -1)
                        prompt_embeddings = self.prompt_embed.repeat(B, 1, 1)
                        em = torch.cat((
                            prompt_embeddings,
                            inputs[:, 1:, :]
                        ), dim=1)
                        em = self.instance_prompt_generator(em)
                        
                        prompt_instance = em[:, :self.num_instance_prompt_tokens, :]
                    prompts = prompt_instance
                    if cfg.MODEL.IPIL_SIMPLE:
                        ins_prompts_list.append(prompt_instance.mean(dim=1))
                    else:
                        ins_prompts_list.append(prompt_instance)
                    #print(prompt_instance)
                else:
                    fusion_score_list = self.instance_prompt_fusion_head(inputs[:, 1:, :].mean(dim=1, keepdim=True))
                    prompt_instance = torch.matmul(fusion_score_list, self.instance_prompt_embeddings)
                    prompt_instance = self.prompt_dropout_1(self.instance_prompt_proj(prompt_instance))
                    prompts = prompt_instance
                    if cfg.MODEL.IPIL_SIMPLE:
                        ins_prompts_list.append(prompt_instance.mean(dim=1))
                    else:
                        ins_prompts_list.append(prompt_instance)
                    
                if cfg.TEST.TSNE:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    #X_tsne_1 = tsne.fit_transform(prompt_modality.reshape(prompt_modality.size(0), 16*768).cpu())
                    X_tsne_2 = tsne.fit_transform(prompt_instance.reshape(prompt_instance.size(0), 16*768).cpu())
                hidden_states = torch.cat((inputs[:, :1, :],
                                prompts,
                                inputs[:, 1:, :]
                                ), dim=1)
                hidden_states = self.blocks[i](hidden_states)

            elif i == (self.depth-2) and self.local_feature:
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    hidden_states[:, -(H*W):, :]
                ), dim=1)
                if cfg.TEST.TSNE:
                    return hidden_states,None,X_tsne_2
                elif cfg.MODEL.IPIL and cfg.MODEL.INS_PMT_ONLYLAYERONE:
                    ins_prompts_list = [hidden_states[:,1:self.num_instance_prompt_tokens+1,:]]
                    #print(hidden_states[:,1:self.num_instance_prompt_tokens+1,:].size(1))
                    return hidden_states, ins_prompts_list
                elif cfg.MODEL.IPIL:
                    #print(len(ins_prompts_list))
                    return hidden_states, ins_prompts_list
                else:
                    return hidden_states
            
            #elif i <= self.deep_prompt_embeddings.shape[1]:
            elif i <= self.depth:
                if not cfg.MODEL.INS_PMT_ONLYLAYERONE:
                    if self.use_instance_prompt_generator:
                        if self.use_naive_prompt:
                            deep_prompt_instance = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[naive_prompt_flag_i][:,i-1]))
                        else:
                            # deep_prompt_embeddings = self.prompt_embed.expand(B, -1, -1)
                            deep_prompt_embeddings = self.prompt_embed.repeat(B, 1, 1)
                            deep_em = torch.cat((
                                deep_prompt_embeddings,
                                hidden_states[:, (1+self.num_instance_prompt_tokens):, :]
                            ), dim=1)
                            #deep_em = self.deep_instance_prompt_generators[i-1](deep_em)
                            deep_em = self.instance_prompt_generator(deep_em)
                            deep_prompt_instance = deep_em[:, :self.num_instance_prompt_tokens, :]
                        deep_prompts = deep_prompt_instance
                        if cfg.MODEL.IPIL_SIMPLE:
                            ins_prompts_list.append(deep_prompt_instance.mean(dim=1))
                        else:
                            ins_prompts_list.append(deep_prompt_instance)
                        
                    else:
                        fusion_score_list = self.deep_instance_prompt_fusion_heads[i-1](hidden_states[:, :1, :])
                        # debug and show
                        '''for aaa in range(fusion_score_list.size()[0]):
                            print(fusion_score_list[aaa].mean(dim=0))
                        print('this one done')'''
                        deep_prompt_instance = torch.matmul(fusion_score_list, self.deep_instance_prompt_embeddings[:,i-1])
                        deep_prompt_instance = self.prompt_dropout_1(self.instance_prompt_proj(deep_prompt_instance))
                        deep_prompts = deep_prompt_instance
                        if cfg.MODEL.IPIL_SIMPLE:
                            ins_prompts_list.append(deep_prompt_instance.mean(dim=1))
                        else:
                            ins_prompts_list.append(deep_prompt_instance)
                        
                    if cfg.TEST.TSNE and i == (self.depth-3):
                        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                        #X_tsne_1 = tsne.fit_transform(prompt_modality.reshape(prompt_modality.size(0), 16*768).cpu())
                        X_tsne_2 = tsne.fit_transform(deep_prompt_instance.reshape(deep_prompt_instance.size(0), 16*768).cpu())
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompts,
                        hidden_states[:, (1+self.num_instance_prompt_tokens):, :]
                    ), dim=1)
                hidden_states = self.blocks[i](hidden_states)


        encoded = self.norm(hidden_states)

        return encoded[:, 0]
    

    def forward_deep_all_prompt(self, inputs, H, W, modality_flag=0):
        B = inputs.shape[0]
        modality_flag = modality_flag.cpu().data.numpy().tolist()
        naive_prompt_flag_m = [0]*len(modality_flag)
        naive_prompt_flag_i = [1]*len(modality_flag)
        ins_prompts_list = []
        for i in range(self.depth):
            if i == 0:
                if self.use_naive_prompt:
                    prompt_modality = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[naive_prompt_flag_m]))
                else:
                    prompt_modality = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[modality_flag]))
                if self.use_instance_prompt_generator:
                    if self.use_naive_prompt:
                        prompt_instance = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings[naive_prompt_flag_i]))
                    else:
                        # prompt_embeddings = self.prompt_embed.expand(B, -1, -1)
                        if cfg.MODEL.BUG_REFINE:
                            prompt_embeddings = self.prompt_embed.repeat(B, 1, 1)
                        else:
                            prompt_embeddings = nn.Parameter(torch.zeros(B, self.num_instance_prompt_tokens, self.embed_dim)).cuda()
                        em = torch.cat((
                            prompt_embeddings,
                            inputs[:, 1:, :]
                        ), dim=1)
                        em = self.instance_prompt_generator(em)
                        prompt_instance = em[:, :self.num_instance_prompt_tokens, :]
                    prompts = torch.cat((prompt_modality, prompt_instance),dim=1)
                    if cfg.MODEL.ADAPTER:
                        prompts = self.prompt_adapters[i](prompts)
                    if cfg.MODEL.IPIL_SIMPLE:
                        ins_prompts_list.append(prompt_instance.mean(dim=1))
                    else:
                        ins_prompts_list.append(prompt_instance)
                    #print(prompt_instance)
                else:
                    fusion_score_list = self.instance_prompt_fusion_head(inputs[:, 1:, :].mean(dim=1, keepdim=True))
                    prompt_instance = torch.matmul(fusion_score_list, self.instance_prompt_embeddings)
                    prompt_instance = self.prompt_dropout_1(self.instance_prompt_proj(prompt_instance))
                    prompts = torch.cat((prompt_modality, prompt_instance),dim=1)
                    if cfg.MODEL.IPIL_SIMPLE:
                        ins_prompts_list.append(prompt_instance.mean(dim=1))
                    else:
                        ins_prompts_list.append(prompt_instance)
                #if cfg.TEST.TSNE:
                    #tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    #X_tsne_1 = tsne.fit_transform(prompt_modality.reshape(prompt_modality.size(0), 16*768).cpu())
                    #X_tsne_2 = tsne.fit_transform(prompt_instance.reshape(prompt_instance.size(0), 16*768).cpu())
                hidden_states = torch.cat((inputs[:, :1, :],
                                prompts,
                                inputs[:, 1:, :]
                                ), dim=1)
                hidden_states = self.blocks[i](hidden_states)

            elif i == (self.depth-2) and self.local_feature:
                hidden_states = torch.cat((
                    hidden_states[:, :1, :],
                    hidden_states[:, -(H*W):, :]
                ), dim=1)
                if cfg.TEST.TSNE:
                    return hidden_states,X_tsne_1,X_tsne_2
                elif cfg.MODEL.IPIL and cfg.MODEL.INS_PMT_ONLYLAYERONE:
                    ins_prompts_list = [hidden_states[:,1+self.num_tokens:self.num_instance_prompt_tokens+self.num_tokens+1,:]]
                    #print(hidden_states[:,1:self.num_instance_prompt_tokens+1,:].size(1))
                    return hidden_states, ins_prompts_list
                elif cfg.MODEL.IPIL:
                    #print(len(ins_prompts_list))
                    return hidden_states, ins_prompts_list
                else:
                    return hidden_states
            
            #elif i <= self.deep_prompt_embeddings.shape[1]:
            elif i <= self.depth:
                if self.use_naive_prompt:
                    deep_prompt_modality = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[naive_prompt_flag_m][:,i-1]))
                else:
                    deep_prompt_modality = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[modality_flag][:,i-1]))
                if not cfg.MODEL.INS_PMT_ONLYLAYERONE:
                    if self.use_instance_prompt_generator:
                        if self.use_naive_prompt:
                            deep_prompt_instance = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[naive_prompt_flag_i][:,i-1]))
                        else:
                            # deep_prompt_embeddings = self.prompt_embed.expand(B, -1, -1)
                            if cfg.MODEL.BUG_REFINE:
                                deep_prompt_embeddings = self.prompt_embed.repeat(B, 1, 1)
                            else:
                                deep_prompt_embeddings = nn.Parameter(torch.zeros(B, self.num_instance_prompt_tokens, self.embed_dim)).cuda()
                            deep_em = torch.cat((
                                deep_prompt_embeddings,
                                hidden_states[:, (1+self.num_instance_prompt_tokens+self.num_tokens):, :]
                            ), dim=1)
                            #deep_em = self.deep_instance_prompt_generators[i-1](deep_em)
                            deep_em = self.instance_prompt_generator(deep_em)
                            deep_prompt_instance = deep_em[:, :self.num_instance_prompt_tokens, :]
                        '''if hidden_states[:, (1+self.num_instance_prompt_tokens+self.num_tokens):, :].size(1) == hidden_states[:, -(H*W):, :].size(1):
                            print('yesyesyesyes')'''
                        deep_prompts = torch.cat((deep_prompt_modality, deep_prompt_instance),dim=1)
                        if cfg.MODEL.ADAPTER:
                            deep_prompts = self.prompt_adapters[i](deep_prompts)
                        if cfg.MODEL.IPIL_SIMPLE:
                            ins_prompts_list.append(deep_prompt_instance.mean(dim=1))
                        else:
                            ins_prompts_list.append(deep_prompt_instance)
                        
                    else:
                        fusion_score_list = self.deep_instance_prompt_fusion_heads[i-1](hidden_states[:, :1, :])
                        # debug and show
                        '''for aaa in range(fusion_score_list.size()[0]):
                            print(fusion_score_list[aaa].mean(dim=0))
                        print('this one done')'''
                        deep_prompt_instance = torch.matmul(fusion_score_list, self.deep_instance_prompt_embeddings[:,i-1])
                        deep_prompt_instance = self.prompt_dropout_1(self.instance_prompt_proj(deep_prompt_instance))
                        deep_prompts = torch.cat((deep_prompt_modality, deep_prompt_instance),dim=1)
                        if cfg.MODEL.IPIL_SIMPLE:
                            ins_prompts_list.append(deep_prompt_instance.mean(dim=1))
                        else:
                            ins_prompts_list.append(deep_prompt_instance)
                        
                    if cfg.TEST.TSNE and i == (self.depth-3):
                        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                        # X_tsne_1 = tsne.fit_transform(deep_prompt_modality.reshape(prompt_modality.size(0), 16*768).cpu())
                        # X_tsne_2 = tsne.fit_transform(deep_prompt_instance.reshape(deep_prompt_instance.size(0), 16*768).cpu())
                        X_tsne_1 = tsne.fit_transform(deep_prompt_modality.reshape(prompt_modality.size(0), 16*768).cpu())
                        X_tsne_2 = tsne.fit_transform(hidden_states[:, :1, :].reshape(hidden_states.size(0), 1*768).cpu())
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompts,
                        hidden_states[:, -(H*W):, :]
                    ), dim=1)
                hidden_states = self.blocks[i](hidden_states)


        encoded = self.norm(hidden_states)

        return encoded[:, 0]

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(torch.zeros(2, num_tokens, prompt_dim)) # (modality, number of prompt vectors, channel dimension) 
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT, val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT)
            print(val)

            if total_d_layer > 0:  # depth-wise prompts
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(2, total_d_layer, num_tokens, prompt_dim)) # (modality, depth, number of prompt vectors, channel dimension) 
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT, val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
            
            
            
            # self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

    def _init_instance_prompt(self, patch, size_instance_prompt_bank, num_instance_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  
        if total_d_layer >= 0:
            self.instance_prompt_embeddings = nn.Parameter(torch.zeros(size_instance_prompt_bank, prompt_dim)) # (modality, number of prompt vectors, channel dimension) 
            # xavier_uniform initialization
            nn.init.uniform_(self.instance_prompt_embeddings.data, -val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT, val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT)
            self.instance_prompt_fusion_head = Multi_Heads(inplanes=prompt_dim, planes=size_instance_prompt_bank, heads=num_instance_tokens).cuda()

            if total_d_layer > 0:  # depth-wise prompts
                self.deep_instance_prompt_embeddings = nn.Parameter(torch.zeros(size_instance_prompt_bank, total_d_layer, prompt_dim)) # (modality, depth, number of prompt vectors, channel dimension) 
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_instance_prompt_embeddings, -val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT, val*cfg.MODEL.PROMPT_SCALE+cfg.MODEL.PROMPT_SHIFT)
                # instance_prompt_fusion_heads
                self.deep_instance_prompt_fusion_heads = nn.ModuleList()
                for i in range(total_d_layer):
                    deep_instance_prompt_fusion_head = Multi_Heads(inplanes=prompt_dim, planes=size_instance_prompt_bank, heads=num_instance_tokens).cuda()
                    self.deep_instance_prompt_fusion_heads.append(deep_instance_prompt_fusion_head)
                #self.instance_prompt_fusion_heads = self.instance_prompt_fusion_heads

            self.instance_prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.instance_prompt_proj.weight, a=0, mode='fan_out')
            #self.prompt_norm_1 = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout_1 = Dropout(0.1) # 现有的实验里面都是没有定义这个层的，而是直接复用了前面_init_prompt里面定义的那个prompt_dropout 不知道会不会有什么影响
            
    def _init_instance_generator(self, prompt_dim, total_d_layer, dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop, attn_drop, drop_path, norm_layer):
        if total_d_layer >= 0:
            block = self.blocks[0]
            layer_norm = self.norm
            self.instance_prompt_generator = copy.deepcopy(block)
            '''self.instance_prompt_generator = Block(
                dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop, attn_drop, drop_path, norm_layer)'''
            #nn.init.kaiming_normal_(self.instance_prompt_generator, a=0, mode='fan_out')
            '''self.deep_instance_prompt_generators = nn.ModuleList()
            if total_d_layer > 0:
                for i in range(total_d_layer):
                    deep_instance_prompt_generator = TransformerEncoderLayer(d_model=prompt_dim, nhead=8).cuda()
                    #TransformerEncoder(TransformerEncoderLayer(d_model=prompt_dim, nhead=8),num_layers=1)
                    #nn.init.kaiming_normal_(deep_instance_prompt_generator, a=0, mode='fan_out')
                    self.deep_instance_prompt_generators.append(deep_instance_prompt_generator)'''

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
