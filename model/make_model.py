import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from config import cfg
from .backbones.vit_pytorch import trunc_normal_

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.specific_bn = cfg.MODEL.SPECIFIC_BN

        self.zero_token = nn.Parameter(torch.zeros(1, 1, self.in_planes))
        trunc_normal_(self.zero_token, std=.02)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, \
                                                        drop_path_rate=cfg.MODEL.DROP_PATH, num_tokens=cfg.MODEL.NUM_TOKEN, use_prompt=cfg.MODEL.USE_PROMPT, \
                                                            num_instance_prompt_tokens=cfg.MODEL.NUM_INS_PMT_TOKEN, size_instance_prompt_bank=cfg.MODEL.SIZE_INS_PMT_BANK, use_instance_prompt=cfg.MODEL.USE_INS_PROMPT)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        if cfg.MODEL.IPIL and not cfg.MODEL.IPIL_SIMPLE:
            self.b3 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)
            self.classifiers_ins_prompts = nn.ModuleList()
            for i in range(10):
                classifier_ins_prompts = nn.Linear(self.in_planes, self.num_classes, bias=False)
                classifier_ins_prompts.apply(weights_init_classifier)
                self.classifiers_ins_prompts.append(classifier_ins_prompts)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        if self.specific_bn:
            self.bottleneck_sub = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub.bias.requires_grad_(False)
            self.bottleneck_sub.apply(weights_init_kaiming)
            self.bottleneck_sub_1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_1.bias.requires_grad_(False)
            self.bottleneck_sub_1.apply(weights_init_kaiming)
            self.bottleneck_sub_2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_2.bias.requires_grad_(False)
            self.bottleneck_sub_2.apply(weights_init_kaiming)
            self.bottleneck_sub_3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_3.bias.requires_grad_(False)
            self.bottleneck_sub_3.apply(weights_init_kaiming)
            self.bottleneck_sub_4 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_sub_4.bias.requires_grad_(False)
            self.bottleneck_sub_4.apply(weights_init_kaiming)
        
        self.bottlenecks_ins_prompts = nn.ModuleList()
        for i in range(10):
            bottleneck_ins_prompts = nn.BatchNorm1d(self.in_planes)
            bottleneck_ins_prompts.apply(weights_init_kaiming)
            self.bottlenecks_ins_prompts.append(bottleneck_ins_prompts)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    #def forward(self, x, label=None, cam_label= torch.tensor([0]).cuda(), view_label=None, modality_flag=torch.tensor([0]).cuda()):  # label is unused if self.cos_layer == 'no'
    def forward(self, x, label=None, cam_label= None, view_label=None, modality_flag=None):  # label is unused if self.cos_layer == 'no'
        #cam_label= torch.tensor([0]).cuda()
        #modality_flag=torch.tensor([1]).cuda()
        if cfg.TEST.TSNE:
            features,X_tsne_1,X_tsne_2 = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        elif cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:
            features, ins_prompts = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        else:
            features = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        
        '''if cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:
            features, ins_prompts = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        else:
            features = self.base(x, cam_label=cam_label, view_label=view_label, modality_flag=modality_flag)
        '''
        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        
        if self.specific_bn:
            global_feat_ir = global_feat[modality_flag==0]
            global_feat_rgb = global_feat[modality_flag==1]
            local_feat_1_ir = local_feat_1[modality_flag==0]
            local_feat_1_rgb = local_feat_1[modality_flag==1]
            local_feat_2_ir = local_feat_2[modality_flag==0]
            local_feat_2_rgb = local_feat_2[modality_flag==1]
            local_feat_3_ir = local_feat_3[modality_flag==0]
            local_feat_3_rgb = local_feat_3[modality_flag==1]
            local_feat_4_ir = local_feat_4[modality_flag==0]
            local_feat_4_rgb = local_feat_4[modality_flag==1]

            feat_ir = self.bottleneck_sub(global_feat_ir)
            feat_rgb = self.bottleneck(global_feat_rgb)
            local_feat_1_ir_bn = self.bottleneck_sub_1(local_feat_1_ir)
            local_feat_2_ir_bn = self.bottleneck_sub_2(local_feat_2_ir)
            local_feat_3_ir_bn = self.bottleneck_sub_3(local_feat_3_ir)
            local_feat_4_ir_bn = self.bottleneck_sub_4(local_feat_4_ir)
            local_feat_1_rgb_bn = self.bottleneck_1(local_feat_1_rgb)
            local_feat_2_rgb_bn = self.bottleneck_2(local_feat_2_rgb)
            local_feat_3_rgb_bn = self.bottleneck_3(local_feat_3_rgb)
            local_feat_4_rgb_bn = self.bottleneck_4(local_feat_4_rgb)

            feat = torch.cat((feat_ir,feat_rgb),dim=0)
            local_feat_1_bn = torch.cat((local_feat_1_ir_bn,local_feat_1_rgb_bn),dim=0)
            local_feat_2_bn = torch.cat((local_feat_2_ir_bn,local_feat_2_rgb_bn),dim=0)
            local_feat_3_bn = torch.cat((local_feat_3_ir_bn,local_feat_3_rgb_bn),dim=0)
            local_feat_4_bn = torch.cat((local_feat_4_ir_bn,local_feat_4_rgb_bn),dim=0)
            
        else:
            feat = self.bottleneck(global_feat)
            local_feat_1_bn = self.bottleneck_1(local_feat_1)
            local_feat_2_bn = self.bottleneck_2(local_feat_2)
            local_feat_3_bn = self.bottleneck_3(local_feat_3)
            local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.specific_bn:
                global_feat = torch.cat((global_feat_ir,global_feat_rgb),dim=0)
                local_feat_1 = torch.cat((local_feat_1_ir,local_feat_1_rgb),dim=0)
                local_feat_2 = torch.cat((local_feat_2_ir,local_feat_2_rgb),dim=0)
                local_feat_3 = torch.cat((local_feat_3_ir,local_feat_3_rgb),dim=0)
                local_feat_4 = torch.cat((local_feat_4_ir,local_feat_4_rgb),dim=0)
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            if cfg.MODEL.USE_INS_PROMPT and cfg.MODEL.IPIL:
                #ins_prompts_feats = [self.bottleneck_ins_prompts(ins_prompt) for ins_prompt in ins_prompts]
                #ins_prompts_cls_score = [self.classifier_ins_prompts(ins_prompts_feat) for ins_prompts_feat in ins_prompts_feats]
                #zero_token = self.zero_token.expand(features.size(0), -1, -1)
                #b_ins_prompts_feats = [self.b3(torch.cat((zero_token, ins_prompt), dim=1)) for ins_prompt in ins_prompts]
                #ins_prompts_feats = [b_ins_prompts_feat[:, 0, :] for b_ins_prompts_feat in b_ins_prompts_feats]
                if cfg.MODEL.IPIL_SIMPLE:
                    for i in range(10):
                        ins_prompts[i] = self.bottlenecks_ins_prompts[i](ins_prompts[i])
                else:
                    zero_token = self.zero_token.expand(features.size(0), -1, -1)
                    b_ins_prompts_feats = [self.b3(torch.cat((zero_token, ins_prompt), dim=1))[:, 0, :] for ins_prompt in ins_prompts]
                    for i in range(10):
                        ins_prompts[i] = self.bottlenecks_ins_prompts[i](b_ins_prompts_feats[i])
                ins_prompts_cls_scores_list = []
                for i in range(10):
                    score = self.classifiers_ins_prompts[i](ins_prompts[i])
                    #print(ins_prompts[i].mean(dim=1, keepdim=True).size())
                    ins_prompts_cls_scores_list.append(score)
                #ins_prompts_cls_score = sum(ins_prompts_cls_scores_list) / len(ins_prompts_cls_scores_list)
                #ins_prompts_cls_score = [self.classifier_ins_prompts(ins_prompt.reshape(ins_prompt.size(0), ins_prompt.size(1)*ins_prompt.size(2))) for ins_prompt in ins_prompts]
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4], ins_prompts_cls_scores_list  # global feature for triplet loss
            else:
                return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                            cls_score_4
                            ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                                local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                if cfg.TEST.TSNE:
                    return torch.cat(
                        [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1),X_tsne_1,X_tsne_2 
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if i not in self.state_dict(): continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
