import torch
from torch import nn
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        N = inputs.size(0)
        #N = len(inputs)
        id_num = N // 2 // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(2, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - 2 * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range (id_num):
            list_an.append(torch.mean(an[i][an[i]>1e-6]))
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean

        return loss

class MSEL(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MSEL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets, flag):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        #target, _ = targets.chunk(2,0)
        target = targets
        N = target.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())

        dist_mat = pdist_torch(inputs, inputs)

        '''dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]'''
        dist_intra_rgb = dist_mat[flag == 0, :][:, flag == 0]
        #print(dist_intra_rgb.size())
        dist_cross_rgb = dist_mat[flag == 0, :][:, flag == 1]
        #print(dist_cross_rgb.size())
        dist_intra_ir = dist_mat[flag == 1, :][:, flag == 1]
        dist_cross_ir = dist_mat[flag == 1, :][:, flag == 0]

        is_pos_intra_rgb = is_pos[flag == 0, :][:, flag == 0]
        #print(is_pos_intra_rgb.size())
        is_pos_cross_rgb = is_pos[flag == 0, :][:, flag == 1]
        #print(is_pos_cross_rgb)
        is_pos_intra_ir = is_pos[flag == 1, :][:, flag == 1]
        is_pos_cross_ir = is_pos[flag == 1, :][:, flag == 0]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos_intra_rgb * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos_intra_ir * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos_cross_rgb].contiguous()  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim =0)

        dist_cross_ir = dist_cross_ir[is_pos_cross_ir].contiguous()  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=0)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class MSEL_new(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MSEL_new, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets, flag):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        #target, _ = targets.chunk(2,0)
        target = targets
        N = target.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())

        dist_mat = pdist_torch(inputs, inputs)

        '''dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]'''
        dist_intra_rgb = dist_mat[flag == 0, :][:, flag == 0]
        #print(dist_intra_rgb.size())
        dist_cross_rgb = dist_mat[flag == 0, :][:, flag == 1]
        #print(dist_cross_rgb.size())
        dist_intra_ir = dist_mat[flag == 1, :][:, flag == 1]
        dist_cross_ir = dist_mat[flag == 1, :][:, flag == 0]

        is_pos_intra_rgb = is_pos[flag == 0, :][:, flag == 0]
        #print(is_pos_intra_rgb.size())
        is_pos_cross_rgb = is_pos[flag == 0, :][:, flag == 1]
        #print(is_pos_cross_rgb)
        is_pos_intra_ir = is_pos[flag == 1, :][:, flag == 1]
        is_pos_cross_ir = is_pos[flag == 1, :][:, flag == 0]

        eye_rgb = torch.randn((is_pos_intra_rgb.size(0),is_pos_intra_rgb.size(0))).cuda()
        eye_ir = torch.randn((is_pos_intra_ir.size(0),is_pos_intra_ir.size(0))).cuda()
        is_pos_intra_rgb = is_pos_intra_rgb * eye_rgb.ne(eye_rgb.diag().diag_embed())
        is_pos_intra_ir = is_pos_intra_ir * eye_ir.ne(eye_ir.diag().diag_embed())
        
        

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())


        limit_rgb = torch.any(is_pos_intra_rgb, dim=1, keepdim=True) * torch.any(is_pos_cross_rgb, dim=1, keepdim=True)
        limit_ir = torch.any(is_pos_intra_ir, dim=1, keepdim=True) * torch.any(is_pos_cross_ir, dim=1, keepdim=True)

        zero_tensor = torch.tensor(0.0).cuda()
        dist_intra_rgb = is_pos_intra_rgb * dist_intra_rgb * limit_rgb
        #intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        dist_intra_rgb_count_nonzero = dist_intra_rgb.count_nonzero(dim=1)
        intra_mean_rgb = torch.where(dist_intra_rgb_count_nonzero>0, torch.sum(dist_intra_rgb, dim=1)/dist_intra_rgb_count_nonzero, zero_tensor)

        dist_intra_ir = is_pos_intra_ir * dist_intra_ir * limit_ir
        #intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        #intra_mean_ir = torch.mean(intra_ir, dim=1)
        dist_intra_ir_count_nonzero = dist_intra_ir.count_nonzero(dim=1)
        intra_mean_ir = torch.where(dist_intra_ir_count_nonzero>0, torch.sum(dist_intra_ir, dim=1)/dist_intra_ir_count_nonzero, zero_tensor)

        dist_cross_rgb = is_pos_cross_rgb * dist_cross_rgb * limit_rgb
        #cross_rgb, _ = dist_cross_rgb.topk(self.num_pos, dim=1 ,largest = True, sorted = False) 
        #cross_mean_rgb = torch.mean(cross_rgb, dim=1)
        dist_cross_rgb_count_nonzero = dist_cross_rgb.count_nonzero(dim=1)
        cross_mean_rgb = torch.where(dist_cross_rgb_count_nonzero>0, torch.sum(dist_cross_rgb, dim=1)/dist_cross_rgb_count_nonzero, zero_tensor)

        dist_cross_ir = is_pos_cross_ir * dist_cross_ir * limit_ir
        #cross_ir, _ = dist_cross_ir.topk(self.num_pos, dim=1 ,largest = True, sorted = False)
        #cross_mean_ir = torch.mean(cross_ir, dim=1)
        dist_cross_ir_count_nonzero = dist_cross_ir.count_nonzero(dim=1)
        cross_mean_ir = torch.where(dist_cross_ir_count_nonzero>0, torch.sum(dist_cross_ir, dim=1)/dist_cross_ir_count_nonzero, zero_tensor)
        
        #dist_cross_rgb = dist_cross_rgb[is_pos_cross_rgb].contiguous()  # [N, num_pos]
        #cross_mean_rgb = torch.mean(dist_cross_rgb, dim =0)

        #dist_cross_ir = dist_cross_ir[is_pos_cross_ir].contiguous()  # [N, num_pos]
        #cross_mean_ir = torch.mean(dist_cross_ir, dim=0)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class MSEL_modal(nn.Module):
    def __init__(self,num_pos,feat_norm = 'no'):
        super(MSEL_modal, self).__init__()
        self.num_pos = num_pos//2
        self.feat_norm = feat_norm

    def forward(self, inputs, targets, flag):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        #target, _ = targets.chunk(2,0)
        target = targets
        N = target.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())

        dist_mat = pdist_torch(inputs, inputs)

        '''dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]
        is_pos_intra_rgb = is_pos[0 : N, 0 : N]
        is_pos_cross_rgb = is_pos[0 : N, N : 2*N]
        is_pos_intra_ir = is_pos[N : 2*N, N : 2*N]
        is_pos_cross_ir = is_pos[N : 2*N, 0 : N]'''
        dist_intra_rgb = dist_mat[flag == 0, :][:, flag == 0]
        #print(dist_intra_rgb.size())
        dist_cross_rgb = dist_mat[flag == 0, :][:, flag == 1]
        #print(dist_cross_rgb.size())
        dist_intra_ir = dist_mat[flag == 1, :][:, flag == 1]
        dist_cross_ir = dist_mat[flag == 1, :][:, flag == 0]

        is_pos_intra_rgb = is_pos[flag == 0, :][:, flag == 0]
        #print(is_pos_intra_rgb.size())
        is_pos_cross_rgb = is_pos[flag == 0, :][:, flag == 1]
        #print(is_pos_cross_rgb)
        is_pos_intra_ir = is_pos[flag == 1, :][:, flag == 1]
        is_pos_cross_ir = is_pos[flag == 1, :][:, flag == 0]
        

        # shape [N, N]
        #is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos_intra_rgb * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos_intra_ir * dist_intra_ir
        #print(dist_intra_ir)
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        #print(intra_ir.size())
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = is_pos_cross_rgb * dist_cross_rgb
        cross_rgb, _ = dist_cross_rgb.topk(self.num_pos, dim=1, largest=True, sorted=False)
        #print(cross_rgb.size())
        cross_mean_rgb = torch.mean(cross_rgb, dim=1)

        dist_cross_ir = is_pos_cross_ir * dist_cross_ir
        cross_ir, _ = dist_cross_ir.topk(self.num_pos, dim=1, largest=True, sorted=False)
        cross_mean_ir = torch.mean(cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class MSEL_Cos(nn.Module):          # for features after bn
    def __init__(self,num_pos):
        super(MSEL_Cos, self).__init__()
        self.num_pos = num_pos

    def forward(self, inputs, targets):

        inputs = nn.functional.normalize(inputs, p=2, dim=1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = 1 - torch.matmul(inputs, torch.t(inputs))

        dist_intra_rgb = dist_mat[0: N, 0: N]
        dist_cross_rgb = dist_mat[0: N, N: 2*N]
        dist_intra_ir = dist_mat[N: 2*N, N: 2*N]
        dist_cross_ir = dist_mat[N: 2*N, 0: N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)  # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
               torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss


class MSEL_Feat(nn.Module):    # compute MSEL loss by the distance between sample and center
    def __init__(self, num_pos):
        super(MSEL_Feat, self).__init__()
        self.num_pos = num_pos

    def forward(self, input1, input2):
        N = input1.size(0)
        id_num = N // self.num_pos

        feats_rgb = input1.chunk(id_num, 0)
        feats_ir = input2.chunk(id_num, 0)

        loss_list = []
        for i in range(id_num):
            cross_center_rgb = torch.mean(feats_rgb[i], dim=0)  # cross center
            cross_center_ir = torch.mean(feats_ir[i], dim=0)

            for j in range(self.num_pos):

                feat_rgb = feats_rgb[i][j]
                feat_ir = feats_ir[i][j]

                intra_feats_rgb = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)  # intra center
                intra_feats_ir = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)

                intra_center_rgb = torch.mean(intra_feats_rgb, dim=0)
                intra_center_ir = torch.mean(intra_feats_ir, dim=0)

                dist_intra_rgb = pdist_torch(feat_rgb.view(1, -1), intra_center_rgb.view(1, -1))
                dist_intra_ir = pdist_torch(feat_ir.view(1, -1), intra_center_ir.view(1, -1))

                dist_cross_rgb = pdist_torch(feat_rgb.view(1, -1), cross_center_ir.view(1, -1))
                dist_cross_ir = pdist_torch(feat_ir.view(1, -1), cross_center_rgb.view(1, -1))

                loss_list.append(torch.pow(dist_cross_rgb - dist_intra_rgb, 2) + torch.pow(dist_cross_ir - dist_intra_ir, 2))

        loss = sum(loss_list) / N / 2

        return loss
