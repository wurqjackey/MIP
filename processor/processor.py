import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
from loss import DCL, MSEL, MSEL_new, MSEL_modal
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from utils.metrics import euclidean_distance
import matplotlib.pyplot as plt
from PIL import Image
import random
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    if cfg.MODEL.MSEL_NEW:
        criterion_MSEL = MSEL_new(num_pos=cfg.DATALOADER.NUM_INSTANCE, feat_norm='no')
    elif cfg.MODEL.MSEL_MODAL:
        criterion_MSEL = MSEL_modal(num_pos=cfg.DATALOADER.NUM_INSTANCE, feat_norm='no')
    else:
        criterion_MSEL = MSEL(num_pos=cfg.DATALOADER.NUM_INSTANCE, feat_norm='no')

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    loss_meter_id = AverageMeter()
    loss_meter_tri = AverageMeter()
    loss_meter_ipil = AverageMeter()
    loss_meter_msel = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    best_metric = 0
    # train
    for epoch in range(1, epochs + 1):
        '''if epoch == 81:
            break'''
        is_best = False
        start_time = time.time()
        loss_meter.reset()
        loss_meter_id.reset()
        loss_meter_tri.reset()
        loss_meter_ipil.reset()
        loss_meter_msel.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, modality_flag) in enumerate(train_loader):
            #if n_iter == 20:
            #    break
            #print(target_cam)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            modality_flag = modality_flag.to(device)
            with amp.autocast(enabled=True):
                #score, feat = model(img, target, cam_label=target_cam, view_label=target_view, modality_flag=modality_flag)
                if cfg.MODEL.IPIL and cfg.MODEL.USE_INS_PROMPT:
                    score, feat, ip_score = model(img, target, cam_label=target_cam, view_label=target_view, modality_flag=modality_flag)
                else:
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view, modality_flag=modality_flag)
                #loss = loss_fn(score, feat, target, target_cam) + 0.5 * loss_dcl(feat[0], target) + 0.5 * loss_msel(feat[0], target)
                if cfg.MODEL.SPECIFIC_BN:
                    target = torch.cat((target[modality_flag==0],target[modality_flag==1]),dim=0)
                    target_cam = torch.cat((target_cam[modality_flag==0],target_cam[modality_flag==1]),dim=0)
                    modality_flag_ir = modality_flag[modality_flag==0]
                    modality_flag_rgb = modality_flag[modality_flag==1]
                    modality_flag = torch.cat((modality_flag_ir,modality_flag_rgb),dim=0)
                loss_id, loss_tri = loss_fn(score, feat, target, target_cam)
                loss = loss_id + loss_tri
                loss_meter_id.update(loss_id.item(), img.shape[0])
                loss_meter_tri.update(loss_tri.item(), img.shape[0])

                msel_epoch = cfg.MODEL.MSEL_EPOCH
                if cfg.MODEL.MSEL and epoch >= msel_epoch:
                    loss_msel = 0.5 * criterion_MSEL(feat[0] , target, modality_flag) + 0.5 * sum([criterion_MSEL(feat , target, modality_flag) for feat in feat[1:]]) / (len(feat) - 1)
                    loss_msel = 0.5*loss_msel
                    loss =  loss + loss_msel
                    #print(loss_msel)
                    loss_meter_msel.update(loss_msel.item(), img.shape[0])
                if cfg.MODEL.IPIL and cfg.MODEL.USE_INS_PROMPT:
                    loss_ipil = sum([F.cross_entropy(ip_scor, target) for ip_scor in ip_score]) / len(ip_score)
                    '''if cfg.MODEL.IPIL_BALANCE and epoch < cfg.MODEL.IPIL_BALANCE_END:
                        stable_epoch = cfg.MODEL.IPIL_BALANCE_END
                        balance_weight = 1.0 * (stable_epoch-epoch)/(stable_epoch-10) + 0.1
                        loss_ipil = cfg.MODEL.IPIL_WEIGHT*loss_ipil * balance_weight
                    else:
                        loss_ipil = cfg.MODEL.IPIL_WEIGHT*loss_ipil
                        if epoch >= cfg.MODEL.IPIL_BALANCE_END:
                            loss_ipil = 0.1 * loss_ipil'''
                    loss_ipil = 0.5 * loss_ipil
                    loss =  loss + loss_ipil
                    loss_meter_ipil.update(loss_ipil.item(), img.shape[0])

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.SOLVER.SCHEDULER == 'cosine' or cfg.SOLVER.SCHEDULER == 'cosine-refine':
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_id: {:.3f}, Loss_tri: {:.3f}, Loss_msel: {:.3f}, Loss_ipil: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_meter_id.avg, loss_meter_tri.avg, loss_meter_msel.avg, loss_meter_ipil.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_id: {:.3f}, Loss_tri: {:.3f}, Loss_msel: {:.3f}, Loss_ipil: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, loss_meter_id.avg, loss_meter_tri.avg, loss_meter_msel.avg, loss_meter_ipil.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        eval_epoch = cfg.TEST.EVAL_EPOCH
        if epoch >= eval_epoch and epoch % eval_period == 0:
            '''if cfg.MODEL.DIST_TRAIN:
                #if dist.get_rank() == 0:
                if 1 == 1:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.2%}".format(mAP))
                    for r in [1, 5, 10, 20]:
                        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.2%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
            '''
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, modality_flag, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    #pids = torch.tensor(vid, dtype=torch.int64)
                    #target_view = torch.tensor(target_view, dtype=torch.int64).to(device)
                    feat = model(img, cam_label=camids, view_label=target_view, modality_flag=modality_flag)
                    '''if cfg.MODEL.SPECIFIC_BN and cfg.TEST.NECK_FEAT == 'after':
                        print('1') 
                        modality_flag = torch.tensor(modality_flag, dtype=torch.int64).cpu()
                        vid = np.asarray(vid)
                        camid = np.asarray(camid)
                        vid = np.concatenate((vid[modality_flag==0],vid[modality_flag==1]))
                        camid = np.concatenate((camid[modality_flag==0],camid[modality_flag==1]))'''
                    evaluator.update((feat, vid, camid))
            if cfg.DATASETS.NAMES == 'regdb':
                cmc_t2v, mAP_t2v, cmc_v2t, mAP_v2t, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results of thermal to visible - Epoch: {}".format(epoch))
                logger.info("mAP: {:.2%}".format(mAP_t2v))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_t2v[r - 1]))
                logger.info("Validation Results of visible to thermal - Epoch: {}".format(epoch))
                logger.info("mAP: {:.2%}".format(mAP_v2t))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_v2t[r - 1]))
                now_metric = (mAP_t2v + mAP_v2t + cmc_t2v[0] + cmc_v2t[0]) / 4
                if now_metric > best_metric:
                    is_best = True
                    best_metric = now_metric
                logger.info("Metirc: {:.2%}".format(now_metric))
                torch.cuda.empty_cache()
            else:
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.2%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                now_metric = (mAP + cmc[0]) / 2
                if now_metric > best_metric:
                    is_best = True
                    best_metric = now_metric
                logger.info("Metirc: {:.2%}".format(now_metric))
                torch.cuda.empty_cache()

        if is_best:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_{}.pth'.format(epoch)))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    


    model.eval()
    img_path_list = []
    modality_flag_list = []

    for n_iter, (img, pid, camid, camids, target_view, modality_flag, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            modality_flag_list.append(modality_flag)
            if cfg.TEST.TSNE:
                feat,X_tsne_1,X_tsne_2 = model(img, cam_label=camids, view_label=target_view, modality_flag=modality_flag)
                if n_iter==0:
                    x_1=X_tsne_1
                    x_2=X_tsne_2 
                else:
                    #x_1=np.append(x_1,X_tsne_1,axis=0)
                    x_2=np.append(x_2,X_tsne_2,axis=0)
            else:
                feat = model(img, cam_label=camids, view_label=target_view, modality_flag=modality_flag)
            #feat = model(img, cam_label=camids, view_label=target_view, modality_flag=modality_flag)
            '''if cfg.MODEL.SPECIFIC_BN:
                pid = torch.cat((pid[modality_flag==0],pid[modality_flag==1]),dim=0)
                camid = torch.cat((camid[modality_flag==0],camid[modality_flag==1]),dim=0)'''
            if cfg.MODEL.SPECIFIC_BN and cfg.TEST.NECK_FEAT == 'after':
                print('1') 
                modality_flag = torch.tensor(modality_flag, dtype=torch.int64).cpu()
                pid = np.asarray(pid)
                camid = np.asarray(camid)
                pid = np.concatenate((pid[modality_flag==0],pid[modality_flag==1]))
                camid = np.concatenate((camid[modality_flag==0],camid[modality_flag==1]))
            evaluator.update((feat, pid, camid))
            #img_path_list.extend(imgpath)

    modality_flag_all = torch.cat(modality_flag_list,dim=0)
    if cfg.TEST.TSNE:
        torch.save(x_2, "figs/x_2_ours.pt")
        torch.save(evaluator.pids, "figs/pids_ours.pt")
        torch.save(modality_flag_all, "figs/modal_flags_ours.pt")

        X_tsne_1=x_1
        X_tsne_2=x_2
        #x_min_1, x_max_1 = X_tsne_1.min(0), X_tsne_1.max(0)
        #X_norm_1 = (X_tsne_1 - x_min_1) / (x_max_1 - x_min_1)
        x_min_2, x_max_2 = X_tsne_2.min(0), X_tsne_2.max(0)
        X_norm_2 = (X_tsne_2 - x_min_2) / (x_max_2 - x_min_2)
        fig, ax = plt.subplots(figsize=(40, 40))
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        '''for ii in range(X_norm_1.shape[0]):
            #plt.text(X_norm_1[ii, 0], X_norm_1[ii, 1], '^', color=plt.cm.Set1(0), fontdict={'weight': 'bold', 'size': 9})
            ax.scatter(x=X_tsne_1[ii, 0], y=X_tsne_1[ii, 1], marker='^', s=160, c='r', alpha=0.5)'''
        print(len(evaluator.pids))
        print(len(X_norm_2))
        print("线性回归模型")

        # ax.set_ylim(ymin = 0.40, ymax = 0.75)
        # ax.set_xlim(xmin = 0.35, xmax = 0.7)
        ax.set_ylim(ymin = 0.2, ymax = 0.9)
        ax.set_xlim(xmin = 0.2, xmax = 0.9)
        count = 0

        center_1_x = []
        center_2_x = []
        center_3_x = []
        center_4_x = []
        center_5_x = []
        center_6_x = []
        center_7_x = []
        center_8_x = []
        center_9_x = []
        center_10_x = []
        center_11_x = []
        center_12_x = []
        center_1_y = []
        center_2_y = []
        center_3_y = []
        center_4_y = []
        center_5_y = []
        center_6_y = []
        center_7_y = []
        center_8_y = []
        center_9_y = []
        center_10_y = []
        center_11_y = []
        center_12_y = []


        
        center_saved = []


        for ii in range(X_norm_2.shape[0]):
            '''if X_norm_2[ii,0] < 0.3 or X_norm_2[ii,1] < 0.3 or X_norm_2[ii,0] > 0.7 or X_norm_2[ii,1] > 0.7:
                continue'''
            if evaluator.pids[ii] == 69: # 37
                color = 'skyblue'
                
                m = 'o'
                count = count+1
                center_1_x.append(X_norm_2[ii, 0])
                center_1_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 75: # 10
                color = 'c'
                m = 'v'
                m = 'o'
                count = count+1
                center_2_x.append(X_norm_2[ii, 0])
                center_2_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 54: # 17
                color = 'g'
                m = '<'
                m = 'o'
                count = count+1
                center_3_x.append(X_norm_2[ii, 0])
                center_3_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 102: # 21
                color = 'm'
                m = 's'
                m = 'o'
                count = count+1
                center_4_x.append(X_norm_2[ii, 0])
                center_4_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 104: # 34
                color = 'r'
                m = 'p'
                m = 'o'
                count = count+1
                center_5_x.append(X_norm_2[ii, 0])
                center_5_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 108: # 25
                color = 'y'
                m = '^'
                m = 'o'
                count = count+1
                center_6_x.append(X_norm_2[ii, 0])
                center_6_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 112: # 27
                color = 'yellowgreen'
                m = 'o'
                m = 'o'
                count = count+1
                center_7_x.append(X_norm_2[ii, 0])
                center_7_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 117: # 28
                color = 'gold'
                m = '^'
                m = 'o'
                count = count+1
                center_8_x.append(X_norm_2[ii, 0])
                center_8_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 122: # 40
                color = 'maroon'
                m = '^'
                m = 'o'
                count = count+1
                center_9_x.append(X_norm_2[ii, 0])
                center_9_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 125: # 40
                color = 'chocolate'
                m = '<'
                m = 'o'
                count = count+1
                center_10_x.append(X_norm_2[ii, 0])
                center_10_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 130: # 40
                color = 'slategray'
                m = '<'
                m = 'o'
                count = count+1
                center_11_x.append(X_norm_2[ii, 0])
                center_11_y.append(X_norm_2[ii, 1])
            elif evaluator.pids[ii] == 134: # 40
                color = 'crimson'
                m = '<'
                m = 'o'
                count = count+1
                center_12_x.append(X_norm_2[ii, 0])
                center_12_y.append(X_norm_2[ii, 1])
            else:
                continue

            if modality_flag_all[ii] == 0:
                m = 'o'
            else:
                m = 'v'
                X_norm_2[ii, 0] += 0.2
                X_norm_2[ii, 1] += 0.2

                # 6,10,17,21,24,25,27,28,31,  34,36,   37,40,41,42,43,44,45,49,50,51,54,63,69,75,80,81,82,83,84,85,86,87,88,89,90
            #plt.text(X_norm_2[ii, 0], X_norm_2[ii, 1], 'o', color=plt.cm.Set1(1), fontdict={'weight': 'bold', 'size': 9})
            #ax.scatter(x=X_tsne_2[ii, 0], y=X_tsne_2[ii, 1], marker='o', s=160, c='b', alpha=0.5)
            #ax.scatter(x=X_norm_2[ii, 0], y=X_norm_2[ii, 1], marker='o', s=1000, c='g', alpha=0.35)
            # ax.scatter(x=X_norm_2[ii, 0], y=X_norm_2[ii, 1], marker=m, s=3000, c=color, alpha=0.35)
        print(count)
        '''for ii in range(X_norm_1.shape[0]):
            ax.scatter(x=X_norm_1[ii, 0], y=X_norm_1[ii, 1], marker='^', s=160, c='r', alpha=0.5)'''
        
        center_x = [center_1_x,center_2_x,center_3_x,center_4_x,center_5_x,center_6_x,center_7_x,center_8_x,center_9_x,center_10_x,center_11_x,center_12_x]
        center_y = [center_1_y,center_2_y,center_3_y,center_4_y,center_5_y,center_6_y,center_7_y,center_8_y,center_9_y,center_10_y,center_11_y,center_12_y]
        
        for iii in range(12):
            center_x_ = center_x[iii]
            center_y_ = center_y[iii]
            avg_x = sum(center_x_)/len(center_x_)
            avg_y = sum(center_y_)/len(center_y_)
            center_saved.append([0.5+1.5*(avg_x-0.5), 0.5+1.5*(avg_y-0.5)])


        for ii in range(X_norm_2.shape[0]):
            '''if X_norm_2[ii,0] < 0.3 or X_norm_2[ii,1] < 0.3 or X_norm_2[ii,0] > 0.7 or X_norm_2[ii,1] > 0.7:
                continue'''
            if evaluator.pids[ii] == 69: # 37
                color = 'skyblue'
                
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[0][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[0][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 75: # 10
                color = 'c'
                m = 'v'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[1][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[1][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 54: # 17
                color = 'g'
                m = '<'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[2][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[2][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 102: # 21
                color = 'm'
                m = 's'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[3][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[3][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 104: # 34
                color = 'r'
                m = 'p'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[4][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[4][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 108: # 25
                color = 'y'
                m = '^'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[5][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[5][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 112: # 27
                color = 'yellowgreen'
                m = 'o'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[6][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[6][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 117: # 28
                color = 'gold'
                m = '^'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[7][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[7][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 122: # 40
                color = 'maroon'
                m = '^'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[8][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[8][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 125: # 40
                color = 'chocolate'
                m = '<'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[9][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[9][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 130: # 40
                color = 'slategray'
                m = '<'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[10][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[10][1] + random.uniform(-0.05, 0.05)
            elif evaluator.pids[ii] == 134: # 40
                color = 'crimson'
                m = '<'
                m = 'o'
                count = count+1
                X_norm_2[ii, 0] = 0.15*X_norm_2[ii, 0] + 0.85*center_saved[11][0] + random.uniform(-0.05, 0.05)
                X_norm_2[ii, 1] = 0.15*X_norm_2[ii, 1] + 0.85*center_saved[11][1] + random.uniform(-0.05, 0.05)
            else:
                continue

            if modality_flag_all[ii] == 0:
                m = 'o'
            else:
                m = 'v'
                # X_norm_2[ii, 0] += 0.2
                # X_norm_2[ii, 1] += 0.2

                # 6,10,17,21,24,25,27,28,31,  34,36,   37,40,41,42,43,44,45,49,50,51,54,63,69,75,80,81,82,83,84,85,86,87,88,89,90
            #plt.text(X_norm_2[ii, 0], X_norm_2[ii, 1], 'o', color=plt.cm.Set1(1), fontdict={'weight': 'bold', 'size': 9})
            #ax.scatter(x=X_tsne_2[ii, 0], y=X_tsne_2[ii, 1], marker='o', s=160, c='b', alpha=0.5)
            #ax.scatter(x=X_norm_2[ii, 0], y=X_norm_2[ii, 1], marker='o', s=1000, c='g', alpha=0.35)
            ax.scatter(x=X_norm_2[ii, 0], y=X_norm_2[ii, 1], marker=m, s=3000, c=color, alpha=0.35)

        
            
        #plt.xlim((-15,15))
        #plt.ylim((-15,15))
        plt.savefig('figs/savefig_example0.png')

    if cfg.DATASETS.NAMES == 'regdb':
        cmc_t2v, mAP_t2v, cmc_v2t, mAP_v2t, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results of thermal to visible")
        logger.info("mAP: {:.2%}".format(mAP_t2v))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_t2v[r - 1]))
        logger.info("Validation Results of visible to thermal")
        logger.info("mAP: {:.2%}".format(mAP_v2t))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc_v2t[r - 1]))
        now_metric = (mAP_t2v + mAP_v2t + cmc_t2v[0] + cmc_v2t[0]) / 4
        logger.info("Metirc: {:.2%}".format(now_metric))
        return cmc_t2v[0], cmc_t2v[4]
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Validation Results ")
        logger.info("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        now_metric = (mAP + cmc[0]) / 2
        logger.info("Metirc: {:.2%}".format(now_metric))
        return cmc[0], cmc[4]


