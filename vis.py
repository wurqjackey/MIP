import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
import cv2
import torch

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    '''if cfg.MODEL.USE_PROMPT and cfg.MODEL.USE_INS_PROMPT:
        result = tensor[:, 1:-32, :].reshape(tensor.size(0), 21, 10, tensor.size(2))
    elif cfg.MODEL.USE_PROMPT or cfg.MODEL.USE_INS_PROMPT:
        result = tensor[:, 1:-16, :].reshape(tensor.size(0), 21, 10, tensor.size(2))
    else:
        result = tensor[:, 1:, :].reshape(tensor.size(0), 21, 10, tensor.size(2))'''
    #result = tensor[:, 1:-16, :].reshape(tensor.size(0), 14, 15, tensor.size(2))
    print(tensor.size())
    result = tensor[:, 1:, :].reshape(tensor.size(0), 21, 10, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ReshapeTransform:
    def __init__(self, model):
        input_size = model.base.patch_embed.img_size
        patch_size = model.base.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):# x是个token序列
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        #拿到所有组成原图的token，将它们reshape回原图的大小
        result = x[:, 1:, :].reshape(x.size(0),#从1开始，忽略掉class_token
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result
    
def image_pre(image_path):
    image_path = image_path
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (128, 256))

    # 预处理图像
    input_tensor = preprocess_image(rgb_img, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    input_tensor = input_tensor.cuda()
    #input_tensor = torch.unsqueeze(input_tensor, dim=0)
    print(input_tensor.size())
    return rgb_img, input_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--input_image", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--input_path", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    #train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    #print(num_classes, camera_num, view_num)
    model = make_model(cfg, num_class=395, camera_num=6, view_num = 1)
    model.load_param(cfg.TEST.WEIGHT)
    device = "cuda"
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    print(model.b1[0].norm1)

    target_layers = [model.b1[0].norm1]

    rgb_img, input_tensor = image_pre(image_path=args.input_image)
        
    
    cam_path = 'vis/cam.jpg'
        # Construct the CAM object once, and then re-use it on many images:
    with GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
        #cam.batch_size = 1
        # targets = [ClassifierOutputTarget(7846)]
        # targets = [BinaryClassifierOutputTarget]
        #targets = 10
        targets = None
        print('yes')
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        print('yes')
        grayscale_cam = grayscale_cam[0, :]
        # print(grayscale_cam.shape)
        # print(rgb_img.shape)
        # print(grayscale_cam)
        # grayscale_cam[8:163][95:105] = 1.0
        print('yes')
        print(type(rgb_img))
        #rgb_img[rgb_img > 128.] += 10000
        visualization = show_cam_on_image(rgb_img/255., grayscale_cam, use_rgb=True, image_weight=0.56)
        print('yes')
        # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
        print('yes')
        cv2.imwrite(cam_path, visualization)
        # cv2.imwrite(cam_path, rgb_img)

    '''if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)'''

