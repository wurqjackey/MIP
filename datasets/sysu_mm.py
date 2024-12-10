# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
'''
import os
from glob import glob

#from fastreid.data.datasets import DATASET_REGISTRY
from .bases import BaseImageDataset

__all__ = ['SYSU_mm', ]


class SYSU_mm(BaseImageDataset):
    """sysu mm
    """
    dataset_dir = "SYSU-MM01"
    dataset_name = "sysumm01"

    def __init__(self, root='data', **kwargs):
        super(SYSU_mm, self).__init__()
        
        self.root = root
        self.train_path = os.path.join(self.root, self.dataset_dir)

        required_files = [self.train_path]
        self.check_before_run(required_files)

        self.train = self.process_train(self.train_path)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)

        

    def process_train(self, train_path):
        data = []

        file_path_list = ['cam1', 'cam2', 'cam4', 'cam5']

        for file_path in file_path_list:
            camid = self.dataset_name + "_" + file_path
            pid_list = os.listdir(os.path.join(train_path, file_path))
            for pid_dir in pid_list:
                pid = self.dataset_name + "_" + pid_dir
                img_list = glob(os.path.join(train_path, file_path, pid_dir, "*.jpg"))
                for img_path in img_list:
                    data.append([img_path, pid, camid, 1])
        return data

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


'''
import os.path as osp
import glob
from .bases import BaseImageDataset
import random
from config import cfg
from utils.seedg import seed_gen

class SYSU_mm(BaseImageDataset):
    dataset_dir = 'SYSU-MM01'

    def __init__(self, root='', verbose=True, pid_begin=0, mode='all', setting='one', trial=0, **kwargs):
        super(SYSU_mm, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_dirs = [
            'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6'
        ]
        self.exp_dir = osp.join(self.dataset_dir, 'exp')
        self.train_id_file = osp.join(self.exp_dir, 'train_id.txt')
        self.val_id_file = osp.join(self.exp_dir, 'val_id.txt')
        self.test_id_file = osp.join(self.exp_dir, 'test_id.txt')
        self.pid_begin = pid_begin
        self._check_before_run()
        self.id_mapping = {}
        self.mode = cfg.DATASETS.MODE
        self.setting = cfg.DATASETS.SETTING
        self.trial = seed_gen() if seed_gen() else cfg.DATASETS.TRIAL
        

        train, query, gallery = self._process_dir()

        if verbose:
            print("=> SYSU-MM01 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files and folders are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        for cam_dir in self.cam_dirs:
            if not osp.exists(osp.join(self.dataset_dir, cam_dir)):
                raise RuntimeError("'{}' is not available".format(osp.join(self.dataset_dir, cam_dir)))
        if not osp.exists(self.exp_dir):
            raise RuntimeError("'{}' is not available".format(self.exp_dir))
        if not osp.exists(self.train_id_file):
            raise RuntimeError("'{}' is not available".format(self.train_id_file))
        if not osp.exists(self.test_id_file):
            raise RuntimeError("'{}' is not available".format(self.test_id_file))

    def _process_dir(self):
        train_ids = self._read_ids(self.train_id_file)
        val_ids = self._read_ids(self.val_id_file)
        test_ids = self._read_ids(self.test_id_file)

        train_dataset = []
        val_dataset = []
        query_dataset = []
        gallery_dataset = []

        # relabel
        pid_container = set()
        for person_id in train_ids:
            pid_container.add(person_id)
        for person_id in val_ids:
            pid_container.add(person_id)
        for person_id in test_ids:
            pid_container.add(person_id)
        #pid_max = len(pid_container)
        #print(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Train dataset
        for cam_dir in self.cam_dirs:
            if cam_dir in ['cam3', 'cam6']:
                modality_id = int(0)
            else:
                modality_id = int(1)
            cam_path = osp.join(self.dataset_dir, cam_dir)
            for person_id in train_ids:
                person_path = osp.join(cam_path, str(person_id).zfill(4))
                person_id = self.relabel_id(person_id)
                #if person_id > 20: continue
                if osp.isdir(person_path):
                    img_paths = glob.glob(osp.join(person_path, '*.jpg'))
                    for img_path in img_paths:
                        train_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, modality_id))

        # Val dataset (appended to training set)
        for cam_dir in self.cam_dirs:
            if cam_dir in ['cam3', 'cam6']:
                modality_id = int(0)
            else:
                modality_id = int(1)
            cam_path = osp.join(self.dataset_dir, cam_dir)
            for person_id in val_ids:
                person_path = osp.join(cam_path, str(person_id).zfill(4))
                person_id = self.relabel_id(person_id)
                #if person_id > 20: continue
                if osp.isdir(person_path):
                    img_paths = glob.glob(osp.join(person_path, '*.jpg'))
                    for img_path in img_paths:
                        val_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, modality_id))

        # Test dataset
        for cam_dir in ['cam3', 'cam6']:
            cam_path = osp.join(self.dataset_dir, cam_dir)
            for person_id in test_ids:
                person_path = osp.join(cam_path, str(person_id).zfill(4))
                if osp.isdir(person_path):
                    img_paths = glob.glob(osp.join(person_path, '*.jpg'))
                    #person_id = pid2label[person_id]
                    for img_path in img_paths:
                        query_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, int(0)))

        if self.mode == 'all':
            gallery_cameras = ['cam1','cam2','cam4','cam5']
        elif self.mode =='indoor':
            gallery_cameras = ['cam1','cam2']
        random.seed(self.trial)
        
        for cam_dir in gallery_cameras:
            cam_path = osp.join(self.dataset_dir, cam_dir)
            for person_id in test_ids:
                person_path = osp.join(cam_path, str(person_id).zfill(4))
                if osp.isdir(person_path):
                    img_paths = glob.glob(osp.join(person_path, '*.jpg'))
                    #person_id = pid2label[person_id]
                    #for img_path in img_paths:
                    #    gallery_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1), 1)
                    if self.setting == 'one':
                        img_path = random.choice(img_paths)
                        gallery_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, int(1))) # modality_id = 1
                    elif self.setting == 'val':
                        img_path = random.choice(img_paths, 5, replace=False)
                        gallery_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, int(1))) # modality_id = 1
                    else:
                        img_path_multi = []
                        for i in range(10):
                            img_path = random.choice(img_paths)
                            gallery_dataset.append((img_path, self.pid_begin + person_id, int(cam_dir[-1]) - 1, 1, int(1))) # modality_id = 1
                            img_paths.remove(img_path)

        return train_dataset+val_dataset, query_dataset, gallery_dataset
    
    def relabel_id(self, original_id):
        if original_id in self.id_mapping:
            return self.id_mapping[original_id]
        else:
            new_id = len(self.id_mapping)
            self.id_mapping[original_id] = new_id
            return new_id

    def _read_ids(self, file_path):
        with open(file_path, 'r') as f:
            ids = f.read().splitlines()
            print(ids)
        ids = ids[0].split(',')
        print(len(ids))
        ids = [int(x) for x in ids]
        return ids
    

