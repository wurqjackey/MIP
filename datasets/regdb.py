import os.path as osp
from .bases import BaseImageDataset

class RegDB(BaseImageDataset):
    dataset_dir = 'RegDB'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(RegDB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.idx_dir = osp.join(self.dataset_dir, 'idx')
        self.thermal_dir = osp.join(self.dataset_dir, 'Thermal')
        self.visible_dir = osp.join(self.dataset_dir, 'Visible')
        self.train_thermal_file = osp.join(self.idx_dir, 'train_thermal_1.txt')
        self.train_visible_file = osp.join(self.idx_dir, 'train_visible_1.txt')
        self.test_thermal_file = osp.join(self.idx_dir, 'test_thermal_1.txt')
        self.test_visible_file = osp.join(self.idx_dir, 'test_visible_1.txt')
        self.pid_begin = pid_begin
        self._check_before_run()
        self.id_mapping = {}

        train, query, gallery = self._process_dir()

        if verbose:
            print("=> RegDB loaded")
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
        if not osp.exists(self.idx_dir):
            raise RuntimeError("'{}' is not available".format(self.idx_dir))
        if not osp.exists(self.thermal_dir):
            raise RuntimeError("'{}' is not available".format(self.thermal_dir))
        if not osp.exists(self.visible_dir):
            raise RuntimeError("'{}' is not available".format(self.visible_dir))
        if not osp.exists(self.train_thermal_file):
            raise RuntimeError("'{}' is not available".format(self.train_thermal_file))
        if not osp.exists(self.train_visible_file):
            raise RuntimeError("'{}' is not available".format(self.train_visible_file))
        if not osp.exists(self.test_thermal_file):
            raise RuntimeError("'{}' is not available".format(self.test_thermal_file))
        if not osp.exists(self.test_visible_file):
            raise RuntimeError("'{}' is not available".format(self.test_visible_file))

    def _process_dir(self):
        train_thermal = self._read_file(self.train_thermal_file)
        train_visible = self._read_file(self.train_visible_file)
        test_thermal = self._read_file(self.test_thermal_file)
        test_visible = self._read_file(self.test_visible_file)

        train = self._generate_data(train_thermal, train_visible)
        query = self._generate_data(thermal_data = test_thermal)
        gallery = self._generate_data(visible_data = test_visible)

        return train, query, gallery

    def _read_file(self, file_path):
        index = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    img_path, pid = line.split(' ')
                    index.append((img_path, int(pid)))
        return index

    def _generate_data(self, thermal_data=None, visible_data=None):
        data = []
        if thermal_data:
            for img_path, pid in thermal_data:
                thermal_img_path = osp.join(self.dataset_dir, img_path)
                data.append((thermal_img_path, pid, 0, 2, 0))  # (image_path, pid, camid=0, modality_id=0)
        if visible_data:
            for img_path, pid in visible_data:
                visible_img_path = osp.join(self.dataset_dir, img_path)
                data.append((visible_img_path, pid, 1, 2, 1))  # (image_path, pid, camid=1, modality_id=1)
        return data
