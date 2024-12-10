from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
from config import cfg

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentityModalitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        self.index_dic_thermal = defaultdict(list) #dict with list value
        self.index_dic_visible = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _, modal) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            if modal == 0:
                self.index_dic_thermal[pid].append(index)

            else:
                self.index_dic_visible[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        batch_idxs_thermal_dict = defaultdict(list)
        batch_idxs_visible_dict = defaultdict(list)
        print(np.random.get_state()[1][0])

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            idxs_thermal = copy.deepcopy(self.index_dic_thermal[pid])
            idxs_visible = copy.deepcopy(self.index_dic_visible[pid])
            #print('before: thermal - {}, visible - {}'.format(len(idxs_thermal), len(idxs_visible)))
            
            
            if cfg.DATASETS.NAMES == 'regdb':
                '''if isinstance(idxs_thermal, list):
                    idxs_thermal = np.array(idxs_thermal)
                    idxs_visible = np.array(idxs_visible)
                    idxs = np.array(idxs)'''
                if len(idxs)%self.num_instances < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                if len(idxs_thermal)%(self.num_instances/2) < (self.num_instances/2):
                    #print(self.num_instances/2-len(idxs_visible)%(self.num_instances/2))
                    idxs_thermal = np.concatenate((idxs_thermal, np.random.choice(idxs_thermal, size=int(self.num_instances/2-len(idxs_visible)%(self.num_instances/2)), replace=True)))
                if len(idxs_visible)%(self.num_instances/2) < (self.num_instances/2):
                    idxs_visible = np.concatenate((idxs_visible, np.random.choice(idxs_visible, size=int(self.num_instances/2-len(idxs_visible)%(self.num_instances/2)), replace=True)))
                if len(idxs_thermal) < len(idxs_visible):
                    idxs_thermal = np.concatenate((idxs_thermal, np.random.choice(idxs_thermal, size=(len(idxs_visible)-len(idxs_thermal)), replace=True)))
                elif len(idxs_thermal) > len(idxs_visible):
                    idxs_visible = np.concatenate((idxs_visible, np.random.choice(idxs_visible, size=(len(idxs_thermal)-len(idxs_visible)), replace=True)))
            else:
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                if len(idxs_thermal) < (self.num_instances/2):
                    idxs_thermal = np.concatenate((idxs_thermal, np.random.choice(idxs_thermal, size=(self.num_instances/2-len(idxs_thermal)), replace=True)))
                if len(idxs_visible) < (self.num_instances/2):
                    idxs_visible = np.concatenate((idxs_visible, np.random.choice(idxs_visible, size=(self.num_instances/2-len(idxs_visible)), replace=True)))
                if len(idxs_thermal) < len(idxs_visible):
                    idxs_thermal = np.concatenate((idxs_thermal, np.random.choice(idxs_thermal, size=(len(idxs_visible)-len(idxs_thermal)), replace=True)))
                elif len(idxs_thermal) > len(idxs_visible):
                    idxs_visible = np.concatenate((idxs_visible, np.random.choice(idxs_visible, size=(len(idxs_thermal)-len(idxs_visible)), replace=True)))
            #print('after: thermal - {}, visible - {}'.format(len(idxs_thermal), len(idxs_visible)))
            random.shuffle(idxs)
            random.shuffle(idxs_thermal)
            random.shuffle(idxs_visible)
            #idxs_dict = dict(zip(idxs_thermal,idxs_visible))
            #batch_idxs = []
            batch_idxs_thermal = []
            batch_idxs_visible = []
            for i in range(len(idxs_thermal)):
                batch_idxs_thermal.append(idxs_thermal[i])
                batch_idxs_visible.append(idxs_visible[i])
                if len(batch_idxs_thermal) == self.num_instances/2:
                    batch_idxs = np.concatenate((batch_idxs_thermal,batch_idxs_visible))
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs_thermal_dict[pid].append(batch_idxs_thermal)
                    batch_idxs_visible_dict[pid].append(batch_idxs_visible)
                    batch_idxs_thermal = []
                    batch_idxs_visible = []
                    #batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            # 0011001100110011
            '''for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)'''
            # 0000000011111111
            for pid in selected_pids:
                batch_idxs = batch_idxs_thermal_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_thermal_dict[pid]) == 0:
                    avai_pids.remove(pid)
            for pid in selected_pids:
                batch_idxs = batch_idxs_visible_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)

        print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return self.length
