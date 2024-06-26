import zipfile
from pathlib import Path
import numpy as np
import io
import torch.utils.data as data
import torch
import os
import itertools,functools
from torch.utils.data import TensorDataset, DataLoader
from DataSet.CustomDataset import CustomIterDataset

def load_npy_from_bytes(bytes_data):
    # return np.load(io.BytesIO(bytes_data))
    return bytes_data

def split_array_bychunk(array, chunksize, include_residual=True):
    len_ = len(array) // chunksize * chunksize
    array, array_residual = array[:len_], array[len_:]
    # array = np.split(array, len_ // chunksize)
    array = [
        array[i * chunksize: (i + 1) * chunksize]
        for i in range(len(array) // chunksize)
    ]
    if include_residual:
        if len(array_residual) == 0:
            return array
        else:
            return array + [
                array_residual,
            ]
    else:
        if len(array_residual) == 0:
            return array, None
        else:
            return array, array_residual


class Widar(CustomIterDataset):
    def __init__(self, root, roomid=None,userid=None,location=None,orientation=None,receiverid=None,sampleid=None,
                 data_shape=None,chunk_size=None,num_shot=1,batch_size=50,mode=None,trainmode=None,trainsize=0.8):
        """
        :param root: the path of the datasets
        :param roomid: choosing the data from specific room, i.e. roomid = [1,2] for choosing the data from room1 & room2
        :param userid: choosing the data from specific user, i.e. userid = [1,2] for choosing the data from user1 & user2
        :param location: choosing the data from specific location, i.e. location = [1,2] for choosing the data from location1 & location2
        :param orientation: choosing the data from specific orientation, i.e. orientation = [1,2] for choosing the data from orientation1 & orientation2
        :param receiverid: choosing the data from specific receiver, i.e. receiverid = [1,2] for choosing the data from receiver1 & receiver2
        :param sampleid: all these experiment repeats few times. by using sampleid, we can choose specific experiment.
        :param data_shape: if data_shape='split': we using repeat x chunck x 1 x subcarry data; else: '1D' we using the subcarry x time data; else '2D' we using the links X subcarry X time
        :param chunk_size: setting the length of every chunk on the time dimension.
        :param num_shot: the number of samples of each gesture(class) in the support set
        :param batch_size: the number of samples of each gesture(class) in both query set and support set.
        :param mode: if mode == "phase", we only use phase value. if mode == "amplitude", we only use amplitude value. otherwise, we use both.
        """

        super().__init__(trainmode,trainsize)
        
        self.root = root
        self.data_shape = data_shape
        self.chunk_size = chunk_size
        self.num_shot = num_shot
        self.batch_size = batch_size
        self.batch_idx = 0
        self.num_class = 6
        self.mode = mode

        # self.rm_info_all = np.load(root/"rm_info_all.npy")  # this is the label of the deleted wrong data

        multi_label = np.load(root/"multi_label.npy.npz")
        # total number: 98789
        self.room_label = multi_label["roomid"].astype(int)  # {1, 2, 3}
        self.userid = multi_label["userid"].astype(int)      # {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
        self.total_samples = len(multi_label["roomid"])
        temp_gesture = multi_label["gesture"].astype(int)    # {1, 2, 3, 4, 6, 9}
        for i,data in enumerate(temp_gesture):
            if temp_gesture[i] == 9:
                temp_gesture[i] = 5
        self.gesture = temp_gesture  # {1, 2, 3, 4, 5, 6}
        self.location = multi_label["location"].astype(int)  # {1, 2, 3, 4, 5}
        self.orientation = multi_label["face_orientation"].astype(int)  # {1, 2, 3, 4, 5}
        # self.sampleid = multi_label["sampleid"]   # {1, 2, 3, 4, 5}
        # self.receiverid = multi_label["receiverid"]  # {1, 2, 3, 4, 5, 6}

        # self.f_amp = zipfile.ZipFile(self.root / "amp.zip", mode="r")
        # self.f_pha = zipfile.ZipFile(self.root / "pha.zip", mode="r")

        self.f_amp=np.load(os.path.join(root,"amp.npy"))
        self.f_pha=np.load(os.path.join(root,"pha.npy"))
        self.select = np.ones(self.total_samples, dtype=bool)
        self.room_select = np.ones(self.total_samples, dtype=bool)
        self.user_select = np.ones(self.total_samples, dtype=bool)
        self.loc_select = np.ones(self.total_samples, dtype=bool)
        self.ori_select = np.ones(self.total_samples, dtype=bool)
        # self.receiver_select = np.ones(self.total_samples, dtype=np.bool)
        # self.sample_select = np.ones(self.total_samples, dtype=np.bool)
        index_temp = np.arange(self.total_samples)

        if roomid is not None:
            self.room_select = functools.reduce(np.logical_or,[*[self.room_label == j for j in roomid]])
            self.select = np.logical_and(self.select,self.room_select)
        if userid is not None:
            self.user_select = functools.reduce(np.logical_or,[ *[self.userid == j for j in userid]])
            self.select = np.logical_and(self.select, self.user_select)
        if location is not None:
            self.loc_select = functools.reduce(np.logical_or,[*[self.location == j for j in location]])
            self.select = np.logical_and(self.select, self.loc_select)
        if orientation is not None:
            self.ori_select = functools.reduce(np.logical_or,[ *[self.orientation == j for j in orientation]])
            self.select = np.logical_and(self.select, self.ori_select)
        # if receiverid is not None:
        #     self.receiver_select = functools.reduce(np.logical_or,[ *[self.receiverid == j for j in receiverid]])
        #     self.select = np.logical_and(self.select, self.receiver_select)
        # if sampleid is not None:
        #     self.sample_select = functools.reduce(np.logical_or, [*[self.sampleid == j for j in sampleid]])
        #     self.select = np.logical_and(self.select, self.sample_select)

        self.index = index_temp[self.select]  # the data for a specified task
        np.random.shuffle(self.index)

        choosed_label = self.gesture[self.index]
        num_sample_per_class = []
        self.sample_index_per_class = []
        for i in range(0,self.num_class):
            temp = self.index[np.where(choosed_label == i)]
            num_sample_per_class.append(len(temp))
            self.sample_index_per_class.append(temp)
        self.min_num_sample_class = min(num_sample_per_class)  # find the minimal number of samples of all classes
        self.num_batch = self.min_num_sample_class // self.batch_size

    def get_item(self, sample_index):
        if self.mode is not None:
            if self.mode == 'phase':
                # pha_sample =  (self.f_pha.read(str(sample_index)))
                pha_sample = load_npy_from_bytes(self.f_pha[sample_index])
                sample = pha_sample.astype(np.float32)  # shape [time,3,30]
            elif self.mode == 'amplitude':
                amp_sample = load_npy_from_bytes(self.f_amp[sample_index])
                sample = amp_sample.astype(np.float32)  # shape [time,3,30]
            else:
                amp_sample = load_npy_from_bytes(self.f_amp[sample_index])
                pha_sample = load_npy_from_bytes(self.f_amp[sample_index])
                amp_sample = amp_sample.astype(np.float32)  # shape [time,3,30]
                pha_sample = pha_sample.astype(np.float32)  # shape [time,3,30]
                sample = np.concatenate((amp_sample, pha_sample), axis=2)
        else:
            amp_sample = load_npy_from_bytes(self.f_amp[sample_index])
            pha_sample = load_npy_from_bytes(self.f_amp[sample_index])
            amp_sample = amp_sample.astype(np.float32)  # shape [time,3,30]
            pha_sample = pha_sample.astype(np.float32)  # shape [time,3,30]
            sample = np.concatenate((amp_sample, pha_sample), axis=2)

        ges_label = self.gesture[sample_index]     # {0,1, 2, 3, 4, 5}
        ges_label = torch.tensor(ges_label).type(torch.LongTensor)

        if self.data_shape == 'split':
            samp, samp_res = split_array_bychunk(sample, self.chunk_size,
                                                 include_residual=False)  # shape list{[chunk,3,30],repeat}
            samp += [sample[-self.chunk_size:], ]
            sample = torch.Tensor(np.array(samp))
            sample = sample.permute(0, 2, 1, 3).type(torch.FloatTensor)  # shape [repeat,3,chunk,30 or 60]
        elif self.data_shape == '1D':
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            if (self.mode == "phase") or (self.mode == "amplitude"):
                sample = sample.permute(1, 2, 0).reshape(90, -1)  # shape [3X30, time]
            else:
                sample = sample.permute(1, 2, 0).reshape(180, -1)  # shape [3X60, time]
        elif self.data_shape == '2D':
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            sample = sample.permute(1, 2, 0)
        else:
            sample = torch.from_numpy(sample).type(torch.FloatTensor)
            if (self.mode == "phase") or (self.mode == "amplitude"):
                sample = sample.permute(1, 2, 0).reshape(90, -1)  # shape [3X30, time]
            else:
                sample = sample.permute(1, 2, 0).reshape(180, -1)  # shape [3X60, time]

        return sample,ges_label,torch.tensor(1)

    def metric_data(self):
        # sampling a batch data and split to supportset and training/testing set
        datas_data = []
        datas_ges_label = []
        query_domain_label = []

        supports_data = []
        supports_ges_label = []
        supports_domain_label = []
        for i in range(self.num_class):
            temp = self.sample_index_per_class[i][
                   self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            for j in range(0, self.batch_size - self.num_shot):
                sample, ges_label, domain_label = self.get_item(temp[j])
                datas_data.append(sample)
                datas_ges_label.append(ges_label)
                query_domain_label.append(domain_label)
            for k in range(self.batch_size - self.num_shot, self.batch_size):
                sample, ges_label, domain_label = self.get_item(temp[k])
                supports_data.append(sample)
                supports_ges_label.append(ges_label)
                supports_domain_label.append(domain_label)

        self.batch_idx += 1
        return (datas_data, datas_ges_label, query_domain_label), (
        supports_data, supports_ges_label, supports_domain_label)

    def get_choose_label(self,id):
        if id == "user":
            return self.userid[self.index]
        if id =="room":
            return self.room_label[self.index]
        if id =="location":
            return self.location[self.index]
        if id =="orientation":
            return self.orientation[self.index]
        if id =="gesture":
            return self.gesture[self.index]
        if id =="receiver":
            return self.receiverid[self.index]
        if id =="sampleid":
            return self.sampleid[self.index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx > self.num_batch-1 :
            self.batch_idx = 0
            raise StopIteration
        return self.metric_data()

    def __len__(self):
        return self.num_batch

    # def __del__(self):
    #     self.f_amp.close()
    #     self.f_pha.close()


if __name__ == "__main__":
    pass


