import glob
import os
import pickle

import scipy.io as scio
import zarr
import numpy as np
root_dir="/data/wifi/WiSR/Widar3/CSI/"
from DataSet.signal_process import deal_CSI
test=False


if  test:
    room_ids = ['1']
    rx_ids = ['1']
    ori_ids = ['1'] #, '2', '3', '4', '5'
    loc_ids=['1'] #,'2','3','4','5'
    user_ids=['1'] #,'2','3'
    ges_ids = ["Push&Pull", "Sweep"] #, "Sweep", "Clap", "Slide", "Draw-Zigzag(Vertical)", "Draw-N(Vertical)"
else:
    room_ids = ['1']
    rx_ids = ['1']
    ori_ids = ['1', '2', '3', '4', '5']  #
    loc_ids = ['1','2','3','4','5']  #
    user_ids = ['1','2','3']  #
    ges_ids = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-Zigzag(Vertical)", "Draw-N(Vertical)"]  #
sequence_len=1800

all_amp = []
all_pha = []

all_gesture = []
all_roomid = []
all_userid = []
all_location = []
# all_receiverid = []
all_face_orientation = []
for room in room_ids:
    for user in user_ids:
        for ges in ges_ids:
            for loc in loc_ids:
                for ori in ori_ids:
                    for rx in rx_ids:
                        mat_file_name_pattern = 'room_' + room + '_user_' + user + '_ges_' + ges + '_loc_' + loc + '_ori_' + ori + '_rx_' + rx + '*_csi.mat'
                        mat_files = glob.glob(os.path.join(root_dir, f"matfile_ours_room1", mat_file_name_pattern))
                        print('###############pattern', mat_file_name_pattern)
                        for mat_file in mat_files:
                            # room_1_user_1_ges_Clap_loc_1_ori_1_rx_1_csi.mat
                            if os.path.isfile(mat_file):
                                print('load     : ', mat_file)
                                mat = scio.loadmat(mat_file)
                                # mat_datas=list(mat.values())[-1][0]
                                mat_datas = list(mat.values())[-1].reshape(-1, 30, 3).transpose(-1, 1, 0)
                                mat_datas = mat_datas[np.newaxis, :]
                                for csi_data in mat_datas:
                                    # amp, pha = deal_CSI(csi_data,  IFfilter=True, IFphasani=True,
                                    #                     padding_length=sequence_len, step_size=1)
                                    # all_amp.append(amp)
                                    # all_pha.append(pha)
                                    all_gesture.append(ges_ids.index(ges))
                                    all_roomid.append(room)
                                    all_userid.append(user)
                                    all_location.append(loc)
                                    all_face_orientation.append(ori)
                            else:
                                raise ValueError('缺少mat:', mat_file)

all_amp=np.array(all_amp)
all_pha=np.array(all_pha)

all_gesture=np.array(all_gesture)
all_roomid=np.array(all_roomid)
all_location=np.array(all_location)
all_userid=np.array(all_userid)
all_face_orientation=np.array(all_face_orientation)

multi_label={
    "roomid":all_roomid,
    "userid":all_userid,
    "gesture":all_gesture,
    "location":all_location,
    "face_orientation":all_face_orientation,
}


out=os.path.join("/data/wifi/WiGr","widar")
if not os.path.exists(out):
    os.makedirs(out)

# np.save(os.path.join(out,"amp.npy"),all_amp)
# np.save(os.path.join(out,"pha.npy"),all_pha)


# with open(os.path.join(out,"multi_label.npy"), 'wb') as file:
#     pickle.dump(multi_label, file)

fileout=os.path.join(out,"multi_label1.npy")
np.savez(fileout, **multi_label)