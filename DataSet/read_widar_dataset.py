import numpy as np
import os

root=os.path.join("/data/wifi/WiGr","widar")


multi_label = np.load(os.path.join(root,"multi_label1.npy.npz") ,allow_pickle=True)

all_amp=np.load(os.path.join(root,"amp.npy"))
pass