# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(labels_name))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if x_val == y_val:
            color = 'white'
        else:
            color = 'black'

        plt.text(x_val, y_val, "%0.2f" % (c,), color=color, fontsize=12, va='center', ha='center')
    # plt.title(title)    # 图像标题
    # plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.tight_layout()  # 显示图形
    plt.show()
    # plt.savefig('/HAR_cm.png', format='png')


if __name__ == "__main__":
    # ARIL cross Location with ResTransform
    cm = [[0.8490566037735849  ,0.22284946236559142  ,0.2475  ,0.3278654970760234  ,0.20229166666666668  ],
          [0.2433606557377049  ,0.8113207547169812  ,0.24625  ,0.24394736842105263  ,0.2284375  ],
          [0.2631693989071039  ,0.25591397849462366  ,0.9571428571428571  ,0.31982456140350873  ,0.18333333333333335  ],
          [0.29516393442622946  ,0.21881720430107526  ,0.2575  ,0.8557692307692307  ,0.2072916666666667  ],
          [0.21830601092896176  ,0.3452150537634408  ,0.21500000000000002  ,0.23017543859649126  ,0.7641509433962265  ]]
    # labels_name = np.array(['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5','Loc6'])
    # labels_name = np.array(['up', 'down', 'left', 'right', 'circle','cross'])
    labels_name = np.array(['push', 'slide', 'raise', 'clap', 'circle','fist'])


    # file = "/data/projs/WiGr/lighting_logs/aril-5/loc-4-Class-style_PN-style_version_13/comfumat_metirc_all.npy"
    file = "/data/projs/WiGr/lighting_logs/csi_301-5/loc-4-Class-style_PN-style_version_7/comfumat_metirc_all.npy"
    file1 = "/data/projs/WiGr/lighting_logs/csi_301-5/loc-3-Class-style_PN-style_version_15/comfumat_metirc_all.npy"
    file2 = "/data/projs/WiGr/lighting_logs/csi_301-5/loc-4-Class-style_PN-style_version_9/comfumat_metirc_all.npy"
    file22 = "/data/projs/WiGr/lighting_logs/aril-5/loc-3-Class-style_PN-style_version_5/comfumat_metirc_all.npy"
    file1 = "/data/projs/WiGr/lighting_logs/widar-5-s305/user-s3-ori[1]-loc[2]-amp&pha-Class-style_PN-style_version_9/comfumat_metirc_all.npy"
    file2 = "/data/projs/WiGr/lighting_logs/widar-s3-05/user-s3-ori[1]-loc[2]-amp&pha-Class-style_PN-style_version_5/comfumat_metirc_all.npy"
    # comfumat_metirc_all2 = np.load(file2)
    # comfumat_metirc_all1 = np.load(file1)
    # cm = np.load(file)
    # cm1 = np.load(file1)
    # cm2 = np.load(file2)
    # cm=(cm1+cm2)/2
    cm=np.load("/data/projs/WiGr/lighting_logs/08-widar-s1/loc-dcosine-s1-ori[2]-u[2]-phase-Class-style_PN-style_version_71/comfumat_metirc_all.npy")
    # cm = (comfumat_metirc_all1[200]+comfumat_metirc_all2[300])/2
    # plot_confusion_matrix((cm[200]+comfumat_metirc_all1[300]++comfumat_metirc_all2[200])/3, labels_name, "In-Domain")
    plot_confusion_matrix((cm[150]), labels_name, "In-Domain")
    # plt.savefig('/HAR_cm.png', format='png')
    # plt.show()