import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import argparse
from collections import defaultdict
import models
import DataSet

from pathlib import Path
import torch.utils.data as tdata
import matplotlib as plt

import numpy as np
from evaluation import confmat

from parameter_config import config
from  pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import itertools
import gc

def run(args):
    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()

    source_data = DataSet.create(name=args.dataset, root=args.root,roomid=args.train_roomid,
                                 userid=args.train_userid,
                                 location=args.train_location,orientation=args.train_orientation,
                                 receiverid=args.train_receiverid,sampleid=args.train_sampleid,
                                 data_shape=args.data_shape,chunk_size=args.chunk_size,num_shot=args.num_shot,
                                 batch_size=args.batch_size,mode=args.mode,trainmode=True,trainsize=0.8)

    target_data = DataSet.create(name=args.dataset, root=args.root,roomid=args.train_roomid,
                                 userid=args.train_userid,
                                 location=args.train_location,orientation=args.train_orientation,
                                 receiverid=args.train_receiverid,sampleid=args.train_sampleid,
                                 data_shape=args.data_shape,chunk_size=args.chunk_size,num_shot=args.num_shot,
                                 batch_size=args.batch_size,mode=args.mode,trainmode=False,trainsize=0.8)

    tr_loader = DataLoader(dataset=source_data,collate_fn=lambda x:x)
    te_loader = DataLoader(dataset=target_data, collate_fn=lambda x:x)

    data_model_match = True  # Whether the data format matches the model
    if args.model_name == 'PrototypicalResNet':
        if args.data_shape != '1D':
            data_model_match = False
        else:
            model = models.create(name=args.model_name,
                                  layers=args.layers,
                                  strides=args.strides,
                                  inchannel=args.ResNet_inchannel,
                                  groups=args.groups,
                                  align=args.align,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  num_domain_linear_flag=args.num_domain_linear_flag,
                                  combine=args.combine,
                                  class_feature_style=args.class_feature_style,
                                  domain_feature_style=args.domain_feature_style,
                                  pn_style=args.pn_style,
                                          use_attention=args.use_attention,
                                  )
    elif args.model_name == 'PrototypicalCnnLstmNet':
        if args.data_shape == 'split':
            model = models.create(name=args.model_name,
                                  in_channel_cnn=args.in_channel_cnn,
                                  out_feature_dim_cnn=args.out_feature_dim_cnn,
                                  out_feature_dim_lstm=args.out_feature_dim_lstm,
                                  num_lstm_layer=args.num_lstm_layer,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  combine=args.combine)
        else:
            data_model_match = False
    elif args.model_name == 'PrototypicalMobileNet':
        if args.data_shape != '2D':
            data_model_match = False
        else:
            model = models.create(name=args.model_name,
                                  width_mult=args.width_mult,
                                  inchannel=args.MobileNet_inchannel,
                                  align=args.out_feature_dim_lstm,
                                  metric_method=args.metric_method,
                                  k_shot=args.num_shot,
                                  num_class_linear_flag=args.num_class_linear_flag,
                                  combine=args.combine,
                                  args=args
                                  )
    else:
        model = models.create(name=args.model_name,
                              in_channel_cnn=args.in_channel_cnn,
                              out_feature_dim_cnn=args.out_feature_dim_cnn,
                              out_feature_dim_lstm=args.out_feature_dim_lstm,
                              num_lstm_layer=args.num_lstm_layer,
                              metric_method=args.metric_method,
                              k_shot=args.num_shot,
                              num_class_linear_flag=args.num_class_linear_flag,
                              combine=args.combine,
                              args=args
                              )

    if data_model_match:
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint( monitor='GesVa_loss', save_last =False, save_top_k =0)
        #自定义log
        existing_versions = []
        exp_name=f"test-input-{args.dataset}"
        if not os.path.exists(os.path.join(args.log_dir,exp_name)):
            os.makedirs(os.path.join(args.log_dir,exp_name))
        for bn in os.listdir(os.path.join(args.log_dir,exp_name)):
            # d = listing["name"]
            # bn = os.path.basename(d)
            if "version" in bn:
                dir_ver = bn.split("_")[-1].replace('/', '')
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            max_version=0
        else:
            max_version=max(existing_versions)+1
        if args.dataset == "widar":
            if args.cross_type=="loc":
                fixed =f"ori{args.train_orientation}-user{args.train_userid}"
            elif args.cross_type == "user":
                fixed =f"ori{args.train_orientation}-loc{args.train_location}"
            else:
                return

        if args.dataset == "csi_301":
            if args.cross_type=="loc":
                fixed =f"room{args.train_roomid}-user{args.train_userid}"
            elif args.cross_type == "user":
                fixed =f"room{args.train_roomid}-loc{args.train_location}"
            elif args.cross_type == "room":
                fixed = f"ori{args.train_userid}-loc{args.train_location}"
            else:
                return
        if args.dataset == "aril":
            fixed=""
        prefix=(f"{args.cross_type}-d{args.metric_method}-s{args.num_shot}--{args.mode}-{args.use_attention}"+(f"-Class-{args.class_feature_style}_" if args.num_class_linear_flag else "") + (
            f"PN-{args.pn_style}_" if args.pn_style else ""))

        suffix=f"-version_{str(max_version)}"
        prefix=prefix+fixed
        version=prefix+suffix
        import pytorch_lightning as pl
        tb_logger = TensorBoardLogger(save_dir=args.log_dir,name=exp_name,version=version)
        trainer = pl.Trainer(callbacks=[checkpoint_callback,],log_every_n_steps=1,max_epochs=args.max_epochs,gpus=1,logger=tb_logger )   # precision = 16
        trainer.fit(model,tr_loader,te_loader)
        print(trainer.logger.log_dir)
        # print(len(model.confmat_linear_all)

        # print([x.shape for x in model.comfmat_metric_all])
        cm_tensor_type = torch.stack(model.comfmat_metric_all)
        cm_numpy_type = cm_tensor_type.numpy()

        # cm = np.array(model.comfmat_metric_all)
        # print(type(model.comfmat_metric_all))
        # print(type(model.comfmat_metric_all[0]))
        from pathlib import Path
        np.save(Path(trainer.logger.log_dir)/'comfumat_metirc_all',cm_numpy_type)

        print(args)
        print(type(args))
        tb_logger.close()

        del model
        del tr_loader
        del te_loader
        del tb_logger
        del trainer
        del pl
        gc.collect()
        # torch.cuda.empty_cache()
        # labels_name = np.array(['user1', 'user2', 'user3', 'user4', 'user5','ss'])
        # confmat.plot_confusion_matrix(cm,labels_name,title="test")


def multi_exps_widar(args,ex_repeat):
    oris = [1, 2, 3, 4, 5]
    users = [1, 2, 3]
    locs = [1, 2, 3, 4, 5]
    shots = [1,2,3]
    shots = [1]

    ori_li = [list(pair) for pair in itertools.combinations(oris, 1)]
    user_li = [list(pair) for pair in itertools.combinations(users, 1)]
    loc_li = [list(pair) for pair in itertools.combinations(locs, 1)]
    if args.cross_type == "loc":
        for ori in ori_li:
            for user in user_li:
                for shot in shots:
                    args.train_orientation = ori
                    args.train_userid=user
                    args.num_shot = shot
                    for j in range(ex_repeat):
                        try:
                            print(args)
                            run(args)
                        except Exception as e:
                            print("#error！！！！！！！", e,args)
    elif args.cross_type == "user":
        for ori in ori_li:
            for loc in loc_li:
                for shot in shots:
                    args.train_orientation = ori
                    args.train_location = loc
                    args.num_shot = shot
                    for j in range(ex_repeat):
                        # print(111)
                        try:
                            print(args)
                            run(args)
                        except Exception as e:
                            print("#error！！！！！！！", e, args)
    else:
        print("Wrong cross_type")
        return


def multi_exps_csida(args,ex_repeat):
    # oris = [1, 2, 3, 4, 5]
    users = [0,1, 2, 3,4]
    rooms=[0,1]
    # locs={
    #     0:{0,1},
    #     1:{0,1,2},
    # }
    # locs = [0,1]
    shots = [1,2,3]
    shots = [1]
    user_li = [list(pair) for pair in itertools.combinations(users, 1)]
    # loc_li = [list(pair) for pair in itertools.combinations(locs, 1)]
    room_li = [list(pair) for pair in itertools.combinations(rooms, 1)]
    if args.cross_type == "loc":
            for user in user_li:
                for room in room_li:
                    for shot in shots:
                        args.train_roomid = room
                        args.train_userid=user
                        # if 0 in room:
                        #     locs=[0,1,2]
                        # else:
                        #     locs = [0, 1]
                        args.num_shot = shot
                        args.train_location=None
                        for j in range(ex_repeat):
                            try:
                                print(args)
                                run(args)
                            except Exception as e:
                                print("#error！！！！！！！", e, args)
    elif args.cross_type == "user":
        for room in room_li:
            if 0 in room:
                locs=[0,1,2]
            else:
                locs = [0, 1]
            loc_li = [list(pair) for pair in itertools.combinations(locs, 1)]
            for loc in loc_li:
                    args.train_roomid = room
                    args.train_location = loc
                    for shot in shots:
                        args.num_shot = shot
                        for j in range(ex_repeat):
                            try:
                                print(args)
                                run(args)
                            except Exception as e:
                                print("#error！！！！！！！", e, args)

    elif args.cross_type == "room":
        for room in room_li:
            for user in user_li:
                # if 0 in room:
                #     locs = [0, 1, 2]
                # else:
                locs = [0, 1]
                loc_li = [list(pair) for pair in itertools.combinations(locs, 1)]
                for loc in loc_li:
                    args.train_roomid = room
                    args.train_location = loc
                    args.train_userid=user
                    # args.num_shot = shot
                    for j in range(ex_repeat):
                        try:
                            print(args)
                            run(args)
                        except Exception as e:
                            print("#error！！！！！！！", e, args)
    else:
        print("Wrong cross_type")
        return

def multi_exps_aril(args,ex_repeat):
    shots = [2,3]
    # shots = [1]
    if args.cross_type == "loc":
        for shot in shots:
            args.num_shot = shot
            args.train_location=None
            for j in range(ex_repeat):
                try:
                    print(args)
                    run(args)
                except Exception as e:
                    print("#error！！！！！！！", e, args)
    else:
        print("Wrong cross_type")
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WiGr cross_domain experiment')

    parser.add_argument('--train_roomid', default=None,
                        help="Setting the room (domain) configuration: Widar:[1,2,3]; Aril:useless; CSI_301: [0,1]")
    parser.add_argument('--train_userid', default=None,
                        help="Setting the user (domain) configuration: Widar:1-17; Aril: useless; CSI_301: 0-4")
    parser.add_argument('--train_location', default=None,
                        help="Setting the location (domain) configuration: Widar:1-5; Aril: 1-16; CSI_301: 0-1/2")
    parser.add_argument('--train_orientation',  default=None,
                        help="Setting the orientation (domain) configuration: Widar:1-5; Aril: useless; CSI_301:useless")
    parser.add_argument('--train_receiverid', default=None,
                        help="Selecting the Receiver: Widar:1-6; Aril: useless; CSI_301: useless")
    parser.add_argument('--train_sampleid', default=None,
                        help="Selecting samples: Widar:1-5; Aril: useless; CSI_301: useless")

    parser.add_argument('--test_roomid',  default=None,
                        help="Setting the room (domain) configuration: Widar:[1,2,3]; Aril:useless; CSI_301: [0,1]")
    parser.add_argument('--test_userid',  default=None,
                        help="Setting the user (domain) configuration: Widar:1-17; Aril: useless; CSI_301: 0-4")
    parser.add_argument('--test_location', default=None,
                        help="Setting the location (domain) configuration: Widar:1-5; Aril: 1-16; CSI_301: 0-1/2")
    parser.add_argument('--test_orientation',  default=None,
                        help="Setting the orientation (domain) configuration: Widar:1-5; Aril: useless; CSI_301:useless")
    parser.add_argument('--test_receiverid',  default=None,
                        help="Selecting the Receiver: Widar:1-6; Aril: useless; CSI_301: useless")
    parser.add_argument('--test_sampleid', default=None,
                        help="Selecting samples: Widar:1-5; Aril: useless; CSI_301: useless")

    parser.add_argument('--root', default="/your/data/path",
                        help="datasets path")
    parser.add_argument('--log_dir', default="./lighting_logs",
                        help="log path")
    parser.add_argument('--dataset', required=True,
                        choices=['aril', 'csi_301', 'widar'],
                        help="the dataset name: aril,csi_301,widar")
    parser.add_argument('--cross_type', required=True,
                        choices=['loc', 'user', 'env','ori','rx','in'],
                        help="the dataset name: aril,csi_301,widar")

    parser.add_argument('--data_shape', default=False,
                        help="the data shape: 1D, 2D, split for the three different models")
    parser.add_argument('--chunk_size', default=None,
                        help="setting the chunk size when using the 'split' data shape")
    parser.add_argument('--num_shot', default=None,
                        help="the number of samples in support set of each class.")
    parser.add_argument('--batch_size', default=None,
                        help="how much samples each class in one batch")
    parser.add_argument('--mode', default='amplitude',
                        help="phase,amplitude,None; useless in aril.")
    parser.add_argument('--align', default=False,
                        help="the series has fixed length or not")

    parser.add_argument('--model_name', default=None,
                        help="name of model:PrototypicalResNet,PrototypicalCnnLstmNet")

    parser.add_argument('--layers', default=None,
                        help="ResNet layer set: eg, [1,1,1,1]")
    parser.add_argument('--strides', default=None,
                        help="the strides of convolution: eg, [1,1,2,2]")
    parser.add_argument('--ResNet_inchannel', default=None,
                        help="input channel of ResNet1D")
    parser.add_argument('--groups', default=None,
                        help="convolutional groups")

    parser.add_argument('--in_channel_cnn', default=None,
                        help="the input channel of cnn encoder")
    parser.add_argument('--out_feature_dim_cnn', default=None,
                        help="the dimensions of cnn")
    parser.add_argument('--out_feature_dim_lstm', default=None,
                        help="the output dimension of lstm")
    parser.add_argument('--num_lstm_layer', default=None,
                        help="the number of lstm layers in LSTM")

    parser.add_argument('--width_mult', default=None,
                        help="the rate of width expansion in MobileNet2D")
    parser.add_argument('--MobileNet_inchannel', default=None,
                        help="the input channel of MobileNet2D")

    parser.add_argument('--metric_method', default=None,
                        help="metric_method")
    parser.add_argument('--num_class_linear_flag', default=None,
                        help="the number of categories and the flag of using linear or not")
    #xjw
    parser.add_argument('--num_domain_linear_flag', default=None,
                        help="the number of categories and the flag of using linear or not")

    parser.add_argument('--combine', default=False,
                        help="combine linear method with metric or not")

    parser.add_argument('--max_epochs', default=None, help="the max epoches")

    parser.add_argument('--class_feature_style', default=None, help="the max epoches")
    parser.add_argument('--domain_feature_style', default=None, help="the max epoches")
    parser.add_argument('--pn_style', default=None, help="the max epoches")

    args = parser.parse_args()

    setting =config[args.dataset]
    source_data_config = setting['source_data_config']
    target_data_config = setting['target_data_config']
    data_sample_config = setting['data_sample_config']
    encoder_config = setting['encoder_config']
    PrototypicalResNet_config = setting['PrototypicalResNet_config']
    PrototypicalCnnLstmNet_config = setting['PrototypicalCnnLstmNet_config']
    PrototypicalMobileNet_config = setting['PrototypicalMobileNet_config']
    max_epochs = setting['max_epochs']

    args.train_roomid = source_data_config['roomid']
    args.train_userid = source_data_config['userid']
    args.train_location = source_data_config['location']
    args.train_orientation = source_data_config['orientation']
    args.train_receiverid = source_data_config['receiverid']
    args.train_sampleid = source_data_config['sampleid']

    args.test_roomid = target_data_config['roomid']
    args.test_userid = target_data_config['userid']
    args.test_location = target_data_config['location']
    args.test_orientation = target_data_config['orientation']
    args.test_receiverid = target_data_config['receiverid']
    args.test_sampleid = target_data_config['sampleid']

    args.root = Path(data_sample_config['root'])
    args.dataset = data_sample_config['dataset']
    args.data_shape = data_sample_config['data_shape']
    args.chunk_size = data_sample_config['chunk_size']
    args.num_shot = data_sample_config['num_shot']
    args.batch_size = data_sample_config['batch_size']
    args.mode = data_sample_config['mode']
    args.align = data_sample_config['align']

    args.model_name = encoder_config['model_name']

    args.layers = PrototypicalResNet_config['layers']
    args.strides = PrototypicalResNet_config['strides']
    args.ResNet_inchannel = PrototypicalResNet_config['inchannel']
    args.groups = PrototypicalResNet_config['groups']

    args.in_channel_cnn = PrototypicalCnnLstmNet_config['in_channel_cnn']
    args.out_feature_dim_cnn = PrototypicalCnnLstmNet_config['out_feature_dim_cnn']
    args.out_feature_dim_lstm = PrototypicalCnnLstmNet_config['out_feature_dim_lstm']
    args.num_lstm_layer = PrototypicalCnnLstmNet_config['num_lstm_layer']

    args.width_mult = PrototypicalMobileNet_config['width_mult']
    args.MobileNet_inchannel = PrototypicalMobileNet_config['inchannel']

    exps = setting['exps']
    ex_repeat = setting['ex_repeat']
    for exp in exps:

        metric_config = exp['metric_config']
        style_config = exp['style_config']
        input_config = exp['input_config']
        args.metric_method = metric_config['metric_method']
        args.use_attention = metric_config['use_attention']
        args.num_class_linear_flag = metric_config['num_class_linear_flag']
        args.num_domain_linear_flag = metric_config['num_domain_linear_flag']
        args.combine = metric_config['combine']
        args.max_epochs = max_epochs
        args.class_feature_style = style_config['class_feature_style']
        args.domain_feature_style = style_config['domain_feature_style']
        args.pn_style = style_config['pn_style']
        args.ResNet_inchannel  = input_config['inchannel']
        args.mode  = input_config['mode']

        # print("Experiment Setting Config:{}".format(setting))

        #代码调整  class_feature_style  domain_feature_style
        if args.dataset == "widar":
            multi_exps_widar(args,ex_repeat)
        elif args.dataset == "csi_301":
            multi_exps_csida(args, ex_repeat)
        elif args.dataset == "aril":
            multi_exps_aril(args, ex_repeat)
            pass
        else:
            pass
        # print(args)

