config = {
    "aril":
        {
            'source_data_config': {'roomid': None, 'userid': None, 'location': None, 'orientation': None,
                                   'receiverid': None, 'sampleid': None, },
            'target_data_config': {'roomid': None, 'userid': None, 'location': None, 'orientation': None,
                                   'receiverid': None,
                                   'sampleid': None, },
            'data_sample_config': {'root': "/data/wifi/WiGr/", 'dataset': "aril", 'data_shape': '1D', 'chunk_size': 50,
                                   'num_shot': 3, 'batch_size': 5, 'mode': 'amplitude', 'align': True},
            'encoder_config': {'model_name': 'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
            'PrototypicalResNet_config': {'layers': [1, 1, 1], 'strides': [1, 2, 2], 'inchannel': 52, 'groups': 1, },
            'PrototypicalCnnLstmNet_config': {'in_channel_cnn': 3, 'out_feature_dim_cnn': 128,
                                              'out_feature_dim_lstm': 256,
                                              'num_lstm_layer': 2},
            'PrototypicalMobileNet_config': {'width_mult': 0.5, 'inchannel': 3, },
            'exps': [
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None}
                },
                # {
                #     'metric_config': {'metric_method': 'Euclidean', 'num_class_linear_flag': 6,
                #                       'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                #     'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                #                      'pn_style': None}
                # }
            ],

            'max_epochs': 150, 'ex_repeat': 1,
        },

    "csi_301":
        {
            'source_data_config': {'roomid': None, 'userid': None, 'location': None, 'orientation': None,
                                   'receiverid': None,
                                   'sampleid': None, },
            'target_data_config': {'roomid': [1], 'userid': None, 'location': None, 'orientation': None,
                                   'receiverid': None,
                                   'sampleid': None, },
            'data_sample_config': {'root': "/data/wifi/WiGr", 'dataset': "csi_301", 'data_shape': '1D',
                                   'chunk_size': 50, 'num_shot': 1, 'batch_size': 5, 'mode': None,
                                   'align': True},
            'encoder_config': {'model_name': 'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
            'PrototypicalResNet_config': {'layers': [1, 1, 1], 'strides': [1, 2, 2], 'inchannel': None, 'groups': 3, },
            'PrototypicalCnnLstmNet_config': {'in_channel_cnn': 3, 'out_feature_dim_cnn': 256,
                                              'out_feature_dim_lstm': 512,
                                              'num_lstm_layer': 3},
            'PrototypicalMobileNet_config': {'width_mult': 1.0, 'inchannel': 3, },
            'exps': [
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config': {"mode": 'phase', 'inchannel': 342}
                },
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config': {"mode": 'amplitude', 'inchannel': 342}
                },
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config': {"mode": 'pha&amp', 'inchannel': 684}
                },
            ],
            'max_epochs': 150, 'ex_repeat': 1,
        },

    "widar":
        {
            'source_data_config': {'roomid': [1], 'userid': None, 'location': None, 'orientation': [2],
                                   'receiverid': [1],
                                   'sampleid': None, },
            'target_data_config': {'roomid': [1], 'userid': None, 'location': None, 'orientation': None,
                                   'receiverid': None,
                                   'sampleid': None, },
            'data_sample_config': {'root': "/data/wifi/WiGr", 'dataset': "widar", 'data_shape': '1D',
                                   'chunk_size': 50, 'num_shot': 3, 'batch_size': 5, 'mode': None,
                                   'align': True},
            'encoder_config': {'model_name': 'PrototypicalResNet'},  # PrototypicalCnnLstmNet,PrototypicalMobileNet
            'PrototypicalResNet_config': {'layers': [1, 1, 1], 'strides': [1, 2, 2], 'inchannel': None, 'groups': 3, },
            'PrototypicalCnnLstmNet_config': {'in_channel_cnn': 3, 'out_feature_dim_cnn': 256,
                                              'out_feature_dim_lstm': 512,
                                              'num_lstm_layer': 3},
            'PrototypicalMobileNet_config': {'width_mult': 1.0, 'inchannel': 3, },
            'exps': [
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config':{"mode":'phase','inchannel':90}
                },
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config': {"mode": 'amplitude', 'inchannel': 90}
                },
                {
                    'metric_config': {'metric_method': 'cosine', 'num_class_linear_flag': 6,
                                      'num_domain_linear_flag': None, 'combine': False, 'use_attention': True, },
                    'style_config': {'class_feature_style': "style", 'domain_feature_style': "gesture",
                                     'pn_style': None},
                    'input_config': {"mode": 'pha&amp', 'inchannel': 180}
                },

            ],
            'max_epochs': 150, 'ex_repeat': 1,
        },
}
