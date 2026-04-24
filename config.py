import os
import random

import numpy
import torch


def set_seed(seed=42):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Config:
    debug = False
    dumpfile_uniqueid = ""
    seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10]  # dont use os.getcwd()
    project_root = os.path.normpath(os.path.join(root_dir, "../../"))
    checkpoint_dir = os.path.join(project_root, "models", "tsmini")
    log_dir = os.path.join(project_root, "logs", "tsmini")

    n_workers = None  # None → multiprocessing.cpu_count() at runtime

    dataset = "porto"
    data_path = ""
    trip_id_col = "TRIP_ID"
    polyline_col = "TRAJ_MERCATOR"
    max_trajs = 1200
    train_ratio = 0.7
    eval_ratio = 0.1
    simi_metric = "dtw"  # dtw, frechet, hausdorff, edr, lcss, erp, sspd

    # ===========general=============
    min_traj_len = 20
    max_traj_len = 200
    cell_size = 100.0
    cellspace_buffer = 500.0

    traj_distance_norm_denominator = 1000

    # ===========TSMini=============
    seq_embedding_dim = 128

    tsmini_conv_channel_dim = 64
    tsmini_conv_hidden_in_dim = 16
    tsmini_conv_kernel_size = 3
    tsmini_conv_stride = 1
    tsmini_patch_emb_dim = 128  # same to seq_embedding_dim

    tsmin_trans_attention_head = 4
    tsmin_trans_attention_dropout = 0.1
    tsmin_trans_attention_layer = 1
    tsmin_trans_hidden_dim = 512

    # ===========trajsimi=============
    trajsimi_encoder_name = "TSMini"
    trajsimi_loss_mse_weight = 0.2

    trajsimi_batch_size = 128  # 128
    trajsimi_epoch = 40
    trajsimi_training_bad_patience = 5
    trajsimi_learning_rate = 0.002  # 0.0001 # 0.0002
    trajsimi_training_lr_degrade_step = 15  # 5
    trajsimi_training_lr_degrade_gamma = 0.5
    trajsimi_learning_weight_decay = 0.00001

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()

    @classmethod
    def post_value_updates(cls):
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)

        if cls.data_path == "":
            cls.data_path = os.path.join(
                cls.root_dir,
                "../../data/samples/porto/porto_sample_N4273.parquet",
            )
        set_seed(cls.seed)

    @classmethod
    def to_str(cls):  # __str__, self
        dic = cls.__dict__.copy()
        lst = list(
            filter(
                lambda p: (not p[0].startswith("__")) and type(p[1]) != classmethod,
                dic.items(),
            )
        )
        return "\n".join([str(k) + " = " + str(v) for k, v in lst])
