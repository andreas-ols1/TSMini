import argparse
import logging
import os
import sys
import warnings

# Suppress pynvml deprecation warning triggered by torch.cuda
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

import torch

from .config import Config
from .task.trajsimi import TrajSimi
from .utils import tool_funcs


def parse_args():
    # dont set default values here! config.py is used to set default values.
    parser = argparse.ArgumentParser(description="train_trajsimi.py")
    parser.add_argument("--dumpfile_uniqueid", type=str, help="see config.py")
    parser.add_argument("--seed", type=int, help="")
    parser.add_argument("--dataset", type=str, help="")
    parser.add_argument(
        "--data-path",
        type=str,
        dest="data_path",
        help="Path to the input parquet file",
    )
    parser.add_argument("--trip_id_col", type=str, help="")
    parser.add_argument("--polyline_col", type=str, help="")
    parser.add_argument("--min_traj_len", type=int, help="")
    parser.add_argument("--max_traj_len", type=int, help="")
    parser.add_argument("--max_trajs", type=int, help="")
    parser.add_argument(
        "--simi_metric",
        type=str,
        choices=["dtw"],
        help="",
    )
    parser.add_argument("--tsmin_trans_attention_layer", type=int, help="")
    parser.add_argument("--trajsimi_loss_mse_weight", type=float, help="")
    parser.add_argument("--trajsimi_batch_size", type=int, help="")
    parser.add_argument("--log_dir", type=str, help="directory for log files")
    parser.add_argument(
        "--checkpoint_dir", type=str, help="directory for .pt checkpoints"
    )

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def main():
    enc_name = Config.trajsimi_encoder_name
    fn_name = Config.simi_metric
    metrics = tool_funcs.Metrics()

    task = TrajSimi()
    metrics.add(task.train())

    logging.info(
        "[EXPFlag]model={},dataset={},fn={},{}".format(
            enc_name, Config.dataset, fn_name, str(metrics)
        )
    )
    return


# nohup python train_trajsimi.py --dataset porto --simi_metric dtw &> result &
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    Config.update(parse_args())

    logging.basicConfig(
        level=logging.DEBUG if Config.debug else logging.INFO,
        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(Config.log_dir, tool_funcs.log_file_name()),
                mode="w",
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("python " + " ".join(sys.argv))
    logging.info("=================================")
    logging.info(Config.to_str())
    logging.info("=================================")

    main()
