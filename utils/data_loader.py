import logging
import math
import random
import time

import numpy as np
import polars as pl
from torch.utils.data import Dataset


def read_trajsimi_traj_dataset(
    parquet_file,
    trip_id_col="TRIP_ID",
    polyline_col="TRAJ_MERCATOR",
    min_traj_len=20,
    max_traj_len=200,
    max_trajs=1200,
    train_ratio=0.7,
    eval_ratio=0.1,
    seed=42,
):
    logging.info("[trajsimi parquet traj dataset] Start loading.")
    _time = time.time()

    df = (
        pl.read_parquet(parquet_file)
        .select([trip_id_col, polyline_col])
        .filter(pl.col(polyline_col).list.len() >= min_traj_len)
        .with_columns(pl.col(polyline_col).list.head(max_traj_len))
    )

    rows = [
        (int(tid), [[float(p[0]), float(p[1])] for p in traj])
        for tid, traj in zip(df[trip_id_col].to_list(), df[polyline_col].to_list())
    ]

    if len(rows) == 0:
        logging.error("[trajsimi parquet traj dataset] no valid rows found")
        exit(200)

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    if max_trajs is not None and max_trajs > 0:
        rows = rows[:max_trajs]

    trajs = [r[1] for r in rows]

    pts = np.array([p for t in trajs for p in t])
    merc_range = [
        float(pts[:, 0].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].min()),
        float(pts[:, 1].max()),
    ]

    n = len(trajs)
    train_end = int(n * train_ratio)
    eval_end = int(n * (train_ratio + eval_ratio))
    trains = trajs[:train_end]
    evals = trajs[train_end:eval_end]
    tests = trajs[eval_end:]

    logging.info(
        "[trajsimi parquet traj dataset] Loaded. @={:.2f}. traj: #total={} (trains/evals/tests={}/{}/{})".format(
            time.time() - _time, n, len(trains), len(evals), len(tests)
        )
    )
    return trains, evals, tests, merc_range


class TrajSimiDatasetTraining(Dataset):
    def __init__(self, trains_traj, batchsize):
        self.trains_traj = trains_traj
        self.n = len(self.trains_traj)

        self.batchsize = batchsize
        self.niters = math.ceil((self.n / batchsize) ** 2)

    def __getitem__(self, index):
        sampled_idxs = random.sample(range(self.n), k=self.batchsize)
        trajs = [self.trains_traj[d_idx] for d_idx in sampled_idxs]

        return trajs, sampled_idxs

    def __len__(self):
        return self.niters
