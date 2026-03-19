"""
Configuration for the EEGNet MLSPred-Bench benchmark.

Benchmark IDs 1–12 correspond to different (SPH, SOP) pairs defined in
the MLSPred-Bench paper. Benchmark 12 (SPH=30 min, SOP=5 min) is the
default because it is the only one with train/valid/test splits available
in the repo.

Data layout (produced by mlspred_bench_v001.py):
    data/bmrkXX/train_values.hdf5   → key "tracings", shape (N, 1280, 20)
    data/bmrkXX/train_labels.csv    → N rows, values 0 or 1
    data/bmrkXX/valid_values.hdf5
    data/bmrkXX/valid_labels.csv
    data/bmrkXX/tests_values.hdf5   (only some benchmarks)
    data/bmrkXX/tests_labels.csv
"""

from dataclasses import dataclass, field
from typing import Optional


# SPH / SOP pairs for each benchmark (minutes)
BENCHMARK_PARAMS = {
    1:  (30,  5),
    2:  (30, 10),
    3:  (30, 15),
    4:  (30, 20),
    5:  (30, 25),
    6:  (30, 30),
    7:  (60,  5),
    8:  (60, 10),
    9:  (60, 15),
    10: (60, 20),
    11: (60, 25),
    12: (60, 30),
}


@dataclass
class DataConfig:
    data_dir: str = "data"          # relative to Seizure_Prediction/
    benchmark_id: int = 12          # which bmrk folder to use

    # EEG / window parameters (fixed by the benchmark)
    n_chans: int = 20
    sfreq: float = 256.0
    window_size_sec: float = 5.0    # 5 s × 256 Hz = 1280 samples

    @property
    def n_times(self) -> int:
        return int(self.window_size_sec * self.sfreq)  # 1280

    @property
    def bmrk_dir(self) -> str:
        return f"{self.data_dir}/bmrk{self.benchmark_id:02d}"

    @property
    def sph_min(self) -> int:
        return BENCHMARK_PARAMS[self.benchmark_id][0]

    @property
    def sop_min(self) -> int:
        return BENCHMARK_PARAMS[self.benchmark_id][1]


@dataclass
class ModelConfig:
    F1: int = 8
    D: int = 2
    F2: Optional[int] = None        # defaults to F1*D inside EEGNet
    kernel_length: int = 64         # ~0.25 s at 256 Hz
    depthwise_kernel_length: int = 16
    pool1_kernel_size: int = 4
    pool2_kernel_size: int = 8
    drop_prob: float = 0.25
    pool_mode: str = "mean"
    conv_spatial_max_norm: float = 1.0
    final_conv_length: str = "auto"
    final_layer_with_constraint: bool = False
    norm_rate: float = 0.25


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # "cosine" | "step" | "none"
    patience: int = 15              # early-stopping patience (epochs)
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "eegnet_bmrk12"


@dataclass
class BenchmarkConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
