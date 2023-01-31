from dataclasses import dataclass
from typing import Any, List

@dataclass
class ReproducOpt:    
    seed: int
    benchmark: bool
    deterministic: bool
    
@dataclass
class NetworkOpt:
    name: str
    network_param: Any
    
@dataclass
class TrainOpt:
  batch_size: int
  total_epoch: int
  time_window: int
    
    
@dataclass
class TsGAEopt:
    dir_name: str
    task_name: str
    optimizer: Any
    reproduc: ReproducOpt
    network: NetworkOpt
    train: TrainOpt
    log: Any
    causal_thres: str
    
@dataclass
class CUTSopt:
    dir_name: str
    task_name: str
    
    @dataclass
    class CUTSargs:
        n_nodes: int
        input_step: int
        batch_size: int
        data_dim: int
        total_epoch: int
        update_every: int
        show_graph_every: int
        
        @dataclass
        class data_pred:
            model: str
            multi_scale: bool
            multi_scale_periods: list
            pred_step: int
            mlp_hid: int
            mlp_layers: int
            lr_data_start: float
            lr_data_end: float
            weight_decay: int

        @dataclass
        class graph_discov:
            lambda_s: 0.1
            lr_graph_start: float
            lr_graph_end: float
            start_tau: 0.3
            end_tau: 0.01
            dynamic_sampling_milestones: list
            dynamic_sampling_periods: list

    causal_thres: str
    reproduc: ReproducOpt
    log: Any