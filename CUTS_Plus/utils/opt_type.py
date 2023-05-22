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
class MultiCADopt:
    dir_name: str
    task_name: str
    
    @dataclass
    class MultiCADargs:
        n_nodes: int
        input_step: int
        window_step: int
        stride: int
        batch_size: int
        sample_per_epoch: int
        data_dim: int
        total_epoch: int
        
        patience: int
        warmup: Any
        
        show_graph_every: int
        val_every: int
        
        n_groups: int
        group_policy: Any
        causal_thres: str
        
        @dataclass
        class data_pred:
            model: str
            merge_policy: str
            lr_data_start: float
            lr_data_end: float
            weight_decay: int
            prob: bool

        @dataclass
        class graph_discov:
            lr_graph_start: float
            lr_graph_end: float
            lambda_s_start: float
            lambda_s_end: float
            tau_start: float
            tau_end: float
            disable_bwd: bool
            separate_bwd: bool
            disable_ind: bool
            disable_graph: bool
            use_true_graph: bool
    
    reproduc: ReproducOpt
    log: Any