from nets.gc_pred_nets import *
from nets.loss import *


def build_net(args, device, mode="pred"):
    # mode = "pred" or "discov"
    # 当pred模式时，图是一维的；当discov模式时，图是二维的

    if args.data_pred.model == "multitask_lstm":
        fitting_model = MultitaskLSTM(
            args.dy_feat_num * args.dy_dim,
            args.st_feat_num * args.st_dim,
            args.data_pred.mlp_hid,
            args.data_pred.pred_dim,
            args.data_pred.mlp_layers,
            args.data_pred.dropout,
        ).to(device)
    elif args.data_pred.model == "multitask_mlp":
        fitting_model = MultitaskMLP(
            args.dy_feat_num * args.dy_dim * args.t_length,
            args.st_feat_num * args.st_dim,
            args.data_pred.mlp_hid,
            args.data_pred.pred_dim,
            args.data_pred.mlp_layers,
            args.data_pred.dropout,
        ).to(device)
    elif args.data_pred.model == "singletask_mlp":
        fitting_model = MaskedMLP(
            args.dy_feat_num * args.dy_dim * args.t_length,
            args.data_pred.mlp_hid,
            args.data_pred.pred_dim,
            args.data_pred.mlp_layers,
            dropout=args.data_pred.dropout,
            act=args.data_pred.act,
        ).to(device)
    elif args.data_pred.model == "multitask_transformer":
        fitting_model = MultitaskTransformer(
            args.dy_feat_num,
            args.st_feat_num,
            args.dy_dim,
            args.st_dim,
            args.data_pred.mlp_hid,
            args.data_pred.pred_dim,
            args.data_pred.mlp_layers,
            dropout=args.data_pred.dropout,
            local_sample_type=args.local_expl.sample_type,
            local_expl=args.local_expl.enable,
            local_sigma=args.local_expl.sigma,
            local_time_cumu_type=args.local_expl.time_cumu_type,
            local_time_chunk_num=args.local_expl.time_chunk_num,
        ).to(device)
    else:
        raise NotImplementedError

    if mode == "pred":
        if hasattr(args, "time_graph") and args.time_graph.enable:
            graph = nn.Parameter(
                torch.ones([args.time_graph.time_chunk_num, args.dy_feat_num + args.st_feat_num]).to(device)
            )
            graph.data[0, :] = 0
            graph.data[1:, :] = 3
        else:
            graph = nn.Parameter(
                torch.ones([args.dy_feat_num + args.st_feat_num,]).to(device) * 0
            )
            
    elif mode == "discov":
        # 当前只支持动态特征的图
        if hasattr(args, "disable_graph") and args.disable_graph:
            print("Using full graph and disable graph discovery...")
            graph = nn.Parameter(torch.ones([args.dy_feat_num, args.dy_feat_num]).to(device) * 1000)
        else:
            graph = nn.Parameter(torch.ones([args.dy_feat_num, args.dy_feat_num]).to(device) * 0)

    # graph = nn.Parameter(torch.zeros([args.dy_feat_num, args.dy_feat_num, args.input_step]).to(device))
    return fitting_model, graph


def build_optim(fitting_model, args, total_epoch):
    if isinstance(fitting_model, nn.Module):
        parameters = fitting_model.parameters()
    elif isinstance(fitting_model, nn.Parameter):
        parameters = [fitting_model]
    else:
        raise NotImplementedError

    if args.optim == "adam":
        data_pred_optimizer = torch.optim.Adam(parameters, lr=args.lr_start, weight_decay=args.weight_decay)
    elif args.optim == "sgd":
        data_pred_optimizer = torch.optim.SGD(parameters, lr=args.lr_start, momentum=0.9)
    else:
        raise NotImplementedError

    if args.scheduler == "cos":
        data_pred_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(data_pred_optimizer, T_max=total_epoch // 5)  # * iters
    elif args.scheduler == "step":
        gamma = (args.lr_end / args.lr_start) ** (1 / total_epoch)
        data_pred_scheduler = torch.optim.lr_scheduler.StepLR(data_pred_optimizer, step_size=1, gamma=gamma)
    else:
        raise NotImplementedError

    return data_pred_optimizer, data_pred_scheduler


def build_loss(name):
    if name == "ce" or name == "crossentropy":
        data_pred_loss = nn.CrossEntropyLoss()
    elif name == "multitask_ce":
        data_pred_loss = Multitask_CE(base_loss="focalloss")
    elif name == "local_expl_regularizer_gauss":
        data_pred_loss = Gauss_Gate_Regularizer_Gauss(sigma=0.5)
    elif name == "local_expl_regularizer_l1":
        data_pred_loss = L1_Regularizer()
    else:
        raise NotImplementedError
    return data_pred_loss
