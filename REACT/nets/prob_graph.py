import torch
from nets.gumbel_softmax import gumbel_softmax


def split_dy_st(graph, dy_feat_num, st_feat_num, batch_dim=False, time_dim=False):
    if batch_dim == True and time_dim == True:
        assert dy_feat_num + st_feat_num == graph.shape[2], "graph shape error"
        dy_graph = graph[:, :, :dy_feat_num]
        st_graph = graph[:, 0, dy_feat_num:]
    elif batch_dim == True and time_dim == False:
        assert dy_feat_num + st_feat_num == graph.shape[1], "graph shape error"
        dy_graph = graph[:, :dy_feat_num]
        st_graph = graph[:, dy_feat_num:]
    elif batch_dim == False and time_dim == True:
        assert dy_feat_num + st_feat_num == graph.shape[1], "graph shape error"
        dy_graph = graph[:, :dy_feat_num]
        st_graph = graph[0, dy_feat_num:]
    elif batch_dim == False and time_dim == False:
        assert dy_feat_num + st_feat_num == graph.shape[0], "graph shape error"
        dy_graph = graph[:dy_feat_num]
        st_graph = graph[dy_feat_num:]
    else:
        raise NotImplementedError
    return dy_graph, st_graph

def cumulative_time_graph(prob_graph, t_length, cumu_type="prod", batch_dim=False):
    if not batch_dim:
        prob_graph = prob_graph[None, :, :]
    
    b, l, n = prob_graph.shape
    assert t_length % l == 0, "t_length should be divisible by time_chunk_n"
    chunk_size = t_length // l
    cum_graph = torch.zeros([b, t_length, n], device=prob_graph.device, dtype=prob_graph.dtype)
    for li in range(l):
        if cumu_type == "prod":
            chunk_prob = torch.prod(prob_graph[:, :li+1], dim=1, keepdim=True)
        elif cumu_type == "sum":
            chunk_prob = torch.sum(prob_graph[:, li:], dim=1, keepdim=True)
        cum_graph[:, li*chunk_size:(li+1)*chunk_size] = chunk_prob.repeat(1, chunk_size, 1)
    
    cum_graph = torch.clip(cum_graph, 0, 1)
    if not batch_dim:
        cum_graph = cum_graph[0]
    
    return cum_graph

def bernonlli_sample(theta, batch_size, prob=True, hard_mask=False, t_length=None, time_cumu_type="prod", time_cumulative=False):
    prob_graph = torch.sigmoid(theta)
    if len(theta.shape) == 1:
        # 不包含时间维度，n对1的图
        sample_matrix = prob_graph[None, :].expand(batch_size, -1)
        if hard_mask:
            sample_matrix = (sample_matrix > 0.5).int()
            return sample_matrix
        else:
            if prob:
                return torch.bernoulli(sample_matrix)
            else:
                sample_matrix[sample_matrix < 1e-3] = torch.zeros_like(sample_matrix[sample_matrix < 1e-3])
                return sample_matrix
    elif len(theta.shape) == 2:
        # 包含时间维度，或者是n对n的图
        if time_cumulative:
            prob_graph = cumulative_time_graph(prob_graph, t_length, cumu_type=time_cumu_type)
        sample_matrix = prob_graph[None, :, :].expand(batch_size, -1, -1)
        if hard_mask:
            sample_matrix = (sample_matrix > 0.5).int()
            return sample_matrix
        else:
            if prob:
                return torch.bernoulli(sample_matrix)
            else:
                sample_matrix[sample_matrix < 1e-3] = torch.zeros_like(sample_matrix[sample_matrix < 1e-3])
                return sample_matrix
    else:
        raise NotImplementedError
        

def gumbel_sample(theta, batch_size, tau=1, t_length=None, time_cumu_type="prod", time_cumulative=False, batch_dim=False, time_dim=True, causalgraph_2d=False):
    prob_graph = torch.sigmoid(theta)
    if len(theta.shape) == 1:
        # 不包含时间维度，n对1的图
        assert batch_dim == False and time_dim == False and causalgraph_2d == False, "graph shape error"
        prob = prob_graph[None, :, None].expand(batch_size, -1, -1)
        logits = torch.concat([prob, (1 - prob)], axis=-1)
        samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
        
    elif len(theta.shape) == 2:
        # 包含时间维度，或者是n对n的图
        if time_cumulative:
            assert (batch_dim == False and time_dim == True and causalgraph_2d == False), "graph shape error"
            prob_graph = cumulative_time_graph(prob_graph, t_length, cumu_type=time_cumu_type, batch_dim=batch_dim)
        else:
            assert (batch_dim == False and time_dim == False and causalgraph_2d == True) or \
                (batch_dim == True and time_dim == False and causalgraph_2d == False), "graph shape error"
        if batch_dim:
            prob = prob_graph[:, :, None]
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, 0]
        else:
            prob = prob_graph[None, :, :, None].expand(batch_size, -1, -1, -1)
            logits = torch.concat([prob, (1 - prob)], axis=-1)
            samples = gumbel_softmax(logits, tau=tau)[:, :, :, 0]
    
    elif len(theta.shape) == 3:
        # 包含时间维度，并且包含batchsize
        assert (batch_dim == True and time_dim == True and causalgraph_2d == False), "graph shape error"
        if time_cumulative:
            prob_graph = cumulative_time_graph(prob_graph, t_length, cumu_type=time_cumu_type, batch_dim=batch_dim)
        prob_graph = prob_graph[:, :, :, None]
        logits = torch.concat([prob_graph, (1 - prob_graph)], axis=-1)
        samples = gumbel_softmax(logits, tau=tau)[:, :, :, 0]
    else:
        raise NotImplementedError
    
    return samples, prob_graph
        



def freeze_graph(args, theta):
    prob_graph = torch.sigmoid(theta)
    dy_graph, st_graph = (
        prob_graph[: args.dy_feat_num],
        prob_graph[args.dy_feat_num :],
    )

    if isinstance(args.graph_discov.freeze_dy_thres, str):
        top_n = int(args.graph_discov.freeze_dy_thres.split("_")[1])
        dy_thres = torch.quantile(dy_graph, 1 - top_n / args.dy_feat_num)
    else:
        dy_thres = args.graph_discov.freeze_dy_thres

    if isinstance(args.graph_discov.freeze_st_thres, str):
        top_n = int(args.graph_discov.freeze_st_thres.split("_")[1])
        st_thres = torch.quantile(st_graph, 1 - top_n / args.st_feat_num)
    else:
        st_thres = args.graph_discov.freeze_st_thres

    dy_graph = (dy_graph > dy_thres).float() * 200 - 100
    st_graph = (st_graph > st_thres).float() * 200 - 100

    theta = torch.concat([dy_graph, st_graph], axis=0)

    args.data_pred.prob = False
    args.data_pred.hard_mask = True
    delattr(args, "graph_discov")
    print("Freezing graph!")
    
    return theta