"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.

deeply borrow from maskrcnn-benchmark and ST3D
"""

import pickle
import time

import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    print("utils/synchronize called")
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        print("new data")
        origin_size = data.size()
        # print("L70origin_size", origin_size)
        tensor = data.reshape(-1)
        # print("L72tensor_shape",tensor.shape)

    tensor_type = tensor.dtype
    print("L74tensor_type",tensor_type)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()]).to("cuda")
    # print("local_size",local_size)
    size_list = [torch.tensor([0]).to("cuda") for _ in range(world_size)]
    synchronize()
    dist.all_gather(size_list, local_size)
    synchronize()
    print("Gathered Size_list", size_list)
    size_list = [int(size.item()) for size in size_list]
    print("Size_list", size_list)
    max_size = max(size_list)
    # print("max_size",max_size)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)


    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def average_reduce_value(data):
    data_list = all_gather(data)
    return sum(data_list) / len(data_list)


def all_reduce(data, op="sum", average=False):

    def op_map(op):
        op_dict = {
            "SUM": dist.ReduceOp.SUM,
            "MAX": dist.ReduceOp.MAX,
            "MIN": dist.ReduceOp.MIN,
            "PRODUCT": dist.ReduceOp.PRODUCT,
        }
        return op_dict[op]

    world_size = get_world_size()
    if world_size > 1:
        reduced_data = data.clone()
        dist.all_reduce(reduced_data, op=op_map(op.upper()))
        if average:
            assert op.upper() == 'SUM'
            return reduced_data / world_size
        else:
            return reduced_data
    return data


# def gather_tensors(tensor,labels=False):
#     """
#     Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
#     dist.gather_all needs the gathered tensors to be of same size.
#     We get the sizes of the tensors first, zero pad them to match the size
#     Then gather and filter the padding

#     Args:
#         tensor: tensor to be gathered
#         labels: bool True if the tensor represents label information TODO:Deepika Remove this arg and make function tensor agnostic 
#     """
#     if labels:
#         assert tensor.ndim == 1,"labels should be of shape 1"
#     else:
#         assert tensor.ndim == 3,"features should be of shape N,1,256"

#     if dist.is_initialized(): # check if dist mode is initialized
#         # Determine sizes first
#         local_size = torch.tensor(tensor.size(), device=tensor.device)
#         WORLD_SIZE = dist.get_world_size()
#         all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]
#         dist.barrier() 
#         dist.all_gather(all_sizes,local_size)
#         dist.barrier()
        
#         # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
#         max_length = max([size[0] for size in all_sizes])
#         if max_length != local_size[0].item():
#             diff = max_length - local_size[0].item()
#             pad_size =[diff.item()] #pad with zeros 
#             if local_size.ndim >= 1:
#                 pad_size.extend(dimension.item() for dimension in local_size[1:])
#             padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
#             tensor = torch.cat((tensor,padding))
        
#         all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
#         dist.barrier()
#         dist.all_gather(all_tensors_padded,tensor)
#         dist.barrier()
#         gathered_tensor = torch.cat(all_tensors_padded)
#         if gathered_tensor.ndim == 1: # diff filtering mechanism for labels TODO:Deepika make this tensor agnostic
#             assert gathered_tensor.ndim == 1, "Label dimension should be N"
#             non_zero_mask = gathered_tensor > 0
#         else:
#             non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
#         gathered_tensor = gathered_tensor[non_zero_mask]
#         return gathered_tensor
#     else:
#         return tensor

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
