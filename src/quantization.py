from collections import OrderedDict

import numpy as np
import torch

from model import ShallowNetwork


def quantize_tensor_as_numpy(x, num_bits=8):
    if len(x.shape) == 1 and x.shape[0] == 1:
        v = np.float32(x[0]).tobytes()
        return np.frombuffer(v, dtype=np.uint8)

    qmin = 0.
    qmax = 2. ** num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    cast_scale = np.float32(scale)
    b_scale = cast_scale.tobytes()

    initial_zero_point = (qmin - min_val) / scale

    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    q_x = zero_point + (x / scale)
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    n_q_x = q_x.numpy()
    n_q_x = np.append(n_q_x, zero_point)
    for bv in b_scale:
        n_q_x = np.append(n_q_x, bv)

    return n_q_x.astype(np.uint8)


def log_model(model, logger=None):
    if logger:
        logger.info(get_state(model))
        return
    print(get_state(model))


def uncompress_8(ar):
    zero_point = ar[-5].astype(np.float)
    b_scale = ar[-4:].tobytes()
    scale = np.frombuffer(b_scale, dtype=np.float32)
    # print("zp={}, scale={}".format(zero_point, scale))
    ret_ar = ar[0:3]
    ret_ar = ret_ar.astype(np.float)
    return scale * (ret_ar - zero_point).astype(np.float64)


def uncompress_6(ar):
    value = np.frombuffer(ar, dtype=np.float32)
    # print("bias={}".format(scale))
    return value.astype(np.float64)


def decompress(ar, bias=False, non_linearity="tanh", scrap=False):
    if not bias:
        if scrap:
            model = ShallowNetwork(3)
            model.linear_1.weight = torch.nn.Parameter(torch.zeros(3, 1))
            model.linear_2.weight = torch.nn.Parameter(torch.zeros(1, 3))
        else:
            model = ShallowNetwork(3)
            model.linear_1.weight = torch.nn.Parameter(torch.from_numpy(uncompress_8(ar[:8])).reshape(3, 1))
            model.linear_2.weight = torch.nn.Parameter(torch.from_numpy(uncompress_8(ar[8:])).reshape(1, 3))
        return model
    if scrap:
        model = ShallowNetwork(3, bias=bias, non_linearity=non_linearity)
        model.linear_1.weight = torch.nn.Parameter(torch.zeros(3, 1))
        model.linear_1.bias = torch.nn.Parameter(torch.zeros(1, 3))
        model.linear_2.weight = torch.nn.Parameter(torch.zeros(1, 3))
        model.linear_2.bias = torch.nn.Parameter(torch.zeros(1, 1))

    else:
        model = ShallowNetwork(3, bias=bias, non_linearity=non_linearity)
        model.linear_1.weight = torch.nn.Parameter(torch.from_numpy(uncompress_8(ar[:8])).reshape(3, 1))
        model.linear_1.bias = torch.nn.Parameter(torch.from_numpy(uncompress_8(ar[8:16])).reshape(1, 3))
        model.linear_2.weight = torch.nn.Parameter(torch.from_numpy(uncompress_8(ar[16:24])).reshape(1, 3))
        model.linear_2.bias = torch.nn.Parameter(torch.from_numpy(uncompress_6(ar[24:])).reshape(1, 1))

    return model


def compress(model):
    state = get_state(model)
    i = 0
    data = []
    for n, p in state.items():
        q_v = quantize_tensor_as_numpy(p)
        data.append(q_v.ravel())
        i += q_v.shape[0]

    arr = np.zeros([i], dtype=np.int8).astype(np.uint8)
    i = 0
    for np_ar in data:
        l = np_ar.shape[0]
        arr[i:i + l] = np_ar
        i += l
    #print("Quantize model in array={}, #Bytes={}".format(arr, arr.nbytes))
    return arr


def get_state(model):
    state = OrderedDict()
    for n, p in model.state_dict().items():
        state[n] = p
    return state


def main():
    torch.set_default_tensor_type(torch.DoubleTensor)

    print('-' * 89)
    print("Initial Shallow Model No Bias")
    print('-' * 89)
    model = ShallowNetwork(3, bias=False)
    log_model(model)

    print('-' * 89)
    print("Quantization with numpy")
    print('-' * 89)
    ar = compress(model)
    model = decompress(ar, bias=False, scrap=False)
    log_model(model)
    print('-' * 89)

    print('-' * 89)
    print("Initial Shallow Model")
    print('-' * 89)
    model = ShallowNetwork(3, bias=True)
    log_model(model)

    print('-' * 89)
    print("Quantization with numpy")
    print('-' * 89)
    ar = compress(model)
    model = decompress(ar, bias=True, scrap=False)
    log_model(model)
    print('-' * 89)


if __name__ == "__main__":
    main()
