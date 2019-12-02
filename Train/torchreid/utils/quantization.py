import math

import torch
import torch.nn as nn
import json
import os


def int_bits(input, percent=0):
    if percent == 0:
        max_v = input.abs().max().item()
    else:
        sorted_items, _ = torch.sort(input.abs().view(-1), descending=True)
        max_v = sorted_items[int(sorted_items.shape[0] * percent)]
    return math.ceil(math.log(max_v, 2))


def quantize_hook(module, input, output, bits, int_input_bits, accum_bits, module_name):
    if isinstance(module, nn.Conv2d):
        act_int_bits = int_bits(output)

        if module_name == 'features.4.expand1x1':  #  to have the same int bits as expand3x3 because they are concatenated
            act_int_bits = 6
        elif module_name == 'features.6.expand1x1':
            act_int_bits = 7
        elif module_name == 'features.9.expand1x1':
            act_int_bits = 9
        elif module_name == 'features.10.expand1x1':
            act_int_bits = 9
        elif module_name == 'features.11.expand1x1':
            act_int_bits = 9
        elif module_name in ['features.0.my_modules.0', 'features.0.my_modules.1', 'features.0.my_modules.3']:
            act_int_bits = 5

        w_int_bits = int_bits(module.weight)
        w_float_bits = bits - w_int_bits - 1

        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input

        old_act_float_bits = inp.dot_place if hasattr(inp, 'dot_place') else bits - int_input_bits - 1

        if module_name == 'features.4.squeeze': # there is concat and we dont know dot_place
            old_act_float_bits = 9
        elif module_name == 'features.6.squeeze':
            old_act_float_bits = 9
        elif module_name == 'features.7.squeeze':
            old_act_float_bits = 8
        elif module_name == 'features.9.squeeze':
            old_act_float_bits = 7
        elif module_name == 'features.10.squeeze':
            old_act_float_bits = 6
        elif module_name == 'features.11.squeeze':
            old_act_float_bits = 6
        elif module_name == 'features.12.squeeze':
            old_act_float_bits = 6


        print(module_name)
        print('out act int bits={}, w_float_bits={}, input_act_float_bits={}'.format(act_int_bits, w_float_bits,
                                                                               old_act_float_bits))
        if act_int_bits + w_float_bits + old_act_float_bits > accum_bits - 1:
            w_float_bits -= act_int_bits + w_float_bits + old_act_float_bits - accum_bits + 1

        #w_float_bits -= 7
        print('final float bits=', w_float_bits)
        module.norm = max(0, act_int_bits + w_float_bits + old_act_float_bits - bits + 1)

        module.max_act = output.abs().max().item()
        output.dot_place = w_float_bits + old_act_float_bits - module.norm
        print('Norm={}, dot_place={}'.format(module.norm, output.dot_place))
        module.dot_place = output.dot_place
        module.weight.data = integerize(module.weight.data, w_float_bits, bits)
        module.weights = nn.ParameterList([nn.Parameter(module.weight.data[:, i, :, :].unsqueeze_(1)) for i in range(module.weight.shape[1])])
        module.bias.data = integerize(module.bias.data, bits - act_int_bits - 1, bits)
        module.w_float_bits = w_float_bits

    elif isinstance(module, nn.ReLU) or isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AdaptiveAvgPool2d):
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        if not hasattr(output, 'dot_place'):
            dot_place = inp.dot_place if hasattr(inp, 'dot_place') else bits - int_input_bits - 1
            output.dot_place = dot_place


def save_dot_place_hook(module, input, output, bits, int_input_bits):
    if isinstance(module, nn.Conv2d):
        output.dot_place = module.dot_place
    else:
        if isinstance(input, tuple):
            inp = input[0]
        else:
            inp = input
        dot_place = inp.dot_place if hasattr(inp, 'dot_place') else bits - int_input_bits - 1
        output.dot_place = dot_place


def integerize(input, float_bits, bits=16):
    bound = math.pow(2.0, bits - 1)
    min_val = -bound
    max_val = bound - 1
    res = torch.floor_(input * math.pow(2., float_bits) + 0.5)
    if (res > max_val).any() or (res < min_val).any():
        print('Overflow. Some values were clipped')
    return torch.clamp(res, min_val, max_val)


def roundnorm_reg(input, n):
    return torch.floor_((input + math.pow(2., n-1)) * math.pow(2., -n))


def gap8_clip(input, bits):
    return torch.clamp(input, -math.pow(2., bits), math.pow(2., bits) - 1)


def round(input, float_bits, bits=16):
    bound = math.pow(2.0, bits - 1)
    min_val = -bound
    max_val = bound - 1
    return torch.clamp(torch.floor_(input * math.pow(2., float_bits) + 0.5), min_val, max_val) * math.pow(2., -float_bits)


def collect_stats(model, loader, use_gpu):
    model.eval()

    for batch_idx, (imgs, pids, _, _) in enumerate(loader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        _ = model(imgs)
        break


def save_quantized_weights(module, name, folder):
    os.makedirs(folder, exist_ok=True)
    if isinstance(module, nn.Conv2d):
        params = {'norm': module.norm,
                  'dot_place': module.dot_place,
                  'weight': module.weight.data.cpu().tolist(),
                  'bias': module.bias.data.cpu().tolist()}

        with open(os.path.join(folder, name + '.json'), 'w') as f:
            json.dump(params, f)


def save_act_hook(module, input, output, save_dir, name):
    os.makedirs(os.path.join(save_dir, name), exist_ok=True)
    with open(os.path.join(save_dir, name, 'input.json'), 'w') as inp_f, \
            open(os.path.join(save_dir, name, 'output.json'), 'w') as out_f:
        if isinstance(input, tuple):
            if len(input) == 1:
                input = input[0]
            else:
                print('Tuple has length = ', len(input), 'for module ', name)
        np_input = input.data.cpu().numpy()
        if len(np_input.shape) == 4:
            np_input = np_input.transpose((0, 2, 3, 1))
        json.dump(np_input.tolist(), inp_f)
        np_output = output.data.cpu().numpy()
        if len(np_output.shape) == 4:
            np_output = np_output.transpose((0, 2, 3, 1))
        json.dump(np_output.tolist(), out_f)
