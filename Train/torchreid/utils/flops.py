import torch
import numpy as np
import copy

IGNORE_MODULE = (0, 0, 0, 0, 0)

def _conv2d_flops(module):
    # unpacking module
    # out_channels, in_channels, K1, K2 = module.weight.data.size()
    # B, C_in, H, W = module.input_size
    # pad_H, pad_W = module.padding
    # use_bias = module.bias
    # stride_H, stride_W = module.stride
    #
    # num_filters = out_channels
    #
    # # sub-calculations for weight multiplications
    # # assuming im2col is used for convolution implementation
    # #
    # # see https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
    # # for more info.
    # filter_size = in_channels * K1 * K2
    # if use_bias is not None:
    #     filter_size += 1
    #
    # num_convs_along_H = (H - K1 + 2*pad_H + 1) / stride_H
    # num_convs_along_W = (W - K2 + 2*pad_W + 1) / stride_W
    # total_convs = num_convs_along_H * num_convs_along_W
    #
    # # calculate total FLOPs
    # mult_flops = filter_size * total_convs * num_filters
    # B, C_out, H_out, W_out = module.output_size
    # mult_flops_new = filter_size * C_out * H_out * W_out
    # assert mult_flops == mult_flops_new
    # add_flops = mult_flops
    # comp_flops = 0
    # activation_size = (K1 + (K1 - 1)) * H * in_channels
    # weight_size = out_channels * in_channels * K1 * K2
    out_channels, in_channels, K1, K2 = module.weight.data.size()
    B, C_out, H_out, W_out = module.output_size
    B, C_in, H, W = module.input_size
    filter_size = in_channels * K1 * K2
    if module.bias is not None:
        filter_size += 1

    mult_flops = filter_size * C_out * H_out * W_out
    add_flops = mult_flops
    comp_flops = 0
    weight_size = np.product(module.weight.data.size())
    activation_size = (K1 + (K1 - 1)) * H * in_channels #   ??????
    return mult_flops, add_flops, comp_flops, activation_size, weight_size

def _relu_flops(module):
    comp_flops = np.product(module.input_size)
    return 0, 0, comp_flops, 0, 0

def _relu6_flops(module):
    comp_flops = 2 * np.product(module.input_size)
    return 0, 0, comp_flops, 0, 0

def _linear_flops(module):
    """
    linear layer is a 2d matmul:
       out = (B, D) x (D, O)

    out is now of shape (B, O)
    """
    B, D = module.input_size
    D, O = module.weight.data.size()
    mult_flops = D * O * B
    add_flops = mult_flops
    comp_flops = 0
    return mult_flops, add_flops, comp_flops, 0, 0

def _bn_flops(module):
    mult_flops = np.product(module.input_size)
    add_flops = mult_flops
    comp_flops = 0
    return mult_flops, add_flops, comp_flops, 0, 0

def calculate_flops(name, module):
    """
    mults: multiplications
    adds: additions
    comps: comparisons
    """
    if not hasattr(module, 'input_size'):
        # the module exists in the network, but its forward()
        # is never called, so ignore its FLOPS.
        return IGNORE_MODULE
    elif name == 'Conv2d':
        return _conv2d_flops(module)
    elif name == 'Linear':
        return _linear_flops(module)
    elif name == 'ReLU':
        return _relu_flops(module)
    elif name == 'ReLU6':
        return _relu6_flops(module)
    elif 'BatchNorm' in name:
        return _bn_flops(module)
    else:
        print('Layer {} was ignored during FLOPs computing!'.format(name))
        return IGNORE_MODULE

def calculate_params(name, module):
    val = 0.
    if hasattr(module, '_parameters'):
        for param in module._parameters.values():
            if param is not None:
                val += np.product(param.data.shape)
        if name == 'BatchNorm2d':
            val += np.product(module.running_mean.shape)
            val += np.product(module.running_var.shape)
    return val

def get_num_gen(gen):
    return sum(1 for x in gen)

def get_modules(gen):
    modules = []
    for child in gen:
        num_children = get_num_gen(child.children())

        # leaf node
        if num_children == 0:
            modules.append((child.__class__.__name__, child, str(child)))
        else:
            modules += get_modules(child.children())
    return modules

def get_input_size(self, input, output):
    if isinstance(input[0], tuple):
        input = input[0]
    self.input_size = input[0].data.size()
    self.output_size = output.data.size()

def calc_flops(orig_net, input_height, input_width, ignore_linear=True, layerwise=False, weight_bits=32, activation_bits=32):
    net = copy.deepcopy(orig_net)

    net = net.cpu()
    net.eval()

    # prepare input
    x = torch.FloatTensor(np.random.random((1, 3, input_height, input_width)))

    modules = get_modules(net.children())
    for name, module, _ in modules:
        module.register_forward_hook(get_input_size)
    mults, adds, comps, params, activation_sizes, weights, max_len = 0, 0, 0, 0, 0, 0, 0
    out = net(x)

    # run model and count FLOPs
    for name, module, s in modules:
        if name == 'Linear' and ignore_linear:
            continue
        mult, add, comp, activation_size, weight_size = calculate_flops(name, module)
        mults += mult
        adds += add
        comps += comp
        weights += weight_size
        params += calculate_params(name, module)
        activation_sizes += activation_size
        max_len = max(max_len, len(s))

    activation_sizes *= activation_bits
    weights *= weight_bits

    print("The model crunches:")
    print("    multiplications: {:.2f} GFLOPs".format(mults / 1e9))
    print("          additions: {:.2f} GFLOPs".format(adds / 1e9))
    print("        comparisons: {:.2f} MFLOPs".format(comps / 1e6))
    print("         parameters: {:.2f} Mparams".format(params / 1e6))
    print("        activations: {:.2f} kbits".format(activation_sizes / 1024))
    print("            weights: {:.2f} kbits".format(weights / 1024))

    if layerwise:
        for name, module, s in modules:
            mult, add, comp, _, _ = calculate_flops(name, module)
            print("{}: mult: {:.2f}% add: {:.2f}%".format(s.ljust(max_len), mult / float(mults) * 100,
                                                          add / float(adds) * 100))
