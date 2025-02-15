import copy
import gc

import numpy as np
import torch
from torch.nn.utils import prune as torch_prune
from torch.nn.utils.prune import BasePruningMethod

"""
Credit for initial implementation goes to https://github.com/seulkiyeom/LRP_pruning (as detailed in LICENSE-LRP-PRUNING)
"""


def prune_linear_layer(model, layer_index, filter_index):
    dense = dict(model.named_modules())[layer_index]
    if not hasattr(dense, "output_mask"):
        # Instantiate output mask tensor of shape (num_output_channels, )
        dense.output_mask = torch.ones(dense.weight.shape[0])
    # Make sure the filter was not pruned before
    assert dense.output_mask[filter_index] != 0

    dense.output_mask[filter_index] = 0
    mask_weight = dense.output_mask.view(-1, 1).expand_as(dense.weight)  # .to(device)
    torch_prune.custom_from_mask(dense, "weight", mask_weight)
    if dense.bias is not None:
        mask_bias = dense.output_mask  # .to(device)
        torch_prune.custom_from_mask(dense, "bias", mask_bias)

    return model


def prune_conv_layer(
    model, layer_index, filter_index, criterion="lrp", cuda_flag=False
):
    """input parameters
    1. model: current model
    2. layer_index: layer index to cut
    3. filter_index: Filter index of the layer you want to cut
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = dict(model.named_modules())[layer_index]

    if not hasattr(conv, "output_mask"):
        # Instantiate output mask tensor of shape (num_output_channels, )
        conv.output_mask = torch.ones(conv.weight.shape[0])

    # Make sure the filter was not pruned before
    assert conv.output_mask[filter_index] != 0

    conv.output_mask[filter_index] = 0

    mask_weight = conv.output_mask.view(-1, 1, 1, 1).expand_as(
        conv.weight
    )  # .to(device)
    torch_prune.custom_from_mask(conv, "weight", mask_weight)

    if conv.bias is not None:
        mask_bias = conv.output_mask  # .to(device)
        torch_prune.custom_from_mask(conv, "bias", mask_bias)

    if cuda_flag:
        conv.weight = conv.weight.cuda()
        # conv.module.bias = conv.module.bias.cuda()

    return model


def prune_linear_layer_reuse(model, layer_index, filter_indices):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = dict(model.named_modules())[layer_index]
    if not hasattr(dense, "output_mask"):
        # Instantiate output mask tensor of shape (num_output_channels, )
        dense.output_mask = torch.ones(dense.weight.shape[0])
    # Make sure the filter was not pruned before
    for filter_index in filter_indices:
        dense.output_mask[filter_index] = 0
    mask_weight_base = copy.deepcopy(dense.output_mask)
    mask_weight = mask_weight_base.view(-1, 1).expand_as(dense.weight).to(device)
    if hasattr(dense, "weight_mask"):
        # maybe what we actually want to do here is to remove the hooks?
        # either that or keep better track of what we've pruned
        dense = torch_prune.remove(dense, name="weight")
    torch_prune.custom_from_mask(dense, "weight", mask_weight)
    if dense.bias is not None:
        mask_bias = mask_weight_base.to(device)
        if hasattr(dense, "bias_mask"):
            dense = torch_prune.remove(dense, name="bias")
        torch_prune.custom_from_mask(dense, "bias", mask_bias)
    torch.cuda.empty_cache()
    gc.collect()
    return model


def prune_conv_layer_reuse(
    model, layer_index, filter_indices, criterion="lrp", cuda_flag=False
):
    """input parameters
    1. model: current model
    2. layer_index: layer index to cut
    3. filter_index: Filter index of the layer you want to cut
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv = dict(model.named_modules())[layer_index]

    if not hasattr(conv, "output_mask"):
        conv.output_mask = torch.ones(conv.weight.shape[0])

        # Instantiate output mask tensor of shape (num_output_channels, )
    if hasattr(conv, "weight_mask"):
        conv = torch_prune.remove(conv, name="weight")

    for filter_index in filter_indices:
        conv.output_mask[filter_index] = 0

    mask_weight_base = copy.deepcopy(conv.output_mask)
    mask_weight = mask_weight_base.view(-1, 1, 1, 1).expand_as(conv.weight).to(device)
    torch_prune.custom_from_mask(conv, "weight", mask_weight)

    if conv.bias is not None:
        mask_bias = mask_weight_base.to(device)
        if hasattr(conv, "bias_mask"):
            conv = torch_prune.remove(conv, name="bias")
        torch_prune.custom_from_mask(conv, "bias", mask_bias)

    if cuda_flag:
        conv.weight = conv.weight.cuda()
        # conv.module.bias = conv.module.bias.cuda()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer_sequential(model, layer_index, filter_index, cuda_flag=False):
    """input parameters
    1. model: current model
    2. layer_index: layer index to cut
    3. filter_index: Filter index of the layer you want to cut
    """
    # _, conv = model.features._modules.items()[layer_index]
    _, conv = list(model.features._modules.items())[
        layer_index
    ]  # The corresponding layer is pulled out first.
    next_conv = None
    offset = 1

    while layer_index + offset < len(
        model.features._modules.items()
    ):  # Repeat the while statement until it is larger than the number of layers in the entire network.
        # res =  model.features._modules.items()[layer_index+offset]
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(
            res[1], torch.nn.modules.conv.Conv2d
        ):  # Based on the current layer, is the next layer a conv layer?
            next_name, next_conv = res
            break
        offset = offset + 1

    # Recreates a new conv_layer (new_conv).
    # Reduce the number on the output side by one.
    new_conv = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels - 1,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    # for p in new_conv.parameters():
    #     p.requires_grad = False

    # The weight is the total number minus 1, excluding the filter.
    new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    new_conv.weight.data = (
        torch.from_numpy(new_weights).cuda()
        if cuda_flag
        else torch.from_numpy(new_weights)
    )

    # For bias, the total number is -1, excluding the value of the filter number.
    bias_numpy = conv.bias.data.cpu().numpy()
    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1 :]
    new_conv.bias.data = (
        torch.from_numpy(bias).cuda() if cuda_flag else torch.from_numpy(bias)
    )

    # The next conv layer also reduces the number of layers on the receiving side.
    if not next_conv is None:
        next_new_conv = torch.nn.Conv2d(
            in_channels=next_conv.in_channels - 1,
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            dilation=next_conv.dilation,
            groups=next_conv.groups,
            bias=True,
        )

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        # for p in next_new_conv.parameters():
        #     p.requires_grad = False

        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1 :, :, :]
        next_new_conv.weight.data = (
            torch.from_numpy(new_weights).cuda()
            if cuda_flag
            else torch.from_numpy(new_weights)
        )

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
            *(
                replace_layers(
                    model.features,
                    i,
                    [layer_index, layer_index + offset],
                    [new_conv, next_new_conv],
                )
                for i, _ in enumerate(model.features)
            )
        )
        del model.features
        del conv

        model.features = features

    else:
        # Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
            *(
                replace_layers(model.features, i, [layer_index], [new_conv])
                for i, _ in enumerate(model.features)
            )
        )
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - params_per_input_channel,
            old_linear_layer.out_features,
        )

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] = old_weights[
            :, : filter_index * params_per_input_channel
        ]
        new_weights[:, filter_index * params_per_input_channel :] = old_weights[
            :, (filter_index + 1) * params_per_input_channel :
        ]

        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = (
            torch.from_numpy(new_weights).cuda()
            if cuda_flag
            else torch.from_numpy(new_weights)
        )

        classifier = torch.nn.Sequential(
            *(
                replace_layers(model.classifier, i, [layer_index], [new_linear_layer])
                for i, _ in enumerate(model.classifier)
            )
        )

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier

    return model
