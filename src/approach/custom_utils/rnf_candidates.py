import copy
import gc
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from PIL import Image
from zennit.composites import EpsilonPlus, EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer, SequentialMergeBatchNorm

from .custom_freezing import freeze_conv2d_params, freeze_linear_params
from .datasets import DualDatasetTraining, MNISTLikeDatasetTraining
from .prune_layer import (prune_conv_layer, prune_conv_layer_reuse,
                          prune_linear_layer, prune_linear_layer_reuse)

EXCLUDED = ["", "dropout", "pool", "out"]
ALLOWED = [nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d]

MAX_REL = 999


class ReferenceDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations, cnn_train=True):
        super(ReferenceDataset, self).__init__()
        self.img_list = img_list
        self.cnn_train = cnn_train
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        label = int(self.img_list[idx].split("_")[-1].split(".jpg")[0])
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.cnn_train:
            data = self.augmentations(img)  # .to(device)
        else:
            data = self.augmentations(img)[None]  # .to(device)
        return (data, label)


def compute_relevance(
    model,
    reference_data,
    target_output=7,
    n_random=5,
    mnist_split=True,
    excluded=None,
    canonizer=False,
    abs=False,
):
    cc = ChannelConcept()
    model = model.to("cpu")
    composite = (
        EpsilonPlusFlat()
        if not canonizer
        else EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    )
    excluded = EXCLUDED if not excluded else excluded
    attributions = {}
    # get layer names of Conv2D and MLP layers
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    attribution = CondAttribution(model)
    # create the attributor, specifying model and composite
    for input_tensor, label in torch.utils.data.DataLoader(
        reference_data, batch_size=1, pin_memory=True
    ):
        # compute with regards to the current label of the image in the reference dataset
        if mnist_split:
            label[label % 2 == 0] = 0
            label[label % 2 != 0] = 1
        conditions = [{"y": [label]}]
        input_tensor.requires_grad = True
        attr = attribution(
            input_tensor, conditions, composite, record_layer=layer_names
        )
        if not attributions:
            attributions = {
                layer: cc.attribute(attr.relevances[layer], abs_norm=True)
                for layer in attr.relevances
            }
        else:
            attributions = {
                layer: attributions[layer].add(
                    cc.attribute(attr.relevances[layer], abs_norm=True)
                )
                for layer in attributions
            }
    return {
        layer: attributions[layer] for layer in layer_names if layer not in excluded
    }


def compute_minus_relevance(
    model,
    reference_data,
    target_outputs=[0, 1],
    n_random=5,
    mnist_split=True,
    excluded=None,
    canonizer=False,
    abs=False,
):
    cc = ChannelConcept()
    model = model.to("cpu")
    composite = (
        EpsilonPlusFlat()
        if not canonizer
        else EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    )
    excluded = EXCLUDED if not excluded else excluded
    attributions = {}
    # get layer names of Conv2D and MLP layers
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    attribution = CondAttribution(model)
    # create the attributor, specifying model and composite
    for input_tensor, label in torch.utils.data.DataLoader(
        reference_data, batch_size=1, pin_memory=True
    ):
        # compute with regards to the current label of the image in the reference dataset
        if mnist_split:
            label[label % 2 == 0] = 0
            label[label % 2 != 0] = 1
        conditions = [{"y": target_outputs}]
        input_tensor.requires_grad = True
        attr = attribution(
            input_tensor, conditions, composite, record_layer=layer_names
        )
        if not attributions:
            attributions = {
                layer: cc.attribute(attr.relevances[layer], abs_norm=True)
                for layer in attr.relevances
            }
        else:
            attributions = {
                layer: attributions[layer].add(
                    cc.attribute(attr.relevances[layer], abs_norm=True)
                )
                for layer in attributions
            }
    return {
        layer: -1 * attributions[layer]
        for layer in layer_names
        if layer not in excluded
    }


def get_r_smallest(relevances, model, r=1, excluded=None):
    layer_sort = {}
    excluded = EXCLUDED if not excluded else excluded
    layers_dict = dict(model.named_modules())
    for layer in relevances:
        if layer not in excluded:
            if hasattr(layers_dict[layer], "output_mask"):
                indices_zero = (layers_dict[layer].output_mask == 0).nonzero().flatten()
                relevances[layer][0][indices_zero] = MAX_REL
            # Instantiate output mask tensor
            len_rel = len(list(relevances[layer].flatten()))
            if len_rel < r:
                rel_values, concept_ids = torch.topk(
                    relevances[layer][0], len_rel, largest=False
                )
            else:
                rel_values, concept_ids = torch.topk(
                    relevances[layer][0], r, largest=False
                )
            for index in range(len(rel_values.cpu())):
                layer_sort[(layer, concept_ids.cpu()[index].item())] = rel_values.cpu()[
                    index
                ].item()
    # then get the lowest r values from all layers
    return relevances, sorted(layer_sort.items(), key=lambda x: x[1])[:r]


def prune_model(
    model,
    relevances,
    device,
    test_loader,
    t=2,
    r=3,
    freeze_multiplier=0.8,
    excluded=None,
    multi_label=False,
    target=0,
):
    start_performance = MNISTLikeDatasetTraining(to_train=[]).test_perf(
        model, device, test_loader, mnist_split=False, multi_label=multi_label
    )
    iteration = 0
    print(
        f"In iteration {iteration} the pruned network has achieved performance {start_performance} on test set"
    )
    new_performance = start_performance
    model_layers = dict(model.named_modules())
    old_model = None
    # btw is there a difference in values between pruned neuron relevances and the rest?
    # like a drastic one that would help us to have less computations?
    flattened = [np.array(v).flatten() for v in relevances.values()]
    all_rels = np.concatenate(flattened)
    counts, _ = np.histogram(all_rels, bins=10000)
    while start_performance - new_performance < t:
        del old_model
        old_model = copy.deepcopy(model)
        relevances, to_prune = get_r_smallest(
            relevances, model, r=counts[iteration], excluded=excluded
        )
        prune_values = [to[0] for to in to_prune]
        prune_keys = np.unique([p[0] for p in prune_values])
        prune_layers = [
            (key, [p[1] for p in prune_values if p[0] == key]) for key in prune_keys
        ]
        # pruned_iterations.append(to_prune)
        for pos in prune_layers:
            if isinstance(model_layers[pos[0]], nn.Conv2d):
                model = prune_conv_layer_reuse(model, pos[0], pos[1])
            else:
                model = prune_linear_layer_reuse(model, pos[0], pos[1])
            torch.cuda.empty_cache()
            gc.collect()
        new_performance = MNISTLikeDatasetTraining(to_train=[]).test_perf(
            model, device, test_loader, mnist_split=False, multi_label=multi_label
        )

        iteration += 1
        print(
            f"In iteration {iteration} the pruned network has achieved performance {new_performance} on test set after pruning {counts[iteration]} filters"
        )
    return old_model


def prune_model_precision(
    model,
    relevances,
    device,
    test_loader,
    t=2,
    r=3,
    freeze_multiplier=0.8,
    excluded=None,
    multi_label=False,
):
    _, start_performance = MNISTLikeDatasetTraining(to_train=[]).precision_per_class(
        model, device, test_loader, mnist_split=False, multi_label=multi_label
    )
    iteration = 0
    pruned_iterations = []
    print(
        f"In iteration {iteration} the pruned network has achieved performance {start_performance} on test set"
    )
    new_performance = start_performance
    model_layers = dict(model.named_modules())
    while start_performance - new_performance < t:
        old_model = copy.deepcopy(model)
        to_prune = get_r_smallest(relevances, model, r=r, excluded=excluded)
        pruned_iterations.append(to_prune)
        model.to("cpu")
        for pos, v in to_prune:
            if isinstance(model_layers[pos[0]], nn.Conv2d):
                model = prune_conv_layer(model, pos[0], pos[1])
            else:
                model = prune_linear_layer(model, pos[0], pos[1])
            torch.cuda.empty_cache()
        model.to(device)
        _, new_performance = MNISTLikeDatasetTraining(to_train=[]).precision_per_class(
            model, device, test_loader, mnist_split=False, multi_label=multi_label
        )
        iteration += 1
        print(
            f"In iteration {iteration} the pruned network has achieved performance {new_performance} on test set"
        )
    return old_model


def get_least_relevant_subset(model, subset, train_reg=DualDatasetTraining(), piece=9):
    cc = ChannelConcept()
    model = model.to("cpu")
    composite = EpsilonPlus()
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    attribution = CondAttribution(model)
    # create the attributor, specifying model and composite
    attributions = {}
    model.eval()
    for input_tensor, label in torch.utils.data.DataLoader(
        subset, batch_size=1, pin_memory=True
    ):
        label = train_reg.reassign_target(label)
        conditions = [{"y": label}]
        input_tensor.requires_grad = True
        attr = attribution(
            input_tensor, conditions, composite, record_layer=layer_names
        )
        if not attributions:
            attributions = {
                layer: cc.attribute(attr.relevances[layer], abs_norm=True)
                for layer in attr.relevances
            }
        else:
            attributions = {
                layer: attributions[layer].add(
                    cc.attribute(attr.relevances[layer], abs_norm=True)
                )
                for layer in attributions
            }
    return {
        layer: torch.topk(
            attributions[layer],
            int(len(attributions[layer][0]) * piece / 100),
            largest=False,
        )
        for layer in attributions
    }


def get_most_relevant_subset(model, subset, train_reg=DualDatasetTraining(), piece=2):
    cc = ChannelConcept()
    model = model.to("cpu")
    composite = EpsilonPlus()
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
    attribution = CondAttribution(model)
    # create the attributor, specifying model and composite
    attributions = {}
    model.eval()
    for input_tensor, label in torch.utils.data.DataLoader(
        subset, batch_size=1, pin_memory=True
    ):
        label = train_reg.reassign_target(label)
        conditions = [{"y": label}]
        input_tensor.requires_grad = True
        attr = attribution(
            input_tensor, conditions, composite, record_layer=layer_names
        )
        if not attributions:
            attributions = {
                layer: cc.attribute(attr.relevances[layer], abs_norm=True)
                for layer in attr.relevances
            }
        else:
            attributions = {
                layer: attributions[layer].add(
                    cc.attribute(attr.relevances[layer], abs_norm=True)
                )
                for layer in attributions
            }
    return {
        layer: torch.topk(
            attributions[layer],
            int(len(attributions[layer][0]) * piece / 100),
            largest=True,
        )
        for layer in attributions
    }


def get_least_relevant_subset_for_all_classes(
    model, subset, train_reg=DualDatasetTraining(), piece=9
):
    """
    Calls the method separately for each class and returns least important for each of the classes
    """
    classes = np.unique(subset.labels)
    class_indices = [
        np.argwhere(np.array(subset.labels) == c).flatten() for c in classes
    ]
    class_subsets = [torch.utils.data.Subset(subset, c) for c in class_indices]
    relevant_dicts = [
        get_least_relevant_subset(model, s, train_reg=train_reg, piece=piece)
        for s in class_subsets
    ]
    relevants = [{key: i[key].indices[0] for key in i} for i in relevant_dicts]
    results = dict(
        reduce(
            lambda x, y: {
                k: torch.IntTensor(np.intersect1d(x.get(k, 0), y.get(k, 0)))
                for k in set(x) & set(y)
            },
            relevants,
        )
    )
    return results


def get_most_relevant_subset_per_class(
    model, subset, train_reg=DualDatasetTraining(), piece=2
):
    """
    Calls the method separately for each class and returns most important for all classes
    """
    classes = np.unique(subset.labels)
    class_indices = [
        np.argwhere(np.array(subset.labels) == c).flatten() for c in classes
    ]
    class_subsets = [torch.utils.data.Subset(subset, c) for c in class_indices]
    relevant_dicts = [
        get_most_relevant_subset(model, s, train_reg=train_reg, piece=piece)
        for s in class_subsets
    ]
    relevants = [{key: i[key].indices[0] for key in i} for i in relevant_dicts]
    results = dict(
        reduce(
            lambda x, y: {
                k: torch.unique(torch.cat((x.get(k, 0), y.get(k, 0)), 0))
                for k in set(x) | set(y)
            },
            relevants,
        )
    )
    return results


def freeze_BUT_masks(
    model_new, masks, hook_handles, layer_frozen_indices, excluded=None, tune_val=0
):
    """
    Tries to freeze everything but the selected masks. Additionally logs how many parameters were frozen in each layer
    (This logging of frozen parameters will be added anyway I think based on layer_frozen_indices)
    """
    modules_new = dict(model_new.named_modules())
    excluded = EXCLUDED if not excluded else excluded
    to_freeze = {}
    for mask in masks:
        for layer in mask:
            if layer not in excluded:
                curr_mask = mask[layer]
                # here, we originally select the nonmasked parts of the pruned model to freeze
                indices = (curr_mask != 0).nonzero()
                # but now we will save out only those to freeze
                if layer in layer_frozen_indices:
                    for el in layer_frozen_indices[layer]:
                        if el not in indices:
                            indices = torch.cat((indices, torch.unsqueeze(el, 0)), 0)
                layer_frozen_indices[layer] = indices
                if list(np.array(indices).flatten()):
                    if layer in hook_handles:
                        prev_hook, prev_bias_hook = hook_handles[layer]
                    else:
                        prev_hook, prev_bias_hook = None, None
                    if isinstance(modules_new[layer], nn.Conv2d):
                        weight_hook_handle, bias_hook_handle = freeze_conv2d_params(
                            modules_new[layer],
                            indices,
                            bias_indices=indices,
                            weight_hook_handle=prev_hook,
                            bias_hook_handle=prev_bias_hook,
                            weight_multi_val=tune_val,
                            bias_multi_val=tune_val,
                        )
                    else:
                        weight_hook_handle, bias_hook_handle = freeze_linear_params(
                            modules_new[layer],
                            indices,
                            bias_indices=indices,
                            weight_hook_handle=prev_hook,
                            bias_hook_handle=prev_bias_hook,
                            weight_multi_val=tune_val,
                            bias_multi_val=tune_val,
                        )
                    hook_handles[layer] = [weight_hook_handle, bias_hook_handle]
    return model_new, hook_handles, layer_frozen_indices


def freeze_masks(
    model_new, masks, hook_handles, layer_frozen_indices, excluded=None, tune_val=0
):
    modules_new = dict(model_new.named_modules())
    excluded = EXCLUDED if not excluded else excluded
    for layer in masks:
        if layer not in excluded:
            curr_mask = masks[layer]
            indices = (curr_mask != 0).nonzero()
            if layer in layer_frozen_indices:
                for el in layer_frozen_indices[layer]:
                    if el not in indices:
                        indices = torch.cat((indices, torch.unsqueeze(el, 0)), 0)
            layer_frozen_indices[layer] = indices
            if list(np.array(indices).flatten()):
                if layer in hook_handles:
                    prev_hook, prev_bias_hook = hook_handles[layer]
                else:
                    prev_hook, prev_bias_hook = None, None
                if isinstance(modules_new[layer], nn.Conv2d):
                    weight_hook_handle, bias_hook_handle = freeze_conv2d_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                        weight_multi_val=tune_val,
                        bias_multi_val=tune_val,
                    )
                else:
                    weight_hook_handle, bias_hook_handle = freeze_linear_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                        weight_multi_val=tune_val,
                        bias_multi_val=tune_val,
                    )
                hook_handles[layer] = [weight_hook_handle, bias_hook_handle]
    return model_new, hook_handles, layer_frozen_indices


def unfreeze_model(hook_handles):
    for layer in hook_handles:
        weight_hook_handle, bias_hook_handle = hook_handles[layer]
        if weight_hook_handle is not None:
            weight_hook_handle.remove()
        if bias_hook_handle is not None:
            bias_hook_handle.remove()


def freeze(
    model_new,
    model_pruned,
    hook_handles,
    layer_frozen_indices,
    excluded=None,
    pruned_parameters=None,
):
    modules_new = dict(model_new.named_modules())
    modules_pruned = dict(model_pruned.named_modules())
    allowed = ALLOWED
    for layer in modules_pruned:
        # here, I don't think this should have the if hasattr, it should be if layer not in excluded
        # and if hasattr then the whole layer should be frozen
        # and add a clause for differences in sizes to accomodate for the additional layers
        if type(modules_pruned[layer]) in allowed and not isinstance(
            layer, nn.Sequential
        ):
            if (
                hasattr(modules_pruned[layer], "output_mask")
                and f"{model_new.head_var}." not in layer
            ):
                indices = (modules_pruned[layer].output_mask != 0).nonzero()
                if layer in layer_frozen_indices:
                    for el in layer_frozen_indices[layer]:
                        if el not in indices:
                            indices = torch.cat((indices, torch.unsqueeze(el, 0)), 0)
            else:
                # this should help with properly freezing whole layers that are not pruned
                output = torch.ones(modules_pruned[layer].weight.shape[0])
                indices = (output != 0).nonzero()
            layer_frozen_indices[layer] = indices
            if list(np.array(indices).flatten()):
                if layer in hook_handles:
                    prev_hook, prev_bias_hook = hook_handles[layer]
                else:
                    prev_hook, prev_bias_hook = None, None
                if isinstance(modules_new[layer], nn.Conv2d):
                    weight_hook_handle, bias_hook_handle = freeze_conv2d_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                    )
                else:
                    weight_hook_handle, bias_hook_handle = freeze_linear_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                    )
                hook_handles[layer] = [weight_hook_handle, bias_hook_handle]
    return model_new, hook_handles, layer_frozen_indices


def freeze_task(
    model_new, model_pruned, hook_handles, layer_frozen_indices, task_pruned
):
    last_key = list(task_pruned.keys())[-1]
    pruned_now = task_pruned[last_key]
    modules_new = dict(model_new.named_modules())
    modules_pruned = dict(model_pruned.named_modules())
    allowed = ALLOWED
    for layer in modules_pruned:
        # here, I don't think this should have the if hasattr, it should be if layer not in excluded
        # and if hasattr then the whole layer should be frozen
        # and add a clause for differences in sizes to accomodate for the additional layers
        if type(modules_pruned[layer]) in allowed and not isinstance(
            modules_pruned[layer], nn.Sequential
        ):
            if layer in pruned_now or layer in layer_frozen_indices:
                output = torch.ones(modules_new[layer].weight.shape[0])
                indices = (output != 0).nonzero()
                if layer in pruned_now:
                    for el in pruned_now[layer]:
                        indices = indices[indices != el]
                if layer in layer_frozen_indices:
                    for el in layer_frozen_indices[layer]:
                        if el not in indices:
                            if len(el.shape) < len(indices.shape):
                                el = torch.unsqueeze(el, 0)
                            indices = torch.cat((indices, el), 0)
            else:
                # this should help with properly freezing whole layers that are not pruned
                output = torch.ones(modules_pruned[layer].weight.shape[0])
                indices = (output != 0).nonzero()
            layer_frozen_indices[layer] = indices
            if list(np.array(indices).flatten()):
                if layer in hook_handles:
                    prev_hook, prev_bias_hook = hook_handles[layer]
                    prev_hook.remove()
                    prev_bias_hook.remove()
                prev_hook, prev_bias_hook = None, None
                if isinstance(modules_new[layer], nn.Conv2d):
                    weight_hook_handle, bias_hook_handle = freeze_conv2d_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                    )
                else:
                    weight_hook_handle, bias_hook_handle = freeze_linear_params(
                        modules_new[layer],
                        indices,
                        bias_indices=indices,
                        weight_hook_handle=prev_hook,
                        bias_hook_handle=prev_bias_hook,
                    )
                hook_handles[layer] = [weight_hook_handle, bias_hook_handle]
    return model_new, hook_handles, layer_frozen_indices
