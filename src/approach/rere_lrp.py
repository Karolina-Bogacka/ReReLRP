import copy
import gc
import time
import warnings
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from datasets_local.exemplars_dataset import ExemplarsDataset
from torch.nn.utils import prune as torch_prune
from torch.utils.data import ConcatDataset

from .custom_utils.context_detection import ExplainableContextDetector
from .custom_utils.datasets import ClassIncrTraining, MNISTLikeDatasetTraining
from .custom_utils.prune_layer import (prune_conv_layer_reuse,
                                       prune_linear_layer_reuse)
from .custom_utils.rnf_candidates import freeze_task, get_r_smallest
from .custom_utils.utils import select_composite
from .incremental_learning import Inc_Learning_Appr

EXCLUDED = ["", "dropout", "pool", "out"]

MAX_REL = 999


class Appr(Inc_Learning_Appr):
    """
    Class implementing the algorithm Recognizing and Remembering Tasks with LRP (ReReLRP)
    """

    def __init__(
        self,
        model,
        device,
        nepochs=60,
        lr=0.8,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0.9,
        wd=1e-10,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        logger=None,
        exemplars_dataset=None,
        lamb=1,
        sample_saved=100,
        acc_drop=2,
        features_saved=30,
        pruning_canonizer="epsilon_plus_flat",
        context_detector_canonizer="epsilon_plus_flat",
        num_trees=100,
        feature_selection_mean=False,
        context_detection_layer=None,
        save_metadata=False,
    ):
        super(Appr, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            wu_nepochs,
            wu_lr_factor,
            fix_bn,
            eval_on_train,
            logger,
            exemplars_dataset,
        )
        self.full_weight = False
        self.lamb = lamb
        self.label_attributions = {}
        self.relevances = {}
        self.task_relevances = {}
        self.layer_frozen_indices = {}
        self.task_layer_frozen_indices = {}
        self.hook_handles = {}
        self.prev_val_loaders = []
        self.task_pruned = {}
        self.eval_flag = False
        self.acc_drop = acc_drop
        self.features_saved = features_saved
        self.pruning_canonizer = pruning_canonizer
        self.context_detector_canonizer = context_detector_canonizer
        self.num_trees = num_trees
        self.feature_selection_mean = feature_selection_mean
        self.context_detection_layer = context_detection_layer
        self.save_metadata = save_metadata
        self.context = ExplainableContextDetector(
            sample_saved=sample_saved,
            features_saved=features_saved,
            num_trees=num_trees,
            feature_selection_mean=feature_selection_mean,
            context_detection_layer=context_detection_layer,
            composite=context_detector_canonizer,
        )
        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        if not have_exemplars:
            warnings.warn(
                "Warning: ReReLRP uses the exemplars abstraction to collect relevance data."
            )

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        """
        We need to include all potentially useful for context detection construction parameters here
        """
        parser.add_argument(
            "--acc_drop",
            default=1,
            type=float,
            required=False,
            help="How much accuracy can we allow the submodel to lose (default=%(default)s)",
        )
        parser.add_argument(
            "--features_saved",
            default=40,
            type=float,
            required=False,
            help="How many features are we using for context detection (default=%(default)s)",
        )
        parser.add_argument(
            "--num_trees",
            default=300,
            type=float,
            required=False,
            help="How many trees are we using with the random forest (default=%(default)s)",
        )
        parser.add_argument(
            "--feature_selection_mean",
            default=True,
            type=bool,
            required=False,
            help="If true, we are using relevance mean on first task for feature selection. Otherwise f1 score from all relevances (default=%(default)s)",
        )
        parser.add_argument(
            "--pruning_canonizer",
            default="epsilon_plus_flat",
            type=str,
            required=False,
            help="The composite used to prune the model (default=%(default)s)",
        )
        parser.add_argument(
            "--context_detector_canonizer",
            default="epsilon_plus_flat",
            type=str,
            required=False,
            help="The composite used for context detection (default=%(default)s)",
        )
        parser.add_argument(
            "--context_detection_layer",
            default=None,
            type=str,
            required=False,
            help="If not a None, we are using only one layer for context detection and the name indicates the layer (default=%(default)s)",
        )
        parser.add_argument(
            "--save_metadata",
            default=False,
            type=bool,
            required=False,
            help="If not False, flags that some metadata like neuron relevances should be stored (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def prepare_model_to_prune(self, t):
        """Prepares the model to be pruned based on its relevance
        by reverting it to its original, unwrapped representation
        and keeping only its last head"""
        new_model = copy.deepcopy(self.model.model)
        dimension_1 = int(self.model.heads[t].weight.size()[1])
        dimension_2 = int(self.model.heads[t].weight.size()[0])
        new_out = nn.Linear(dimension_1, dimension_2)
        new_out.weight = copy.deepcopy(self.model.heads[t].weight)
        new_out.bias = copy.deepcopy(self.model.heads[t].bias)
        if (
            type(getattr(new_model, new_model.head_var))
            is not torch.nn.modules.container.Sequential
        ):
            setattr(new_model, self.model.model.head_var, new_out)
        else:
            seq = getattr(new_model, self.model.model.head_var)
            seq.append(new_out)
        return new_model

    def reconnect_head_after_pruning(self, model, train_model, t):
        """Reconstructs the FACIL model representation from the pruned model."""
        new_model = copy.deepcopy(model)
        if (
            type(getattr(new_model, new_model.head_var))
            is not torch.nn.modules.container.Sequential
        ):
            target_layer = getattr(train_model, train_model.head_var)
            dimension_1 = int(target_layer.weight.size()[1])
            dimension_2 = int(target_layer.weight.size()[0])
            target = nn.Linear(dimension_1, dimension_2)
            target.weight = copy.deepcopy(target_layer.weight)
            target.bias = copy.deepcopy(target_layer.bias)
            setattr(new_model, new_model.head_var, target)
            if hasattr(getattr(model, model.head_var), "output_mask"):
                mask = getattr(model, model.head_var).output_mask
                last_layer = getattr(new_model, new_model.head_var)
                last_layer.output_mask = torch.ones(last_layer.weight.shape[0])
                last_layer.output_mask[self.model.task_offset[t - 1] :] = copy.deepcopy(
                    mask
                )
                mask_weight = (
                    last_layer.output_mask.view(-1, 1)
                    .expand_as(last_layer.weight)
                    .to(last_layer.weight.device)
                )
                torch_prune.custom_from_mask(last_layer, "weight", mask_weight)
                mask_bias = last_layer.output_mask.to(last_layer.bias.device)
                torch_prune.custom_from_mask(last_layer, "bias", mask_bias)
        else:
            target_layer = getattr(train_model, train_model.head_var)[-1]
            dimension_1 = int(target_layer.weight.size()[1])
            dimension_2 = int(target_layer.weight.size()[0])
            target = nn.Linear(dimension_1, dimension_2)
            target.weight = copy.deepcopy(target_layer.weight)
            target.bias = copy.deepcopy(target_layer.bias)
            getattr(new_model, new_model.head_var)[-1] = target
            if hasattr(getattr(model, model.head_var), "output_mask"):
                mask = getattr(model, model.head_var).output_mask
                last_layer = getattr(new_model, new_model.head_var)[-1]
                last_layer.output_mask = torch.ones(last_layer.weight.shape[0])
                last_layer.output_mask[self.model.task_offset[t - 1] :] = copy.deepcopy(
                    mask
                )
                mask_weight = (
                    last_layer.output_mask.view(-1, 1)
                    .expand_as(last_layer.weight)
                    .to(last_layer.weight.device)
                )
                torch_prune.custom_from_mask(last_layer, "weight", mask_weight)
                mask_bias = last_layer.output_mask.to(last_layer.bias.device)
                torch_prune.custom_from_mask(last_layer, "bias", mask_bias)
        return new_model

    def compute_relevance(self, model, reference_data, t, excluded=None):
        """
        Computes the relevance of the model with respect to the reference data
        """
        cc = ChannelConcept()
        model = model.eval().to("cuda")
        composite = select_composite(model, self.pruning_canonizer)
        excluded = EXCLUDED if not excluded else excluded
        attributions = {}
        # get layer names of Conv2D and MLP layers
        layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
        attribution = CondAttribution(model)
        # create the attributor, specifying model and composite
        for input_tensor, label in torch.utils.data.DataLoader(
            reference_data, batch_size=1, pin_memory=True
        ):
            # compute with regards to the current label of the image in the reference dataset. since we're taking a subset
            # of the heads for the t-1 task, we can get the length of the tasks here
            offset = self.model.task_offset[t]
            input_tensor = input_tensor.to("cuda")
            conditions = [{"y": [label - offset]}]
            input_tensor.requires_grad = True
            attr = attribution(
                input_tensor, conditions, composite, record_layer=layer_names
            )
            if not attributions:
                attributions = {
                    layer: cc.attribute(attr.relevances[layer], abs_norm=True).cpu()
                    for layer in attr.relevances
                }
            else:
                attributions = {
                    layer: attributions[layer].add(
                        cc.attribute(attr.relevances[layer], abs_norm=True).cpu()
                    )
                    for layer in attributions
                }
            # We also want to keep some info about the relevances to plot their changes
            int_label = int(label)
            if int_label not in self.label_attributions:
                self.label_attributions[int_label] = {
                    layer: cc.attribute(attr.relevances[layer], abs_norm=True).cpu()
                    for layer in attr.relevances
                }
            else:
                self.label_attributions[int_label] = {
                    layer: [
                        self.label_attributions[int_label].get(layer, 0),
                        cc.attribute(attr.relevances[layer], abs_norm=True).cpu(),
                    ]
                    for layer in self.label_attributions[int_label]
                }
        if self.save_metadata:
            torch.save(
                self.label_attributions,
                f"{self.logger.exp_path}/label_attributions.pth",
            )
        return {
            layer: attributions[layer] for layer in layer_names if layer not in excluded
        }

    def message_frozen(self, t):
        """
        Offers the number of parameters frozen per layer (aside from the heads)
        """
        for key in self.layer_frozen_indices:
            self.logger.log_scalar(
                task=t,
                iter=0,
                name=f"frozen_{key}",
                value=len(self.layer_frozen_indices[key]),
                group="test",
            )

    def message_pruned(self, t):
        """
        Deals with logging the number of parameters pruned after each task
        """
        total_pruned = sum([len(self.task_pruned[t][p]) for p in self.task_pruned[t]])
        self.logger.log_scalar(
            task=t, iter=0, name="pruned", value=total_pruned, group="test"
        )

    def prune_model(
        self,
        model,
        relevances,
        device,
        test_loader,
        task,
        t=2,
        excluded=None,
        multi_label=False,
    ):
        to_keep = []
        to_train = [
            i
            for i in range(
                self.model.task_offset[task],
                self.model.task_offset[task] + self.model.task_cls[task],
            )
        ]
        continuous_trained = [i for i in range(self.model.task_offset[task])]
        start_performance = MNISTLikeDatasetTraining(
            to_train=to_train, to_keep=to_keep, continuous_trained=continuous_trained
        ).test_perf(
            model,
            device,
            test_loader,
            mnist_split=False,
            multi_label=multi_label,
            reassign=True,
        )
        iteration = 0
        print(
            f"In iteration {iteration} the pruned network has achieved performance {start_performance} on test set"
        )
        new_performance = start_performance
        model_layers = dict(model.named_modules())
        old_model = None
        flattened = [np.array(v).flatten() for v in relevances.values()]
        all_rels = np.concatenate(flattened)
        # using counts makes for a useful approximation here
        # for the most part
        counts, _ = np.histogram(all_rels, bins=10000)
        self.logger.log_scalar(
            task=t, iter=t + 1, name="loss", value=f"{sum(counts)}", group="test"
        )
        counts = counts[counts != 0]
        while start_performance - new_performance < float(t) and iteration < len(
            counts
        ):
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
            for pos in prune_layers:
                if isinstance(model_layers[pos[0]], nn.Conv2d):
                    model = prune_conv_layer_reuse(model, pos[0], pos[1])
                else:
                    model = prune_linear_layer_reuse(model, pos[0], pos[1])
                torch.cuda.empty_cache()
                gc.collect()
            new_performance = MNISTLikeDatasetTraining(
                to_train=to_train,
                to_keep=to_keep,
                continuous_trained=continuous_trained,
            ).test_perf(
                model,
                device,
                test_loader,
                mnist_split=False,
                multi_label=multi_label,
                reassign=True,
            )
            if start_performance - new_performance < float(t):
                if task not in self.task_pruned:
                    self.task_pruned[task] = {p[0]: p[1] for p in prune_layers}
                else:
                    new_dict = {p[0]: p[1] for p in prune_layers}
                    if new_dict:
                        self.task_pruned[task] = {
                            k: self.task_pruned[task].get(k, []) + new_dict.get(k, [])
                            for k in set(self.task_pruned[task]) | set(new_dict)
                        }
            iteration += 1
            print(
                f"In iteration {iteration} the pruned network has achieved performance {new_performance} on test set after pruning {counts[iteration-1]} filters"
            )
        if task not in self.task_pruned:
            self.task_pruned[task] = {}
        return old_model

    def init_train_reg(self, t):
        """
        Initializes the testing regime for the training process
        """
        to_keep_parts = [
            [
                j + sum(self.model.task_cls[u] for u in range(i))
                for j in range(self.model.task_cls[i])
            ]
            for i in range(t)
        ]
        to_keep = sum([list(np.array(t)) for t in to_keep_parts], [])
        cont = [
            [
                j + sum(self.model.task_cls[u] for u in range(i))
                for j in range(self.model.task_cls[i])
            ]
            for i in range(t + 1)
        ]
        continuous = sum([list(np.array(t)) for t in cont], [])
        return ClassIncrTraining(to_keep=to_keep, continuous_trained=continuous)

    def prepare_grid(self, t):
        if t == 0:
            return self.model
        else:
            model = self.reconnect_head(omit_last=1)
            model_to_prune = self.reconnect_head_after_pruning(
                self.model_to_prune, model, t
            )
            model = self.add_new_head(model)
            model, self.hook_handles, self.layer_frozen_indices = freeze_task(
                model,
                model_to_prune,
                self.hook_handles,
                self.layer_frozen_indices,
                self.task_pruned,
            )
            for param in model.parameters():
                param.requires_grad = True
            model.eval()
            new_self = self.model
            if (
                type(getattr(model, model.head_var))
                is not torch.nn.modules.container.Sequential
            ):
                clone_weight = getattr(model, model.head_var).weight.clone()
                clone_bias = getattr(model, model.head_var).bias.clone()
                clone_index = 0
                to_enumerate_heads = new_self.heads
                for index_small, head in enumerate(to_enumerate_heads):
                    for column_index in range(head.out_features):
                        with torch.no_grad():
                            new_self.heads[index_small].weight[column_index] = (
                                clone_weight[clone_index]
                            )
                            new_self.heads[index_small].bias[column_index] = clone_bias[
                                clone_index
                            ]
                        clone_index += 1
                copy_seq = copy.deepcopy(
                    getattr(new_self.model, new_self.model.head_var)
                )
                setattr(model, model, copy_seq)
            else:
                clone_weight = getattr(model, model.head_var)[-1].weight.clone()
                clone_bias = getattr(model, model.head_var)[-1].bias.clone()
                clone_index = 0
                to_enumerate_heads = new_self.heads
                for index_small, head in enumerate(to_enumerate_heads):
                    for column_index in range(head.out_features):
                        with torch.no_grad():
                            new_self.heads[index_small].weight[column_index] = (
                                clone_weight[clone_index]
                            )
                            new_self.heads[index_small].bias[column_index] = clone_bias[
                                clone_index
                            ]
                        clone_index += 1
                del getattr(model, model.head_var)[-1]
            new_self.model = model
            for param in self.model.heads[:-1].parameters():
                param.requires_grad = False
            return new_self

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        # Warm-up phase
        if t > 0:
            self.train_model = self.reconnect_head(omit_last=1)
            model_to_prune = self.reconnect_head_after_pruning(
                self.model_to_prune, self.train_model, t
            )
            if self.save_metadata:
                torch.save(
                    self.train_model,
                    f"{self.logger.exp_path}/pretrained_model_frank.pth",
                )
            self.train_model = self.add_new_head(self.train_model)
            self.train_model, self.hook_handles, self.layer_frozen_indices = (
                freeze_task(
                    self.train_model,
                    model_to_prune,
                    self.hook_handles,
                    self.layer_frozen_indices,
                    self.task_pruned,
                )
            )
            self.message_frozen(t)
            for param in self.train_model.parameters():
                param.requires_grad = True
            self.train_model.eval()

        if self.warmup_epochs:
            last_layer = getattr(self.train_model, self.train_model.head_var)
            self.optimizer = torch.optim.SGD(last_layer.parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                last_layer.train()
                for images, targets in trn_loader:
                    outputs = self.train_model(images.to(self.device))
                    outputs_old = (
                        self.train_model_old(images.to(self.device)) if t > 0 else None
                    )
                    loss = self.local_criterion(
                        t,
                        outputs,
                        targets.to(self.device),
                        offset=True,
                        outputs_old=outputs_old,
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        last_layer.parameters(), self.clipgrad
                    )
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw, total_acc_tag = 0, 0, 0
                    self.train_model.eval()
                    self.train_reg.test_per_head(
                        self.train_model, self.device, self.ref_test_loader
                    )
                    for images, targets in trn_loader:
                        outputs = self.train_model(images.to(self.device))
                        outputs_old = (
                            self.train_model_old(images.to(self.device))
                            if t > 0
                            else None
                        )
                        loss = self.local_criterion(
                            t,
                            outputs,
                            targets.to(self.device),
                            offset=True,
                            outputs_old=outputs_old,
                        )
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            pred[m] = outputs[m].argmax()
                        hits_taw = (pred == targets.to(self.device)).float()
                        pred = outputs.argmax(1)
                        hits_tag = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                        total_acc_tag += hits_tag.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc, trn_acc_tag = (
                    total_loss / total_num,
                    total_acc_taw / total_num,
                    total_acc_tag / total_num,
                )
                warmupclock2 = time.time()
                print(
                    "| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% , TAg acc={:5.1f}%|".format(
                        e + 1,
                        warmupclock1 - warmupclock0,
                        warmupclock2 - warmupclock1,
                        trn_loss,
                        100 * trn_acc,
                        100 * trn_acc_tag,
                    )
                )
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup"
                )
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup"
                )

    def reconnect_head(self, omit_last=0):
        """Reconnects the head in order to allow for efficient RNF computation"""
        new_model = copy.deepcopy(self.model.model)
        heads_to_use = (
            self.model.heads if not omit_last else self.model.heads[:-omit_last]
        )
        dimension_1 = int(self.model.heads[0].weight.size()[1])
        dimension_2 = int(
            sum(self.model.heads[i].out_features for i in range(len(heads_to_use)))
        )
        new_out = nn.Linear(dimension_1, dimension_2)
        # here, we will take out the weights from the full head with specific indices
        weights_to_add = [self.model.heads[i].weight for i in range(len(heads_to_use))]
        new_out.weight = nn.Parameter(torch.cat(weights_to_add, 0))
        bias_to_add = [self.model.heads[i].bias for i in range(len(heads_to_use))]
        new_out.bias = nn.Parameter(torch.cat(bias_to_add, 0))
        if (
            type(getattr(new_model, new_model.head_var))
            is not torch.nn.modules.container.Sequential
        ):
            setattr(new_model, new_model.head_var, new_out)
        else:
            seq = getattr(new_model, new_model.head_var)
            seq.append(new_out)
        return new_model

    def add_new_head(self, train_model):
        """After freezing, adds the new head to the train model to allow for further computations"""
        dimension_1 = int(self.model.heads[0].weight.size()[1])
        dimension_2 = int(
            sum(self.model.heads[i].out_features for i in range(len(self.model.heads)))
        )
        new_out = nn.Linear(dimension_1, dimension_2)
        if (
            type(getattr(train_model, train_model.head_var))
            is not torch.nn.modules.container.Sequential
        ):
            weights_to_add = [
                getattr(train_model, train_model.head_var).weight,
                self.model.heads[-1].weight,
            ]
            new_out.weight = nn.Parameter(torch.cat(weights_to_add, 0))
            bias_to_add = [
                getattr(train_model, train_model.head_var).bias,
                self.model.heads[-1].bias,
            ]
            new_out.bias = nn.Parameter(torch.cat(bias_to_add, 0))
            setattr(train_model, train_model.head_var, new_out)
        else:
            weights_to_add = [
                getattr(train_model, train_model.head_var)[-1].weight,
                self.model.heads[-1].weight,
            ]
            new_out.weight = nn.Parameter(torch.cat(weights_to_add, 0))
            bias_to_add = [
                getattr(train_model, train_model.head_var)[-1].bias,
                self.model.heads[-1].bias,
            ]
            new_out.bias = nn.Parameter(torch.cat(bias_to_add, 0))
            seq = getattr(train_model, train_model.head_var)
            seq[-1] = new_out
        return train_model

    def update_self_model(self, train_model, omit_last=0):
        """Updates the smaller heads with the effects of computation"""
        new_train = copy.deepcopy(train_model)
        if (
            type(getattr(train_model, train_model.head_var))
            is not torch.nn.modules.container.Sequential
        ):
            clone_weight = getattr(new_train, new_train.head_var).weight.clone()
            clone_bias = getattr(new_train, new_train.head_var).bias.clone()
            clone_index = 0
            to_enumerate_heads = (
                self.model.heads if not omit_last else self.model.heads[:-omit_last]
            )
            for index_small, head in enumerate(to_enumerate_heads):
                for column_index in range(head.out_features):
                    with torch.no_grad():
                        self.model.heads[index_small].weight[column_index] = (
                            clone_weight[clone_index]
                        )
                        self.model.heads[index_small].bias[column_index] = clone_bias[
                            clone_index
                        ]
                    clone_index += 1
            copy_seq = copy.deepcopy(
                getattr(self.model.model, self.model.model.head_var)
            )
            setattr(new_train, new_train.head_var, copy_seq)
        else:
            clone_weight = getattr(new_train, new_train.head_var)[-1].weight.clone()
            clone_bias = getattr(new_train, new_train.head_var)[-1].bias.clone()
            clone_index = 0
            to_enumerate_heads = (
                self.model.heads if not omit_last else self.model.heads[:-omit_last]
            )
            for index_small, head in enumerate(to_enumerate_heads):
                for column_index in range(head.out_features):
                    with torch.no_grad():
                        self.model.heads[index_small].weight[column_index] = (
                            clone_weight[clone_index]
                        )
                        self.model.heads[index_small].bias[column_index] = clone_bias[
                            clone_index
                        ]
                    clone_index += 1
            del getattr(new_train, new_train.head_var)[-1]
        self.model.model = new_train

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        if t > 0:
            self.train_model_old = deepcopy(self.train_model)
        else:
            self.train_model_old = self.reconnect_head(omit_last=0)
        self.train_model_old.eval()
        if self.save_metadata:
            torch.save(
                self.train_model_old, f"{self.logger.exp_path}/train_model_old.pth"
            )
        for param in self.train_model_old.parameters():
            param.requires_grad = False
        self.train_reg = self.init_train_reg(t)
        # try to merge the heads before the whole computation
        # and update heads afterwards
        self.train_model = copy.deepcopy(self.train_model_old)
        # so t larger than 0 here means that it's training after the first task
        self.train_model.eval()
        trn_loader = torch.utils.data.DataLoader(
            trn_loader.dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        self.hook_handles = {}
        # add the main freezing here
        excluded = ["", "dropout", "pool", "log", "out"]
        model_to_prune = copy.deepcopy(self.train_model)
        model_to_prune = self.prepare_model_to_prune(t)
        new_dataset = ConcatDataset(
            [
                self.prev_val_loaders[i].dataset
                for i in range(len(self.prev_val_loaders))
            ]
        )
        self.ref_test_loader = torch.utils.data.DataLoader(
            new_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        self.prune_test_loader = torch.utils.data.DataLoader(
            self.prev_val_loaders[-1].dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        model_to_prune.to(self.device)
        self.relevances = self.compute_relevance(
            model_to_prune, self.exemplars_dataset, t, excluded=excluded
        )
        self.task_relevances[t] = copy.deepcopy(self.relevances)
        if self.save_metadata:
            torch.save(
                self.task_relevances, f"{self.logger.exp_path}/task_relevances.pth"
            )
        model_to_prune.to(self.device)
        self.model_to_prune = self.prune_model(
            model_to_prune,
            self.relevances,
            self.device,
            self.prune_test_loader,
            task=t,
            excluded=excluded,
            t=self.acc_drop,
        )
        self.task_layer_frozen_indices[t] = copy.deepcopy(self.layer_frozen_indices)
        if self.save_metadata:
            torch.save(
                self.task_layer_frozen_indices,
                f"{self.logger.exp_path}/task_layer_frozen_indices_frank.pth",
            )
            torch.save(
                self.task_pruned, f"{self.logger.exp_path}/task_pruned_frank.pth"
            )
        self.train_model.eval()
        # now we need to construct and retrain context detectors
        self.context.update_pruned(
            self.task_pruned, copy.deepcopy(self.train_model), self.model.task_cls
        )
        # if we want to tune hyperparams based on validation set we will communicate it here
        val_loader = self.prev_val_loaders[-1]
        self.context.recompute_relevance_backup(
            self.exemplars_dataset, t, val_loader=val_loader
        )
        self.context.retrain_detectors()
        self.message_pruned(t)

    def local_train_loop(self, t, trn_loader, val_loader, nepochs=None):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = copy.deepcopy(self.train_model.state_dict())

        self.optimizer = torch.optim.SGD(
            self.train_model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            momentum=self.momentum,
        )
        combined_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(
                [val_loader.dataset, self.ref_test_loader.dataset]
            ),
            batch_size=64,
        )
        # Loop epochs
        local_nepochs = nepochs if nepochs else self.nepochs
        for e in range(local_nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, train_acc_2 = self.eval(t, trn_loader)
                clock2 = time.time()
                print(
                    "| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}%, TAg acc={:5.1f}%|".format(
                        e + 1,
                        clock1 - clock0,
                        clock2 - clock1,
                        train_loss,
                        100 * train_acc,
                        100 * train_acc_2,
                    ),
                    end="",
                )
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="loss", value=train_loss, group="train"
                )
                self.logger.log_scalar(
                    task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train"
                )
                self.logger.log_scalar(
                    task=t,
                    iter=e + 1,
                    name="acc_tag",
                    value=100 * train_acc_2,
                    group="train",
                )
            else:
                print(
                    "| Epoch {:3d}, time={:5.1f}s | Train: skip eval |".format(
                        e + 1, clock1 - clock0
                    ),
                    end="",
                )

            # Valid
            clock3 = time.time()
            # full_weight = self.full_weight
            # self.full_weight = False
            self.train_reg.test_per_head(self.train_model, self.device, combined_loader)
            valid_loss, valid_acc, val_acc_2 = self.local_eval(t, val_loader)
            # self.full_weight = full_weight
            clock4 = time.time()
            print(
                " Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}%, TAg acc={:5.1f}%||".format(
                    clock4 - clock3, valid_loss, 100 * valid_acc, 100 * val_acc_2
                ),
                end="",
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="loss", value=valid_loss, group="valid"
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid"
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="acc_tag", value=100 * val_acc_2, group="valid"
            )

            # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = copy.deepcopy(self.train_model.state_dict())
                patience = self.lr_patience
                print(" *", end="")
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    lr /= self.lr_factor
                    print(" lr={:.1e}".format(lr), end="")
                    if lr < self.lr_min:
                        # if the lr decreases below minimum, stop the training session
                        print()
                        break
                    # reset patience and recover best model so far to continue training
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]["lr"] = lr
                    self.train_model.load_state_dict(deepcopy(best_model))
            self.logger.log_scalar(
                task=t, iter=e + 1, name="patience", value=patience, group="train"
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="lr", value=lr, group="train"
            )
            print()
        self.train_model.load_state_dict(deepcopy(best_model))
        self.update_self_model(self.train_model, omit_last=0)

    def _calculate_metrics(self, outputs, all_preds, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = (
                all_preds[this_task][m].argmax() + self.model.task_offset[this_task]
            )
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        # we have to change this to accept outputs instead of all preds
        hits_tag = (
            torch.stack(outputs).to(self.device) == targets.to(self.device)
        ).float()
        return hits_taw, hits_tag

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            # Forward current model
            if self.eval_flag:
                # we're evaluating at the end of a task and want the full model with context detectors
                # and we need to have to use those detectors for something
                outputs, all_preds = self.context.predict(
                    images.to(self.device), self.model.task_offset
                )
                curr_len = len(all_preds[0])
                parsed_preds = [
                    torch.stack([p[t1][0] for p in all_preds]) for t1 in range(curr_len)
                ]
                targets = targets.type(torch.LongTensor).to("cuda")
                loss = self.criterion(t, parsed_preds, targets)
                targets = targets.to("cpu")
                hits_taw, hits_tag = self._calculate_metrics(
                    outputs, parsed_preds, targets
                )
            else:
                with torch.no_grad():
                    outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
            # Log
            total_loss += loss.item() * len(targets)
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        self.eval_flag = False
        if t > 0:
            self.train_reg = self.init_train_reg(t)
            # so t larger than 0 here means that it's training after the first task
            self.train_model.eval()
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )
            # add the main freezing here
            self.update_self_model(self.train_model, omit_last=0)
            self.local_train_loop(t, trn_loader, val_loader, nepochs=self.nepochs)
            clock3 = time.time()
            combined_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset(
                    [val_loader.dataset, self.ref_test_loader.dataset]
                ),
                batch_size=64,
            )
            valid_loss, valid_acc, val_acc_2 = self.local_eval(t, combined_loader)
            clock4 = time.time()
            print(
                " Combined valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}%, TAg acc={:5.1f}%||".format(
                    clock4 - clock3, valid_loss, 100 * valid_acc, 100 * val_acc_2
                ),
                end="",
            )
            self.train_model = copy.deepcopy(self.train_model)

            self.update_self_model(self.train_model, omit_last=0)
            self.full_weight = False
            self.eval_flag = True
        else:
            super().train_loop(t, trn_loader, val_loader)
        # here, we will reconstruct the head in a moment

        self.prev_val_loaders.append(val_loader)
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )
        if self.save_metadata:
            torch.save(
                self.exemplars_dataset, f"{self.logger.exp_path}/exemplars_dataset.pth"
            )

    def local_eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.train_model.eval()
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.train_model(images.to(self.device))
                outputs_old = (
                    self.train_model_old(images.to(self.device)) if t > 0 else None
                )
                loss = self.local_criterion(
                    t,
                    outputs,
                    targets.to(self.device),
                    offset=True,
                    outputs_old=outputs_old,
                )
                outputs = self.model(images.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        local_offset = False
        if t > 0:
            if self.fix_bn:
                self.model.freeze_bn()
                for m in self.train_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        if hasattr(m, "weight"):
                            m.weight.requires_grad_(False)
                        if hasattr(m, "bias"):
                            m.bias.requires_grad_(False)
                        m.eval()
            local_offset = True
            model = self.train_model
        else:
            model = self.model
        for images, targets in trn_loader:
            # Forward current model
            outputs = model(images.to(self.device))
            outputs_old = (
                self.train_model_old(images.to(self.device)) if t > 0 else None
            )
            loss = self.local_criterion(
                t,
                outputs,
                targets.to(self.device),
                offset=local_offset,
                outputs_old=outputs_old,
            )
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
            self.optimizer.step()
        if t > 0:
            self.update_self_model(self.train_model, omit_last=0)

    def local_criterion(self, t, outputs, targets, offset=False, outputs_old=None):
        import pdb

        """Returns the loss value"""
        # I think we may want to change the lower part to make the
        # whole method more technically correct and consistent
        # new_weights[self.model.task_offset[t] :] = 1
        if t > 0 and self.full_weight:
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        elif offset:
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        else:
            loss = torch.nn.functional.cross_entropy(
                outputs[t], targets - self.model.task_offset[t]
            )
        return loss
