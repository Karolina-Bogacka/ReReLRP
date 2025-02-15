"""
This file contains simple utility scripts, such as methods used to organize the labels in the dataset.
"""

import os
from abc import ABC, abstractmethod
from logging import INFO

import numpy as np
import torch
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from zennit.composites import (EpsilonAlpha2Beta1Flat, EpsilonPlusFlat,
                               LayerMapComposite,
                               SpecialFirstLayerMapComposite)
from zennit.layer import Sum
from zennit.rules import Epsilon, Flat, Norm, Pass, ZPlus
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from zennit.types import Activation, AvgPool, BatchNorm, Convolution, Linear


def select_composite(model, name):
    """
    Selects the composite to be used for the model.
    """
    canonizers = (
        [VGGCanonizer()]
        if "vgg" in model.__module__
        else [ResNetCanonizer()] if "resnet" in model.__module__ else []
    )
    match name:
        case "epsilon":
            layer_map = [
                (Convolution, Epsilon(epsilon=1e-6)),  # any convolutional layer
                (Linear, Epsilon(epsilon=1e-6)),  # this is the dense Linear, not any
            ]
            return LayerMapComposite(layer_map, canonizers=canonizers)
        case "epsilon_flat":
            first_map = [(Linear, Flat())]
            layer_map = [
                (Convolution, Epsilon(epsilon=1e-6)),  # any convolutional layer
                (Linear, Epsilon(epsilon=1e-6)),  # this is the dense Linear, not any
            ]
            return SpecialFirstLayerMapComposite(
                layer_map, first_map=first_map, canonizers=canonizers
            )
        case "zplus":
            layer_map = [
                (Convolution, ZPlus()),  # any convolutional layer
                (Linear, ZPlus()),  # this is the dense Linear, not any
            ]
            return LayerMapComposite(layer_map, canonizers=canonizers)
        case "zplus_flat":
            first_map = [(Linear, Flat())]
            layer_map = [
                (Convolution, ZPlus()),  # any convolutional layer
                (Linear, ZPlus()),  # this is the dense Linear, not any
            ]
            return SpecialFirstLayerMapComposite(
                layer_map, first_map=first_map, canonizers=canonizers
            )
        case "epsilon_plus_flat":
            composite = EpsilonPlusFlat(canonizers=canonizers)
            return composite
        case "epsilon_alpha2_beta1_flat":
            composite = EpsilonAlpha2Beta1Flat(canonizers=canonizers)
            return composite


class FeatureSelector(ABC):

    def __init__(self, features_saved=30, select_layer=None):
        self.features_saved = features_saved
        self.select_layer = select_layer

    def _select_layers(self, data):
        first_key = list(data.keys())[0]
        keys_sorted = (
            sorted(data[first_key].keys())
            if not self.select_layer
            else [self.select_layer]
        )
        return {
            k: [
                np.concatenate(
                    [data[k][s][i].flatten().cpu().numpy() for s in keys_sorted]
                )
                for i in range(len(data[k][keys_sorted[-1]]))
            ]
            for k in data
        }

    @abstractmethod
    def fit_transform(self, data, target_var):
        pass

    @abstractmethod
    def transform(self, data, target_var):
        pass

    @abstractmethod
    def transform_x(self, data):
        pass


class SelectKBestSelector(FeatureSelector):

    def __init__(self, features_saved=30, select_layer=None):
        super().__init__(features_saved, select_layer=select_layer)
        self.selector = SelectKBest(mutual_info_classif, k=self.features_saved)

    def fit_transform(self, data, target_var):
        data = self._select_layers(data)
        self.selector = SelectKBest(mutual_info_classif, k=int(self.features_saved))
        train = {key: [np.array(it) for it in data[key]] for key in data}
        X_train, y_train = [], []
        # here, we go with min to minimize imbalance
        for it in range(min([len(train[key]) for key in train])):
            for key in train:
                if it < len(train[key]):
                    row = train[key][it]
                    y = 0 if key == target_var else 1
                    X_train.append(row)
                    y_train.append(y)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_new = self.selector.fit_transform(X_train, y_train)
        return X_new, y_train

    def transform(self, data, target_var):
        data = self._select_layers(data)
        train = {key: [np.array(it) for it in data[key]] for key in data}
        X_train, y_train = [], []
        # here, we go with max to make sure we include all test data
        for it in range(max([len(train[key]) for key in train])):
            for key in train:
                if it < len(train[key]):
                    row = train[key][it]
                    y = 0 if key == target_var else 1
                    X_train.append(row)
                    y_train.append(y)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_new = self.selector.transform(X_train)
        return X_new, y_train

    def transform_x(self, data):
        keys_sorted = (
            sorted(data.keys()) if not self.select_layer else [self.select_layer]
        )
        data = np.concatenate([data[s].flatten().cpu().numpy() for s in keys_sorted])
        train = self.selector.transform(np.array([data]))
        return train


class FeatureSelectionMean(FeatureSelector):

    def __init__(self, features_saved=30, select_layer=None):
        super().__init__(features_saved, select_layer=select_layer)
        self.most_relevant = []

    def fit_transform(self, data, target_var):
        data = self._select_layers(data)
        self.most_relevant = np.argsort(np.mean(np.array(data[target_var]), axis=0))[
            -int(self.features_saved) :
        ]
        train = {
            key: [np.take(it, self.most_relevant) for it in data[key]] for key in data
        }
        X_train, y_train = [], []
        # here, we go with max to make sure we include all test data
        for it in range(min([len(train[key]) for key in train])):
            for key in train:
                if it < len(train[key]):
                    row = train[key][it]
                    y = 0 if key == target_var else 1
                    X_train.append(row)
                    y_train.append(y)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train

    def transform(self, data, target_var):
        data = self._select_layers(data)
        train = {
            key: [np.take(it, self.most_relevant) for it in data[key]] for key in data
        }
        X_train, y_train = [], []
        # here, we go with max to make sure we include all test data
        for it in range(min([len(train[key]) for key in train])):
            for key in train:
                if it < len(train[key]):
                    row = train[key][it]
                    y = 0 if key == target_var else 1
                    X_train.append(row)
                    y_train.append(y)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        return X_train, y_train

    def transform_x(self, data):
        keys_sorted = (
            sorted(data.keys()) if not self.select_layer else [self.select_layer]
        )
        data = np.concatenate([data[s].flatten().cpu().numpy() for s in keys_sorted])
        train = np.array([np.take(data, self.most_relevant)])
        return train
