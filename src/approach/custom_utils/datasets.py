import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassIncrTraining:

    def __init__(
        self,
        to_keep=["0", "1"],
        to_train=["6", "7", "8", "9"],
        continuous_trained=["0", "1", "2", "3", "4", "5"],
    ) -> None:
        self.to_keep = [int(c) for c in to_keep]
        self.to_train = [int(c) for c in to_train]
        self.continuous_trained = [int(c) for c in continuous_trained]

    def split_head_target(self, outputs, targets):
        # first, get the indices for which separate computation will take place
        t1 = np.concatenate(
            [np.argwhere(targets.cpu() == k).flatten() for k in self.to_keep]
        )
        t2 = np.setdiff1d([i for i in range(len(targets))], t1)
        # now, use it to select the right outputs and targets
        outputs1 = outputs[t1]
        outputs2 = outputs[t2]
        targets1 = targets[t1]
        targets2 = targets[t2]
        # now, let's change the target values accordingly now
        changed_vals = np.setdiff1d(self.continuous_trained, self.to_keep)
        targets2 = torch.IntTensor(
            np.array([np.argwhere(changed_vals == int(t)) for t in targets2]).flatten()
        )
        return [outputs1, outputs2], [targets1, targets2]

    def train(
        self,
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        multi_label=False,
        scheduler=None,
        weights=None,
    ):
        model.train()
        total_loss = 0
        if not weights:
            weights = [
                1 if i not in self.to_keep else 0 for i in self.continuous_trained
            ]
        transformed_weights = torch.FloatTensor(weights).to(device)
        loss_prod = nn.CrossEntropyLoss(weight=transformed_weights)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if multi_label:
                target = target.squeeze(-1)
            loss = loss_prod(output, target)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        return total_loss

    def precision_per_class(
        self, model, device, test_loader, classes_to_check=[0, 1], multi_label=False
    ):
        model.eval()
        prec_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in classes_to_check:
                    if i not in prec_dict:
                        prec_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        inter_len = len(np.intersect1d(pred_class, target_class))
                        prec_dict[i][0] += inter_len
                        prec_dict[i][1] += len(pred_class)
        prec_dict = {
            i: 100 * prec_dict[i][0] / prec_dict[i][1] if prec_dict[i][1] > 0 else 0
            for i in prec_dict
        }
        prec_score = np.mean(np.array([prec_dict[k] for k in prec_dict]))
        print(f"Checking class accuracies are {prec_dict}")
        return prec_dict, prec_score

    def test_per_class(self, model, device, test_loader, multi_label=False):
        model.eval()
        acc_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in range(len(output[0])):
                    if i not in acc_dict:
                        acc_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                        acc_dict[i][1] += len(target_class)
        acc_dict = {
            i: 100 * acc_dict[i][0] / acc_dict[i][1] if acc_dict[i][1] > 0 else 0
            for i in acc_dict
        }
        print(f"Checking class accuracies are {acc_dict}")
        return acc_dict

    def test(self, model, device, test_loader, multi_label=False, loss=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        acc_dict = self.test_per_class(
            model, device, test_loader, multi_label=multi_label
        )
        test_loss /= total

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )
        if loss:
            return loss, 100.0 * correct / total, acc_dict
        else:
            return 100.0 * correct / total

    def get_per_class(self, output, target, acc_dict={}):
        if len(output) > 0:
            for i in range(len(output[0])):
                if i not in acc_dict:
                    acc_dict[i] = [0, 0]
                if i in target.cpu().detach().flatten():
                    pred_class = np.where(
                        output.cpu().detach().numpy().argmax(axis=1, keepdims=True) == i
                    )[0]
                    target_class = np.where(target.cpu().detach().numpy() == i)[0]
                    acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                    acc_dict[i][1] += len(target_class)
        return acc_dict

    def test_per_head(self, model, device, test_loader, multi_label=False):
        model.eval()
        # here, we assume that the head contains a part responsible for classes in to_keep and to_train
        head_labels = [
            [i for i in self.to_keep],
            [i for i in self.continuous_trained if i not in self.to_keep],
        ]
        head_correct = [0 for _ in head_labels]
        head_total = [0 for _ in head_labels]
        head_dicts = [{} for _ in head_labels]
        with torch.no_grad():
            for data, target in test_loader:
                full_target = target
                if multi_label:
                    full_target = full_target.squeeze(-1)
                data, full_target = data.to(device), full_target.to(device)
                output = model(data)
                # now we will have to compute metrics for each head
                head_outputs, head_targets = self.split_head_target(output, full_target)
                for i in range(len(head_labels)):
                    head_target = head_targets[i].to(device)
                    head_output = head_outputs[i][
                        :, head_labels[i]
                    ]  # sum up batch loss
                    pred = head_output.argmax(
                        dim=1, keepdim=True
                    )  # get the index of the max log-probability
                    head_correct[i] += pred.eq(head_target.view_as(pred)).sum().item()
                    head_total[i] += len(head_target)
                    head_dicts[i] = self.get_per_class(
                        head_output, head_target, head_dicts[i]
                    )
        accuracies = [
            100 * head_correct[i] / head_total[i] if head_total[i] > 0 else -1
            for i in range(len(head_correct))
        ]
        head_dicts = [
            {
                i: 100 * acc_dict[i][0] / acc_dict[i][1] if acc_dict[i][1] > 0 else 0
                for i in acc_dict
            }
            for acc_dict in head_dicts
        ]
        print(f"Checking class accuracies are {head_dicts}")
        print(
            "\nTest set: Accuracy of head 1: {}/{} ({:.0f}%) Accuracy of head 2: {}/{} ({:.0f}%)\n".format(
                head_correct[0],
                head_total[0],
                accuracies[0],
                head_correct[1],
                head_total[1],
                accuracies[1],
            )
        )
        return accuracies, head_dicts

    def test_perf(self, model, device, test_loader, multi_label=False):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                # target = self.reassign_target(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        return correct / len(test_loader.dataset) * 100


class DualDatasetTraining:

    def __init__(
        self,
        to_keep=["0", "1"],
        to_train=["6", "7", "8", "9"],
        continuous_trained=["0", "1", "2", "3", "4", "5"],
    ) -> None:
        self.to_keep = [int(c) for c in to_keep]
        self.to_train = [int(c) for c in to_train]
        self.continuous_trained = [int(c) for c in continuous_trained]

    def reassign_target(self, targets):
        indices_taken = np.array(
            [
                np.argwhere(np.array(self.continuous_trained) == int(k))
                for k in self.to_keep
            ]
        ).flatten()
        indices_free = [
            i for i in range(len(self.continuous_trained)) if i not in indices_taken
        ]
        for index, target in enumerate(self.to_train):
            if target not in self.continuous_trained:
                targets[targets == target] = self.continuous_trained[
                    indices_free[index]
                ]
        return targets

    def split_head_target(self, outputs, targets):
        # first, get the indices for which separate computation will take place
        t1 = np.concatenate(
            [np.argwhere(targets.cpu() == k).flatten() for k in self.to_keep]
        )
        t2 = np.setdiff1d([i for i in range(len(targets))], t1)
        # now, use it to select the right outputs and targets
        outputs1 = outputs[t1]
        outputs2 = outputs[t2]
        targets1 = targets[t1]
        targets2 = targets[t2]
        # now, let's change the target values accordingly now
        changed_vals = np.setdiff1d(self.continuous_trained, self.to_keep)
        targets2 = torch.IntTensor(
            np.array([np.argwhere(changed_vals == int(t)) for t in targets2]).flatten()
        )
        return [outputs1, outputs2], [targets1, targets2]

    def train(
        self,
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        multi_label=False,
        scheduler=None,
        weights=None,
    ):
        model.train()
        total_loss = 0
        if not weights:
            weights = [
                1 if i not in self.to_keep else 0 for i in self.continuous_trained
            ]
        transformed_weights = torch.FloatTensor(weights).to(device)
        loss_prod = nn.CrossEntropyLoss(weight=transformed_weights)
        for batch_idx, (data, target) in enumerate(train_loader):
            target = self.reassign_target(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if multi_label:
                target = target.squeeze(-1)
            loss = loss_prod(output, target)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        return total_loss

    def precision_per_class(
        self, model, device, test_loader, classes_to_check=[0, 1], multi_label=False
    ):
        model.eval()
        prec_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in classes_to_check:
                    if i not in prec_dict:
                        prec_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        inter_len = len(np.intersect1d(pred_class, target_class))
                        prec_dict[i][0] += inter_len
                        prec_dict[i][1] += len(pred_class)
        prec_dict = {
            i: 100 * prec_dict[i][0] / prec_dict[i][1] if prec_dict[i][1] > 0 else 0
            for i in prec_dict
        }
        prec_score = np.mean(np.array([prec_dict[k] for k in prec_dict]))
        print(f"Checking class accuracies are {prec_dict}")
        return prec_dict, prec_score

    def test_per_class(self, model, device, test_loader, multi_label=False):
        model.eval()
        acc_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in range(len(output[0])):
                    if i not in acc_dict:
                        acc_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                        acc_dict[i][1] += len(target_class)
        acc_dict = {
            i: 100 * acc_dict[i][0] / acc_dict[i][1] if acc_dict[i][1] > 0 else 0
            for i in acc_dict
        }
        print(f"Checking class accuracies are {acc_dict}")
        return acc_dict

    def test(self, model, device, test_loader, multi_label=False, loss=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        acc_dict = self.test_per_class(
            model, device, test_loader, multi_label=multi_label
        )
        test_loss /= total

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )
        if loss:
            return loss, 100.0 * correct / total, acc_dict
        else:
            return 100.0 * correct / total

    def get_per_class(self, output, target, acc_dict={}):
        if len(output) > 0:
            for i in range(len(output[0])):
                if i not in acc_dict:
                    acc_dict[i] = [0, 0]
                if i in target.cpu().detach().flatten():
                    pred_class = np.where(
                        output.cpu().detach().numpy().argmax(axis=1, keepdims=True) == i
                    )[0]
                    target_class = np.where(target.cpu().detach().numpy() == i)[0]
                    acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                    acc_dict[i][1] += len(target_class)
        return acc_dict

    def test_per_head(self, model, device, test_loader, multi_label=False):
        model.eval()
        # here, we assume that the head contains a part responsible for classes in to_keep and to_train
        head_labels = [
            [i for i in self.to_keep],
            [i for i in self.continuous_trained if i not in self.to_keep],
        ]
        head_correct = [0 for _ in head_labels]
        head_total = [0 for _ in head_labels]
        head_dicts = [{} for _ in head_labels]
        with torch.no_grad():
            for data, target in test_loader:
                full_target = self.reassign_target(target)
                if multi_label:
                    full_target = full_target.squeeze(-1)
                data, full_target = data.to(device), full_target.to(device)
                output = model(data)
                # now we will have to compute metrics for each head
                head_outputs, head_targets = self.split_head_target(output, full_target)
                for i in range(len(head_labels)):
                    head_target = head_targets[i].to(device)
                    head_output = head_outputs[i][
                        :, head_labels[i]
                    ]  # sum up batch loss
                    pred = head_output.argmax(
                        dim=1, keepdim=True
                    )  # get the index of the max log-probability
                    head_correct[i] += pred.eq(head_target.view_as(pred)).sum().item()
                    head_total[i] += len(head_target)
                    head_dicts[i] = self.get_per_class(
                        head_output, head_target, head_dicts[i]
                    )
        accuracies = [
            100 * head_correct[i] / head_total[i] if head_total[i] > 0 else -1
            for i in range(len(head_correct))
        ]
        head_dicts = [
            {
                i: 100 * acc_dict[i][0] / acc_dict[i][1] if acc_dict[i][1] > 0 else 0
                for i in acc_dict
            }
            for acc_dict in head_dicts
        ]
        print(f"Checking class accuracies are {head_dicts}")
        print(
            "\nTest set: Accuracy of head 1: {}/{} ({:.0f}%) Accuracy of head 2: {}/{} ({:.0f}%)\n".format(
                head_correct[0],
                head_total[0],
                accuracies[0],
                head_correct[1],
                head_total[1],
                accuracies[1],
            )
        )
        return accuracies, head_dicts

    def test_perf(self, model, device, test_loader, multi_label=False):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                # target = self.reassign_target(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        return correct / len(test_loader.dataset) * 100


class MNISTLikeDatasetTraining:

    def __init__(
        self,
        to_keep=["0", "1"],
        to_train=["6", "7", "8", "9"],
        continuous_trained=["0", "1", "2", "3", "4", "5"],
    ) -> None:
        self.to_keep = [int(c) for c in to_keep]
        self.to_train = [int(c) for c in to_train]
        self.continuous_trained = [int(c) for c in continuous_trained]

    def reassign_target(self, targets):
        for target in self.to_train:
            if target not in self.to_keep:
                targets[targets == target] = (
                    target - len(self.continuous_trained) + len(self.to_keep)
                )
        return targets

    def train(
        self,
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        mnist_split=True,
        mma_loss=True,
        extra_layer=None,
        coefficient=1,
        multi_label=False,
        scheduler=None,
        front_layers=None,
    ):
        model.train()
        total_loss = 0
        loss_prod = nn.CrossEntropyLoss()
        for batch_idx, (data, target) in enumerate(train_loader):
            if mnist_split:
                target[target % 2 == 0] = 0
                target[target % 2 != 0] = 1
            target = self.reassign_target(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_prod = nn.CrossEntropyLoss()
            if multi_label:
                target = target.squeeze(-1)
            loss = loss_prod(output, target)
            total_loss += loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        return total_loss

    def precision_per_class(
        self,
        model,
        device,
        test_loader,
        classes_to_check=[0, 1],
        mnist_split=True,
        multi_label=False,
    ):
        model.eval()
        prec_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                if mnist_split:
                    target[target % 2 == 0] = 0
                    target[target % 2 != 0] = 1
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in classes_to_check:
                    if i not in prec_dict:
                        prec_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        inter_len = len(np.intersect1d(pred_class, target_class))
                        prec_dict[i][0] += inter_len
                        prec_dict[i][1] += len(pred_class)
        prec_dict = {
            i: 100 * prec_dict[i][0] / prec_dict[i][1] if prec_dict[i][1] > 0 else 0
            for i in prec_dict
        }
        prec_score = np.mean(np.array([prec_dict[k] for k in prec_dict]))
        print(f"Checking class accuracies are {prec_dict}")
        return prec_dict, prec_score

    def test_per_class(
        self, model, device, test_loader, mnist_split=True, multi_label=False
    ):
        model.eval()
        acc_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                if mnist_split:
                    target[target % 2 == 0] = 0
                    target[target % 2 != 0] = 1
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in range(len(output[0])):
                    if i not in acc_dict:
                        acc_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                        acc_dict[i][1] += len(target_class)
        acc_dict = {
            i: 100 * acc_dict[i][0] / acc_dict[i][1] if acc_dict[i][1] > 0 else 0
            for i in acc_dict
        }
        print(f"Checking class accuracies are {acc_dict}")
        return acc_dict

    def test(
        self,
        model,
        device,
        test_loader,
        mnist_split=True,
        multi_label=False,
        loss=False,
    ):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                if mnist_split:
                    target[target % 2 == 0] = 0
                    target[target % 2 != 0] = 1
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        acc_dict = self.test_per_class(
            model, device, test_loader, mnist_split=mnist_split, multi_label=multi_label
        )
        test_loss /= total

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )
        if loss:
            return loss, 100.0 * correct / total, acc_dict
        else:
            return 100.0 * correct / total

    def test_per_head(
        self,
        model,
        device,
        test_loader,
        mnist_split=True,
        multi_label=False,
        loss=False,
        eval_weights=[0, 0, 1, 1, 1, 1],
    ):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                if mnist_split:
                    target[target % 2 == 0] = 0
                    target[target % 2 != 0] = 1
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        acc_dict = self.test_per_class(
            model, device, test_loader, mnist_split=mnist_split, multi_label=multi_label
        )
        test_loss /= total

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )
        if loss:
            return loss, 100.0 * correct / total, acc_dict
        else:
            return 100.0 * correct / total

    def test_perf(
        self,
        model,
        device,
        test_loader,
        mnist_split=True,
        multi_label=False,
        reassign=False,
    ):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if mnist_split:
                    target[target % 2 == 0] = 0
                    target[target % 2 != 0] = 1
                if multi_label:
                    target = target.squeeze(-1)
                if reassign:
                    target = self.reassign_target(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        return correct / len(test_loader.dataset) * 100


class MedLikeDatasetTraining:

    def __init__(self, to_train=[0, 3, 4]) -> None:
        self.to_train = to_train
        self.translated_labels = {to_train[i]: i for i in range(len(to_train))}

    def reassign_target(self, targets):
        for target in self.to_train:
            targets[targets == target] = self.translated_labels[target]
        return targets

    def train(
        self,
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        mma_loss=True,
        extra_layer=None,
        coefficient=1,
        multi_label=False,
        scheduler=None,
    ):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = self.reassign_target(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_prod = nn.CrossEntropyLoss()
            if multi_label:
                target = target.squeeze(-1)
            loss = loss_prod(output, target)
            total_loss += loss
            if mma_loss:
                # try to add bias to make the whole thing as different as possible
                full_weight = torch.nn.Parameter(
                    torch.cat((extra_layer.weight, model.out.weight), 0)
                )
                extra_loss = get_cosine_embedding_loss(full_weight)
                loss += coefficient * extra_loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        return total_loss

    def test_per_class(self, model, device, test_loader, multi_label=False):
        model.eval()
        acc_dict = {}
        with torch.no_grad():
            for data, target in test_loader:
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                for i in range(len(output[0])):
                    if i not in acc_dict:
                        acc_dict[i] = [0, 0]
                    if i in target.cpu().detach().flatten():
                        pred_class = np.where(
                            output.cpu().detach().numpy().argmax(axis=1, keepdims=True)
                            == i
                        )[0]
                        target_class = np.where(target.cpu().detach().numpy() == i)[0]
                        acc_dict[i][0] += len(np.intersect1d(pred_class, target_class))
                        acc_dict[i][1] += len(target_class)
        acc_dict = {i: 100 * acc_dict[i][0] / acc_dict[i][1] for i in acc_dict}
        print(f"Checking class accuracies are {acc_dict}")
        return acc_dict

    def test(self, model, device, test_loader, multi_label=False, loss=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = self.reassign_target(target)
                if multi_label:
                    target = target.squeeze(-1)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        acc_dict = self.test_per_class(
            model, device, test_loader, multi_label=multi_label
        )
        test_loss /= total

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, total, 100.0 * correct / total
            )
        )
        if loss:
            return loss, 100.0 * correct / total, acc_dict
        else:
            return 100.0 * correct / total

    def test_perf(self, model, device, test_loader, multi_label=False):
        model.eval()
        model.to(device)
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if multi_label:
                    target = target.squeeze(-1)
                # target = self.reassign_target(target)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        return correct / len(test_loader.dataset) * 100


def permute_mnist(mnist, seed):
    """Given the training set, permute pixels of each img the same way."""

    np.random.seed(seed)
    h = w = 28
    perm_inds = list(range(h * w))
    np.random.shuffle(perm_inds)
    # print(perm_inds)
    perm_mnist = []
    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
    return perm_mnist
