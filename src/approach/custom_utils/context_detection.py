import copy

import numpy as np
import torch
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from torch.nn.utils import prune as torch_prune

# from cuml.ensemble import RandomForestClassifier
from .prune_layer import prune_conv_layer_reuse, prune_linear_layer_reuse
from .utils import FeatureSelectionMean, SelectKBestSelector, select_composite


class BaseContextDetector:

    def __init__(
        self,
        sample_saved=100,
        features_saved=30,
        num_trees=30,
        context_detection_layer=None,
        composite="epsilon_plus_flat",
    ):
        self.sample_saved = sample_saved
        self.features_saved = features_saved
        self.num_trees = num_trees
        self.context_detection_layer = context_detection_layer
        self.context_submodels = []
        self.context_detectors = []
        self.task_pruned = {}
        self.relevances = {}
        self.val_relevances = {}
        self.composite = composite

    def update_pruned(self, new_task_pruned, model, task_lens):
        """
        Update information about the filters pruned to create each submodel for each task.
        Instantiate a new pruned model to use later for efficient relevance computation.
        """
        # First, update the stored information on pruned models
        self.task_pruned = new_task_pruned
        self.context_submodels = []
        # Then, try to recreate the models based on the pruning info
        # Since we can't be 100% sure there was nothing learnable, we have to do it
        # based on newest version of the model (to be compatible with the current face of the method)
        # There must be some error when head sizes are different
        for start_index, _ in enumerate(new_task_pruned):
            new_model = copy.deepcopy(model)
            if (
                type(getattr(new_model, new_model.head_var))
                is not torch.nn.modules.container.Sequential
            ):
                last_layer = getattr(new_model, new_model.head_var)
                dimension_1 = int(last_layer.weight.size()[1])
                task_len = int(task_lens[start_index])
                new_out = torch.nn.Linear(dimension_1, task_len)
                new_out.weight = torch.nn.Parameter(
                    copy.deepcopy(
                        last_layer.weight.detach().cpu()[
                            sum(task_lens[:start_index]) : sum(task_lens[:start_index])
                            + task_len,
                            :,
                        ]
                    )
                )
                new_out.bias = torch.nn.Parameter(
                    copy.deepcopy(
                        last_layer.bias.detach().cpu()[
                            sum(task_lens[:start_index]) : sum(task_lens[:start_index])
                            + task_len
                        ]
                    )
                )
                setattr(new_model, new_model.head_var, new_out)
            else:
                last_layer = getattr(new_model, new_model.head_var)[-1]
                dimension_1 = int(last_layer.weight.size()[1])
                task_len = int(task_lens[start_index])
                new_out = torch.nn.Linear(dimension_1, task_len)
                new_out.weight = torch.nn.Parameter(
                    copy.deepcopy(
                        last_layer.weight.detach().cpu()[
                            sum(task_lens[:start_index]) : sum(task_lens[:start_index])
                            + task_len,
                            :,
                        ]
                    )
                )
                new_out.bias = torch.nn.Parameter(
                    copy.deepcopy(
                        last_layer.bias.detach().cpu()[
                            sum(task_lens[:start_index]) : sum(task_lens[:start_index])
                            + task_len
                        ]
                    )
                )
                getattr(new_model, new_model.head_var)[-1] = new_out
            new_model = new_model.eval()
            # now prune the model according to instructions
            # ok
            new_model = new_model.to("cuda")
            model_layers = dict(new_model.named_modules())
            prune_layers = [
                (key, new_task_pruned[start_index][key])
                for key in new_task_pruned[start_index]
            ]
            for pos in prune_layers:
                if isinstance(model_layers[pos[0]], torch.nn.Conv2d):
                    new_model = prune_conv_layer_reuse(new_model, pos[0], pos[1])
                else:
                    new_model = prune_linear_layer_reuse(new_model, pos[0], pos[1])
                torch.cuda.empty_cache()
            for module in new_model.modules():
                if hasattr(module, "weight_mask"):
                    module = torch_prune.remove(module, name="weight")
                if hasattr(module, "bias_mask"):
                    module = torch_prune.remove(module, name="bias")
            self.context_submodels.append(new_model)
            torch.save(new_model, f"new_model_{start_index}.pth")

    def recompute_relevance(self, samples, task_id, val_loader=None):
        """
        Now we're moving towards data and using old models to compute relevance.
        """
        if task_id not in self.relevances:
            self.relevances[task_id] = {}
        cc = ChannelConcept()
        composite = select_composite(self.context_submodels[0], self.composite)
        layer_names = get_layer_names(
            self.context_submodels[0], [torch.nn.Conv2d, torch.nn.Linear]
        )
        for m_index, model in enumerate(self.context_submodels):
            model = copy.deepcopy(model)
            model = model.eval()
            model_inf = copy.deepcopy(model)
            model_inf = model_inf.to("cuda")
            model_inf = model_inf.eval()
            attribution = CondAttribution(model, device="cuda", no_param_grad=True)
            self.relevances[m_index][task_id] = {layer: [] for layer in layer_names}
            for data, _ in torch.utils.data.DataLoader(
                samples, batch_size=1, pin_memory=True
            ):
                data = data.to("cuda")
                with torch.no_grad():
                    data.requires_grad_(False)
                    output = model_inf(data)
                cond = output.argmax(dim=1, keepdim=True)
                # let's see whether relying only on good predictions would help
                conditions = [{"y": [int(cond.flatten()[0])]}]
                data.requires_grad_(True)
                attr = attribution(
                    data, conditions, composite, record_layer=layer_names
                )
                self.relevances[m_index][task_id] = {
                    layer: self.relevances[m_index][task_id][layer]
                    + [cc.attribute(attr.relevances[layer], abs_norm=True)]
                    for layer in layer_names
                }
            if val_loader is not None:
                if task_id not in self.val_relevances:
                    self.val_relevances[task_id] = {}
                self.val_relevances[m_index][task_id] = {
                    layer: [] for layer in layer_names
                }
                for data, _ in torch.utils.data.DataLoader(
                    val_loader.dataset, batch_size=1, pin_memory=True
                ):
                    data = data.to("cuda")
                    with torch.no_grad():
                        data.requires_grad_(False)
                        output = model_inf(data)
                    cond = output.argmax(dim=1, keepdim=True)
                    # let's see whether relying only on good predictions would help
                    conditions = [{"y": [int(cond.flatten()[0])]}]
                    data.requires_grad_(True)
                    attr = attribution(
                        data, conditions, composite, record_layer=layer_names
                    )
                    self.val_relevances[m_index][task_id] = {
                        layer: self.val_relevances[m_index][task_id][layer]
                        + [cc.attribute(attr.relevances[layer], abs_norm=True)]
                        for layer in layer_names
                    }

        torch.save(self.relevances, "computed_relevances.pth")
        torch.save(self.val_relevances, "computed_val_relevances.pth")

    def recompute_relevance_backup(self, samples, task_id, val_loader=None):
        """
        Now we're moving towards data and using old models to compute relevance.
        """
        if task_id not in self.relevances:
            self.relevances[task_id] = {}
        cc = ChannelConcept()
        composite = select_composite(self.context_submodels[0], self.composite)
        layer_names = get_layer_names(
            self.context_submodels[0], [torch.nn.Conv2d, torch.nn.Linear]
        )
        for data in torch.utils.data.DataLoader(samples, batch_size=512, shuffle=False):
            data = data[0]
            predictions = [[] for _ in range(int(data.shape[0]))]
            for index, model in enumerate(self.context_submodels):
                # First, we compute relevances on every context submodel
                # To save on computation, we will also store the prediction results
                self.relevances[index][task_id] = {layer: [] for layer in layer_names}
                model = copy.deepcopy(model)
                model = model.eval()
                # model = model.to("cuda")
                model_inf = copy.deepcopy(model)
                model_inf = model_inf.to("cuda")
                model_inf = model_inf.eval()
                data = data.to("cuda")
                with torch.no_grad():
                    data = data.requires_grad_(False)
                    output = model_inf(data)
                conds = output.argmax(dim=1, keepdim=True).flatten()
                for p in range(len(predictions)):
                    predictions[p].append(conds[p])
                data = data.requires_grad_(True)
                conds = [int(c) for c in conds]
                class_indices = {
                    i: np.where(np.array(conds) == i)[0] for i in np.unique(conds)
                }
                # here, we have to group the conds by value to make more efficient batch predictions
                for cond, indices in class_indices.items():
                    attribution = CondAttribution(
                        model, device="cuda", no_param_grad=True
                    )
                    conditions = [{"y": [int(cond)]}]
                    data_slice = torch.index_select(
                        data, 0, torch.tensor(indices).to("cuda")
                    )
                    data_slice = data_slice.requires_grad_(True)
                    attr = attribution(
                        data_slice,
                        conditions,
                        composite,
                        record_layer=layer_names,
                        on_device="cuda",
                    )
                    for local, i in enumerate(indices):
                        self.relevances[index][task_id] = {
                            layer: self.relevances[index][task_id][layer]
                            + [
                                cc.attribute(
                                    attr.relevances[layer][local].unsqueeze(0),
                                    abs_norm=True,
                                )
                            ]
                            for layer in layer_names
                        }
        if val_loader is not None:
            if task_id not in self.val_relevances:
                self.val_relevances[task_id] = {}
            for data in torch.utils.data.DataLoader(
                val_loader.dataset, batch_size=512, shuffle=False
            ):
                data = data[0]
                predictions = [[] for _ in range(int(data.shape[0]))]
                for index, model in enumerate(self.context_submodels):
                    self.val_relevances[index][task_id] = {
                        layer: [] for layer in layer_names
                    }
                    # First, we compute relevances on every context submodel
                    # To save on computation, we will also store the prediction results
                    model = copy.deepcopy(model)
                    model = model.eval()
                    # model = model.to("cuda")
                    model_inf = copy.deepcopy(model)
                    model_inf = model_inf.to("cuda")
                    data = data.to("cuda")
                    model_inf = model_inf.eval()
                    with torch.no_grad():
                        data = data.requires_grad_(False)
                        output = model_inf(data)
                    conds = output.argmax(dim=1, keepdim=True).flatten()
                    for p in range(len(predictions)):
                        predictions[p].append(conds[p])
                    data = data.requires_grad_(True)
                    conds = [int(c) for c in conds]
                    class_indices = {
                        i: np.where(np.array(conds) == i)[0] for i in np.unique(conds)
                    }
                    # here, we have to group the conds by value to make more efficient batch predictions
                    for cond, indices in class_indices.items():
                        attribution = CondAttribution(
                            model, device="cuda", no_param_grad=True
                        )
                        conditions = [{"y": [int(cond)]}]
                        data_slice = torch.index_select(
                            data, 0, torch.tensor(indices).to("cuda")
                        )
                        data_slice = data_slice.requires_grad_(True)
                        attr = attribution(
                            data_slice,
                            conditions,
                            composite,
                            record_layer=layer_names,
                            on_device="cuda",
                        )
                        for local, i in enumerate(indices):
                            self.val_relevances[index][task_id] = {
                                layer: self.val_relevances[index][task_id][layer]
                                + [
                                    cc.attribute(
                                        attr.relevances[layer][local].unsqueeze(0),
                                        abs_norm=True,
                                    )
                                ]
                                for layer in layer_names
                            }

        torch.save(self.relevances, "computed_relevances.pth")
        torch.save(self.val_relevances, "computed_val_relevances.pth")


class ExplainableContextDetector(BaseContextDetector):
    """
    A class implementing task detection based on relevance of features.
    """

    def __init__(
        self,
        sample_saved=100,
        features_saved=30,
        num_trees=30,
        feature_selection_mean=True,
        context_detection_layer=None,
        max_task_used_for_training=10,
        composite="epsilon_plus_flat",
    ):
        super().__init__(
            sample_saved=sample_saved,
            features_saved=features_saved,
            num_trees=num_trees,
            context_detection_layer=context_detection_layer,
            composite=composite,
        )
        self.feature_selection_mean = feature_selection_mean
        self.max_task_used_for_training = max_task_used_for_training
        self.feat_selectors = []
        self.all_params = {}

    def retrain_detectors(self):
        """
        Here, we will retrain detectors based on the relevance of features.
        """
        self.context_detectors = []
        self.feat_selectors = []
        last_item = len(self.relevances) - 1
        for m_index, data in self.relevances.items():
            # We have to omit the last task because we only have positive samples for it
            if m_index != last_item:
                selector = (
                    FeatureSelectionMean(
                        features_saved=self.features_saved,
                        select_layer=self.context_detection_layer,
                    )
                    if self.feature_selection_mean
                    else SelectKBestSelector(
                        features_saved=self.features_saved,
                        select_layer=self.context_detection_layer,
                    )
                )
                X, y = selector.fit_transform(data, m_index)
                if self.val_relevances:
                    # if we have validation data, we will use it to tune the hyperparameters
                    val_data = self.val_relevances[m_index]
                    X_val, y_val = selector.transform(val_data, m_index)
                    X_train = np.concatenate([X, X_val])
                    y_train = np.concatenate([y, y_val])
                    train_indices = np.arange(X.shape[0])
                    test_indices = np.arange(X_val.shape[0]) + X.shape[0]
                    search = GridSearchCV(
                        RandomForestClassifier(),
                        {
                            "n_estimators": [int(self.num_trees)],
                            "max_depth": [40],
                            "min_samples_leaf": [5],
                            "criterion": ["log_loss"],
                            "max_features": ["log2"],
                            "bootstrap": [False],
                        },
                        cv=[(train_indices, test_indices)],
                        refit=False,
                        scoring="f1",
                        n_jobs=-1,
                    ).fit(X_train, y_train)
                    print(
                        f"Best params for task {m_index}: {search.best_params_} with score {search.best_score_}"
                    )
                    detector = RandomForestClassifier(
                        **search.best_params_, n_jobs=-1
                    ).fit(X, y)
                else:
                    detector = RandomForestClassifier(
                        n_estimators=self.num_trees,
                        n_jobs=-1,
                    )
                    detector.fit(X, y)
                self.context_detectors.append(detector)
                self.feat_selectors.append(selector)

    def predict(self, test_data, task_offset):
        """
        Here, we will have to go through each subtree and try to predict the task.
        """
        results = []
        all_preds = []
        cc = ChannelConcept()
        composite = select_composite(self.context_submodels[0], self.composite)
        layer_names = get_layer_names(
            self.context_submodels[-1], [torch.nn.Conv2d, torch.nn.Linear]
        )
        for data in torch.utils.data.DataLoader(
            test_data, batch_size=1024, shuffle=False
        ):
            predictions = [[] for _ in range(int(data.shape[0]))]
            local_preds = [[] for _ in range(int(data.shape[0]))]
            local_rels = [[] for _ in range(int(data.shape[0]))]
            for index, model in enumerate(self.context_submodels):
                # First, we compute relevances on every context submodel
                # To save on computation, we will also store the prediction results
                model = copy.deepcopy(model)
                model = model.eval()
                # model = model.to("cuda")
                model_inf = copy.deepcopy(model)
                model_inf = model_inf.to("cuda")
                model_inf = model_inf.eval()
                with torch.no_grad():
                    data = data.requires_grad_(False)
                    output = model_inf(data)
                conds = output.argmax(dim=1, keepdim=True).flatten()
                for p in range(len(predictions)):
                    predictions[p].append(conds[p])
                    local_preds[p].append(output[p].view(1, -1))
                data = data.requires_grad_(True)
                conds = [int(c) for c in conds]
                class_indices = {
                    i: np.where(np.array(conds) == i)[0] for i in np.unique(conds)
                }
                # here, we have to group the conds by value to make more efficient batch predictions
                for cond, indices in class_indices.items():
                    attribution = CondAttribution(
                        model, device="cuda", no_param_grad=True
                    )
                    conditions = [{"y": [int(cond)]}]
                    data_slice = torch.index_select(
                        data, 0, torch.tensor(indices).to("cuda")
                    )
                    data_slice = data_slice.requires_grad_(True)
                    attr = attribution(
                        data_slice,
                        conditions,
                        composite,
                        record_layer=layer_names,
                        on_device="cuda",
                    )
                    for local, i in enumerate(indices):
                        local_rels[i].append(
                            {
                                layer: cc.attribute(
                                    attr.relevances[layer][local].unsqueeze(0),
                                    abs_norm=True,
                                )
                                for layer in layer_names
                            }
                        )
            probs = [[] for _ in range(int(data.shape[0]))]
            for i in range(len(probs)):
                for index, (detector, selector) in enumerate(
                    zip(self.context_detectors, self.feat_selectors)
                ):
                    sample = selector.transform_x(local_rels[i][index])
                    probs[i].append(detector.predict_proba(sample)[0])
            # Now, we have to combine the results from all detectors
            fin_probs = [
                [
                    (
                        np.prod([probs[p][j][1] for j in range(it)] + [probs[p][it][0]])
                        if it < len(self.context_detectors)
                        else np.prod([probs[p][j][1] for j in range(it)])
                    )
                    for it in range(len(self.context_submodels))
                ]
                for p in range(len(probs))
            ]
            fin = np.argmax(fin_probs, axis=1)
            pred = [
                task_offset[fin[p]] + predictions[p][fin[p]]
                for p in range(len(predictions))
            ]
            results.extend(pred)
            all_preds.extend(local_preds)
        return results, all_preds

    def predict_backup(self, test_data, task_offset):
        """
        Here, we will have to go through each subtree and try to predict the task.
        """
        results = []
        all_preds = []
        cc = ChannelConcept()
        composite = select_composite(self.context_submodels[0], self.composite)
        layer_names = get_layer_names(
            self.context_submodels[-1], [torch.nn.Conv2d, torch.nn.Linear]
        )
        for data in torch.utils.data.DataLoader(test_data, batch_size=1):
            relevances = []
            predictions = []
            local_preds = []
            for index, model in enumerate(self.context_submodels):
                # First, we compute relevances on every context submodel
                # To save on computation, we will also store the prediction results
                model = copy.deepcopy(model)
                model = model.eval()
                # model = model.to("cuda")
                model_inf = copy.deepcopy(model)
                model_inf = model_inf.to("cuda")
                model_inf = model_inf.eval()
                attribution = CondAttribution(model, device="cuda", no_param_grad=True)
                with torch.no_grad():
                    data = data.requires_grad_(False)
                    output = model_inf(data)
                cond = int(output.argmax(dim=1, keepdim=True).flatten()[0])
                conditions = [{"y": [cond]}]
                predictions.append(cond)
                local_preds.append(output)
                data = data.requires_grad_(True)
                attr = attribution(
                    data,
                    conditions,
                    composite,
                    record_layer=layer_names,
                    on_device="cuda",
                )
                relevances.append(
                    {
                        layer: cc.attribute(attr.relevances[layer], abs_norm=True)
                        for layer in layer_names
                    }
                )
            probs = []
            for index, (detector, selector) in enumerate(
                zip(self.context_detectors, self.feat_selectors)
            ):
                sample = selector.transform_x(relevances[index])
                probs.append(detector.predict_proba(sample)[0])
            # Now, we have to combine the results from all detectors
            fin_probs = [
                (
                    np.prod([probs[j][1] for j in range(it)] + [probs[it][0]])
                    if it < len(self.context_detectors)
                    else np.prod([probs[j][1] for j in range(it)])
                )
                for it in range(len(self.context_submodels))
            ]
            fin = np.argmax(fin_probs)
            pred = task_offset[fin] + predictions[fin]
            results.append(pred)
            all_preds.append(local_preds)
        return results, all_preds
