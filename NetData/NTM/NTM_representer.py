import argparse
import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RouteNTMMatrix(nn.Module):
    """
    Route/time adaptation of NetData.models.NTM.

    The original NTM mixes:
    - a GCP branch
    - a mode-product branch

    Here we keep only route/time embeddings, so the original 3D interaction is
    reduced to a 2D route-time interaction while preserving the same two-branch
    structure.
    """

    def __init__(self, num_routes, num_time, embedding_dim, k=100, c=5):
        super().__init__()
        self.route_embeddings = nn.Embedding(num_routes, embedding_dim)
        self.time_embeddings = nn.Embedding(num_time, embedding_dim)
        nn.init.xavier_uniform_(self.route_embeddings.weight)
        nn.init.xavier_uniform_(self.time_embeddings.weight)

        self.w = nn.Parameter(torch.empty(embedding_dim, k))
        self.mode_a = nn.Parameter(torch.empty(embedding_dim, c))
        self.mode_b = nn.Parameter(torch.empty(embedding_dim, c))
        self.mode_bias = nn.Parameter(torch.zeros(c, c))
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.mode_a)
        nn.init.xavier_uniform_(self.mode_b)

        self.fc = nn.Linear(c * c + k, 1)
        self.k = k
        self.c = c

    def forward(self, route_idx, time_idx):
        if route_idx.dim() == 0:
            route_idx = route_idx.view(1)
        if time_idx.dim() == 0:
            time_idx = time_idx.view(1)

        route_embeds = self.route_embeddings(route_idx)
        time_embeds = self.time_embeddings(time_idx)

        gcp = F.relu(torch.mm(route_embeds * time_embeds, self.w))

        interaction = torch.einsum("ni,nj->nij", route_embeds, time_embeds)
        interaction = torch.einsum("bij,ia->baj", interaction, self.mode_a)
        interaction = torch.einsum("baj,jc->bac", interaction, self.mode_b)
        interaction = torch.sigmoid(interaction + self.mode_bias)
        interaction = interaction.reshape(interaction.size(0), -1)

        output = torch.cat((gcp, interaction), dim=1)
        output = self.fc(output)
        return torch.squeeze(output, dim=-1)


class CustomLoss(nn.Module):
    def __init__(self, loss_type="mae"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1, 1)
        y_true = y_true.view(-1, 1)

        if self.loss_type == "mae":
            loss = torch.abs(y_pred - y_true)
        elif self.loss_type == "mse":
            loss = (y_pred - y_true) ** 2
        elif self.loss_type == "mae_mse":
            loss = torch.abs(y_pred - y_true) + 0.5 * (y_pred - y_true) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss


def collect_valid_positions(matrix_data, time_start, time_end, route_filter=None):
    if route_filter is None:
        route_filter = np.arange(matrix_data.shape[0], dtype=np.int64)
    else:
        route_filter = np.asarray(route_filter, dtype=np.int64)

    valid_positions = []
    for route_idx in route_filter:
        for time_idx in range(time_start, time_end):
            value = matrix_data[route_idx, time_idx]
            if np.isfinite(value) and value != 0:
                valid_positions.append((int(route_idx), int(time_idx), float(value)))
    return valid_positions


def random_sampling_with_representer(
    matrix_data,
    time_start,
    time_end,
    seed_num,
    sample_rate=0.8,
    min_train_samples=100,
    route_filter=None,
):
    rng = np.random.default_rng(seed_num)
    valid_positions = collect_valid_positions(matrix_data, time_start, time_end, route_filter=route_filter)

    num_valid = len(valid_positions)
    if num_valid == 0:
        return []

    sample_rate = float(np.clip(sample_rate, 0.0, 1.0))
    num_train = int(round(num_valid * sample_rate))
    num_train = max(num_train, min_train_samples)
    num_train = min(num_train, num_valid)

    if num_train == num_valid:
        return valid_positions

    selected_indices = rng.choice(num_valid, size=num_train, replace=False)
    return [valid_positions[idx] for idx in selected_indices]


def compute_sample_importance_gradient(model, route_idx, time_idx, criterion, values, batch_size=64):
    route_grad_norms = []
    time_grad_norms = []
    sample_importances = []
    sample_grads = []

    was_training = model.training
    model.eval()

    total_samples = int(route_idx.size(0))
    try:
        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_routes = route_idx[start:end]
            batch_times = time_idx[start:end]
            batch_values = values[start:end]

            with torch.enable_grad():
                predictions = model(batch_routes, batch_times)
                for offset in range(end - start):
                    loss_j = criterion(predictions[offset : offset + 1], batch_values[offset : offset + 1]).sum()
                    grads = torch.autograd.grad(
                        outputs=loss_j,
                        inputs=[model.route_embeddings.weight, model.time_embeddings.weight],
                        retain_graph=(offset < (end - start - 1)),
                        allow_unused=False,
                    )

                    route_grad = grads[0][batch_routes[offset]].detach().clone()
                    time_grad = grads[1][batch_times[offset]].detach().clone()

                    route_norm = torch.norm(route_grad).item()
                    time_norm = torch.norm(time_grad).item()

                    route_grad_norms.append(route_norm)
                    time_grad_norms.append(time_norm)
                    sample_importances.append(route_norm + time_norm)
                    sample_grads.append(torch.cat([route_grad, time_grad], dim=0).cpu().numpy())
    finally:
        model.train(was_training)

    sample_importances = np.asarray(sample_importances, dtype=np.float32)
    sample_grads = np.asarray(sample_grads, dtype=np.float32)

    mean_importance = float(sample_importances.mean()) if len(sample_importances) > 0 else 0.0
    if mean_importance > 0:
        sample_importances = sample_importances / (mean_importance + 1e-8)

    grad_info = {
        "route_grad_norm": float(np.mean(route_grad_norms)) if route_grad_norms else 0.0,
        "time_grad_norm": float(np.mean(time_grad_norms)) if time_grad_norms else 0.0,
        "mean_importance": float(sample_importances.mean()) if len(sample_importances) > 0 else 0.0,
        "std_importance": float(sample_importances.std()) if len(sample_importances) > 0 else 0.0,
        "min_importance": float(sample_importances.min()) if len(sample_importances) > 0 else 0.0,
        "max_importance": float(sample_importances.max()) if len(sample_importances) > 0 else 0.0,
    }
    return sample_importances, sample_grads, grad_info


def compute_route_similarity_from_sample_grads(
    sample_grads,
    route_indices,
    time_indices,
    num_routes,
    similarity_threshold=0.7,
):
    if len(sample_grads) == 0:
        route_similarity_mat = np.full((num_routes, num_routes), -1.0, dtype=np.float32)
        route_neighbor_counts = np.zeros(num_routes, dtype=np.int64)
        return route_similarity_mat, route_neighbor_counts

    grads_tensor = torch.tensor(sample_grads, dtype=torch.float32, device=device)
    grad_norms = torch.norm(grads_tensor, p=2, dim=1, keepdim=True).clamp_min(1e-8)
    normalized = grads_tensor / grad_norms

    route_similarity_sum = np.zeros((num_routes, num_routes), dtype=np.float64)
    route_similarity_count = np.zeros((num_routes, num_routes), dtype=np.float64)

    unique_times = np.unique(time_indices)
    for time_value in unique_times:
        sample_ids = np.where(time_indices == time_value)[0]
        if len(sample_ids) < 2:
            continue

        sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long, device=device)
        sim_block = torch.mm(normalized[sample_ids_tensor], normalized[sample_ids_tensor].T).cpu().numpy()
        route_block = route_indices[sample_ids]

        block_size = len(sample_ids)
        for i in range(block_size):
            route_i = int(route_block[i])
            for j in range(block_size):
                route_j = int(route_block[j])
                if route_i == route_j:
                    continue
                route_similarity_sum[route_i, route_j] += sim_block[i, j]
                route_similarity_count[route_i, route_j] += 1.0

    route_similarity_mat = route_similarity_sum / np.maximum(route_similarity_count, 1.0)
    route_similarity_mat = (route_similarity_mat + route_similarity_mat.T) / 2.0
    np.fill_diagonal(route_similarity_mat, -1.0)

    route_neighbor_counts = np.sum(route_similarity_mat > similarity_threshold, axis=1).astype(np.int64)
    return route_similarity_mat.astype(np.float32), route_neighbor_counts


def compute_metrics(predictions, ground_truth):
    valid_mask = np.isfinite(ground_truth) & (ground_truth != 0)
    if not valid_mask.any():
        return None

    valid_predictions = predictions[valid_mask]
    valid_ground_truth = ground_truth[valid_mask]

    mae = float(np.mean(np.abs(valid_predictions - valid_ground_truth)))
    mse = float(np.mean((valid_predictions - valid_ground_truth) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(
        np.mean(np.abs(valid_predictions - valid_ground_truth) / np.maximum(np.abs(valid_ground_truth), 1e-8))
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "valid_count": int(valid_mask.sum()),
        "total_count": int(valid_mask.size),
    }


class OnlineNTMRepresenterLearner:
    def __init__(self, matrix_data, config):
        self.matrix_data = np.asarray(matrix_data)
        if self.matrix_data.ndim != 2:
            raise ValueError("matrix_data must be a 2D numpy array")

        self.num_routes, self.num_time = self.matrix_data.shape

        self.embedding_dim = int(config.get("embedding_dim", 64))
        self.k = int(config.get("k", 100))
        self.c = int(config.get("c", 5))
        self.lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-7))
        self.epochs_per_step = int(config.get("epochs_per_step", 100))
        self.stage2_epochs = int(config.get("stage2_epochs", 100))
        self.loss_type = config.get("loss_type", "mae")
        self.patience = int(config.get("patience", 10))
        self.val_split = float(config.get("val_split", 0.2))
        self.history_start = int(config.get("history_start", 0))
        self.history_end = int(config.get("history_end", 0))
        self.sample_rate = float(config.get("sample_rate", 1.0))
        self.min_train_samples = int(config.get("min_train_samples", 100))
        self.global_seed = int(config.get("global_seed", 42))
        self.use_representer = bool(config.get("use_representer", True))
        self.use_similarity = bool(config.get("use_similarity", True))
        self.representer_method = config.get("representer_method", "gradient")
        self.route_selection_ratio = float(config.get("route_selection_ratio", 0.15))
        self.dissimilar_threshold = float(config.get("dissimilar_threshold", 0.0))
        self.similarity_threshold = float(config.get("similarity_threshold", 0.7))
        self.save_numpy_outputs = bool(config.get("save_numpy_outputs", False))
        self.save_dir = config.get("save_dir", "./online_results_ntm")
        self.top_level_dir = config.get("top_level_dir", self.save_dir)

        if self.history_end <= self.history_start:
            raise ValueError("history_end must be larger than history_start")
        if self.history_end >= self.num_time:
            raise ValueError(
                f"history_end={self.history_end} is out of range for a matrix with {self.num_time} time steps"
            )
        if not 0 < self.sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        if not 0 < self.route_selection_ratio <= 1:
            raise ValueError("route_selection_ratio must be in (0, 1]")

        self.top_route_ratio = self.route_selection_ratio * 0.8
        self.random_route_ratio = self.route_selection_ratio * 0.2
        self.exp4_importance_ratio = self.route_selection_ratio

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.top_level_dir, exist_ok=True)

        self.prediction_dirs = {}
        self.completed_full_dirs = {}
        self.completed_missing_dirs = {}
        if self.save_numpy_outputs:
            for exp_key in ("exp1", "exp2", "exp3", "exp4"):
                prediction_dir = os.path.join(self.top_level_dir, f"predictions_{exp_key}")
                completed_full_dir = os.path.join(self.top_level_dir, f"completed_full_{exp_key}")
                completed_missing_dir = os.path.join(self.top_level_dir, f"completed_missing_{exp_key}")

                os.makedirs(prediction_dir, exist_ok=True)
                os.makedirs(completed_full_dir, exist_ok=True)
                os.makedirs(completed_missing_dir, exist_ok=True)

                self.prediction_dirs[exp_key] = prediction_dir
                self.completed_full_dirs[exp_key] = completed_full_dir
                self.completed_missing_dirs[exp_key] = completed_missing_dir

        self.model = None
        self.optimizer = None
        self.training_data = None
        self.sample_importances = None
        self.route_importances = None
        self.route_similarity_mat = None
        self.route_neighbor_counts = None
        self.stage1_model_state = None

        self.train_route_indices = None
        self.train_time_indices = None
        self.train_route_indices_tensor = None
        self.train_time_indices_tensor = None
        self.train_values_tensor = None

        self.set_seed(self.global_seed)

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.global_seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def create_model(self):
        model = RouteNTMMatrix(
            num_routes=self.num_routes,
            num_time=self.num_time,
            embedding_dim=self.embedding_dim,
            k=self.k,
            c=self.c,
        ).to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model

    def _samples_to_tensors(self, samples):
        if not samples:
            return None, None, None

        route_indices = np.asarray([sample[0] for sample in samples], dtype=np.int64)
        time_indices = np.asarray([sample[1] for sample in samples], dtype=np.int64)
        values = np.asarray([sample[2] for sample in samples], dtype=np.float32)

        route_tensor = torch.tensor(route_indices, dtype=torch.long, device=device)
        time_tensor = torch.tensor(time_indices, dtype=torch.long, device=device)
        value_tensor = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)
        return route_tensor, time_tensor, value_tensor

    def prepare_training_data(self):
        samples = random_sampling_with_representer(
            self.matrix_data,
            time_start=self.history_start,
            time_end=self.history_end,
            seed_num=self.global_seed,
            sample_rate=self.sample_rate,
            min_train_samples=self.min_train_samples,
        )
        return self._samples_to_tensors(samples)

    def prepare_training_data_from_routes(self, selected_route_indices):
        samples = collect_valid_positions(
            self.matrix_data,
            time_start=self.history_start,
            time_end=self.history_end,
            route_filter=np.asarray(selected_route_indices, dtype=np.int64),
        )
        return self._samples_to_tensors(samples)

    def _split_indices(self, num_samples):
        indices = np.random.permutation(num_samples)
        if num_samples <= 1:
            return indices, indices

        num_val = max(1, int(round(num_samples * self.val_split)))
        num_val = min(num_val, num_samples - 1)
        num_train = num_samples - num_val

        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        if len(val_indices) == 0:
            val_indices = train_indices[:1]
        return train_indices, val_indices

    def _get_criterion(self):
        return CustomLoss(self.loss_type)

    def _train_current_model(self, route_indices, time_indices, values, epochs, verbose):
        if values is None:
            return None

        num_samples = int(values.size(0))
        train_indices, val_indices = self._split_indices(num_samples)

        train_route_indices = route_indices[train_indices]
        train_time_indices = time_indices[train_indices]
        train_values = values[train_indices]

        val_route_indices = route_indices[val_indices]
        val_time_indices = time_indices[val_indices]
        val_values = values[val_indices]

        train_dataset = torch.utils.data.TensorDataset(
            train_route_indices,
            train_time_indices,
            train_values,
        )

        batch_size = min(128, len(train_dataset))
        batch_size = max(batch_size, 1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        criterion = self._get_criterion()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = copy.deepcopy(self.model.state_dict())

        train_loss_history = []
        val_loss_history = []

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch_routes, batch_times, batch_values in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_routes, batch_times)
                loss = criterion(predictions, batch_values)
                total_train_loss += loss.sum().item()
                loss.mean().backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / max(len(train_dataset), 1)

            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_route_indices, val_time_indices)
                val_loss_tensor = criterion(val_predictions, val_values)
                avg_val_loss = float(val_loss_tensor.mean().item())

            train_loss_history.append(float(avg_train_loss))
            val_loss_history.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == epochs):
                print(
                    f"  epoch {epoch + 1:03d}/{epochs} | "
                    f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
                    f"best_val={best_val_loss:.6f} | patience={patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  early stopping triggered at epoch {epoch + 1}")
                break

        self.model.load_state_dict(best_state)
        self.model.train()

        return {
            "criterion": criterion,
            "train_route_indices": train_route_indices,
            "train_time_indices": train_time_indices,
            "train_values": train_values,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "final_train_loss": float(train_loss_history[-1]) if train_loss_history else None,
            "best_val_loss": float(best_val_loss),
        }

    def train_model_with_representer(self, epochs=None, verbose=True):
        if epochs is None:
            epochs = self.epochs_per_step

        if self.training_data is None:
            self.training_data = self.prepare_training_data()

        route_indices, time_indices, values = self.training_data
        if values is None:
            return None

        artifacts = self._train_current_model(route_indices, time_indices, values, epochs, verbose)
        if artifacts is None:
            return None

        self.train_route_indices_tensor = artifacts["train_route_indices"]
        self.train_time_indices_tensor = artifacts["train_time_indices"]
        self.train_values_tensor = artifacts["train_values"]

        self.train_route_indices = self.train_route_indices_tensor.detach().cpu().numpy()
        self.train_time_indices = self.train_time_indices_tensor.detach().cpu().numpy()

        if self.use_representer:
            print("  computing sample importance with gradient representers...")
            sample_importances, sample_grads, grad_info = compute_sample_importance_gradient(
                self.model,
                self.train_route_indices_tensor,
                self.train_time_indices_tensor,
                artifacts["criterion"],
                self.train_values_tensor,
            )
            self.sample_importances = sample_importances

            print(
                "  sample importance stats | "
                f"mean={grad_info['mean_importance']:.6f} | "
                f"std={grad_info['std_importance']:.6f} | "
                f"min={grad_info['min_importance']:.6f} | "
                f"max={grad_info['max_importance']:.6f}"
            )

            if self.use_similarity:
                print("  computing route similarity from time-aligned sample gradients...")
                self.route_similarity_mat, self.route_neighbor_counts = compute_route_similarity_from_sample_grads(
                    sample_grads=sample_grads,
                    route_indices=self.train_route_indices,
                    time_indices=self.train_time_indices,
                    num_routes=self.num_routes,
                    similarity_threshold=self.similarity_threshold,
                )

                if self.save_numpy_outputs:
                    np.save(os.path.join(self.save_dir, "route_similarity_mat.npy"), self.route_similarity_mat)
                    np.save(os.path.join(self.save_dir, "route_neighbor_counts.npy"), self.route_neighbor_counts)

        self.stage1_model_state = copy.deepcopy(self.model.state_dict())
        return artifacts

    def compute_route_importance(self):
        if self.sample_importances is None or self.train_route_indices is None:
            return None

        route_importances = np.zeros(self.num_routes, dtype=np.float32)
        route_counts = np.zeros(self.num_routes, dtype=np.int64)

        for idx, route_idx in enumerate(self.train_route_indices):
            route_importances[route_idx] += self.sample_importances[idx]
            route_counts[route_idx] += 1

        valid_routes = route_counts > 0
        route_importances[valid_routes] = route_importances[valid_routes] / route_counts[valid_routes]

        self.route_importances = route_importances
        if self.save_numpy_outputs:
            np.save(os.path.join(self.save_dir, "route_importances.npy"), route_importances)
            np.save(os.path.join(self.save_dir, "route_sample_count.npy"), route_counts)
        return route_importances

    def _selection_counts(self):
        total = max(1, int(round(self.num_routes * self.route_selection_ratio)))
        total = min(total, self.num_routes)

        top = max(1, int(round(total * 0.8)))
        top = min(top, total)
        other = max(0, total - top)
        return total, top, other

    def select_mixed_routes(self, route_importances, from_top_routes=False):
        _, num_top_routes, num_other_routes = self._selection_counts()
        sorted_routes = np.argsort(route_importances)
        top_routes = sorted_routes[-num_top_routes:]

        if num_other_routes == 0:
            return np.unique(top_routes), np.asarray(top_routes), np.asarray([], dtype=np.int64)

        if from_top_routes:
            important_pool_size = min(
                self.num_routes,
                max(
                    num_top_routes + num_other_routes,
                    int(round(self.num_routes * min(self.route_selection_ratio * 2.0, 1.0))),
                ),
            )
            important_pool = sorted_routes[-important_pool_size:]
            candidate_pool = np.setdiff1d(important_pool, top_routes)
            if len(candidate_pool) < num_other_routes:
                candidate_pool = np.setdiff1d(np.arange(self.num_routes), top_routes)
            self.set_seed(self.global_seed + 35)
            other_routes = np.random.choice(
                candidate_pool,
                size=min(num_other_routes, len(candidate_pool)),
                replace=False,
            )
        else:
            candidate_pool = np.setdiff1d(np.arange(self.num_routes), top_routes)
            threshold = self.dissimilar_threshold

            if self.route_similarity_mat is None:
                self.set_seed(self.global_seed + 36)
                other_routes = np.random.choice(
                    candidate_pool,
                    size=min(num_other_routes, len(candidate_pool)),
                    replace=False,
                )
            else:
                dissimilar_candidates = []
                for candidate in candidate_pool:
                    similarities = self.route_similarity_mat[candidate, top_routes]
                    if np.all(similarities < threshold):
                        dissimilar_candidates.append(candidate)

                while len(dissimilar_candidates) < num_other_routes and threshold < 1.0:
                    threshold += 0.1
                    dissimilar_candidates = []
                    for candidate in candidate_pool:
                        similarities = self.route_similarity_mat[candidate, top_routes]
                        if np.all(similarities < threshold):
                            dissimilar_candidates.append(candidate)

                if len(dissimilar_candidates) < num_other_routes:
                    dissimilar_candidates = candidate_pool.tolist()

                self.set_seed(self.global_seed + 36)
                other_routes = np.random.choice(
                    np.asarray(dissimilar_candidates, dtype=np.int64),
                    size=min(num_other_routes, len(dissimilar_candidates)),
                    replace=False,
                )

        selected_routes = np.unique(np.concatenate([top_routes, other_routes]))
        return (
            selected_routes.astype(np.int64),
            np.asarray(top_routes, dtype=np.int64),
            np.asarray(other_routes, dtype=np.int64),
        )

    def train_model_with_selected_routes(self, selected_route_indices, epochs=None, verbose=True, seed_offset=0):
        if epochs is None:
            epochs = self.stage2_epochs

        self.set_seed(self.global_seed + seed_offset)
        data = self.prepare_training_data_from_routes(selected_route_indices)
        route_indices, time_indices, values = data
        if values is None:
            return None
        return self._train_current_model(route_indices, time_indices, values, epochs, verbose)

    def predict_routes(self, route_indices, target_time_idx, batch_size=4096):
        self.model.eval()
        route_indices = np.asarray(route_indices, dtype=np.int64)
        target_times = np.full(route_indices.shape, target_time_idx, dtype=np.int64)

        predictions = []
        with torch.no_grad():
            for start in range(0, len(route_indices), batch_size):
                end = min(start + batch_size, len(route_indices))
                route_tensor = torch.tensor(route_indices[start:end], dtype=torch.long, device=device)
                time_tensor = torch.tensor(target_times[start:end], dtype=torch.long, device=device)
                batch_predictions = self.model(route_tensor, time_tensor)
                predictions.append(batch_predictions.detach().cpu().numpy())

        if not predictions:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(predictions).astype(np.float32)

    def _save_filled_outputs(self, exp_key, target_time, predictions):
        if not self.save_numpy_outputs:
            return None, None, None

        target_name = f"t{target_time}"
        predictions_file = os.path.join(self.prediction_dirs[exp_key], f"predictions_{target_name}.npy")
        np.save(predictions_file, predictions)

        completed_full = np.array(self.matrix_data, copy=True)
        completed_full[:, target_time] = predictions
        completed_full_file = os.path.join(self.completed_full_dirs[exp_key], f"completed_full_{target_name}.npy")
        np.save(completed_full_file, completed_full)

        completed_missing = np.array(self.matrix_data, copy=True)
        fill_mask = ~np.isfinite(completed_missing[:, target_time]) | (completed_missing[:, target_time] == 0)
        completed_missing[fill_mask, target_time] = predictions[fill_mask]
        completed_missing_file = os.path.join(
            self.completed_missing_dirs[exp_key],
            f"completed_missing_{target_name}.npy",
        )
        np.save(completed_missing_file, completed_missing)

        return predictions_file, completed_full_file, completed_missing_file

    @staticmethod
    def _metrics_or_nan(result, metric_name):
        if not result:
            return np.nan
        metrics = result.get("metrics")
        if not metrics:
            return np.nan
        return metrics.get(metric_name, np.nan)

    def print_experiment_comparison(self, result_exp1, result_exp2, result_exp3, result_exp4):
        print("\n" + "=" * 80)
        print("Experiment Results Comparison")
        print("=" * 80)
        print("Smaller MAE/MSE/RMSE/MAPE is better")
        print("-" * 80)

        metrics = ["mae", "mse", "rmse", "mape"]
        metric_names = ["MAE", "MSE", "RMSE", "MAPE"]
        print(f"{'Metric':<15} {'Exp1':<20} {'Exp2':<20} {'Exp3':<20} {'Exp4':<20}")
        print("-" * 100)

        for metric, name in zip(metrics, metric_names):
            val_exp1 = self._metrics_or_nan(result_exp1, metric)
            val_exp2 = self._metrics_or_nan(result_exp2, metric)
            val_exp3 = self._metrics_or_nan(result_exp3, metric)
            val_exp4 = self._metrics_or_nan(result_exp4, metric)
            print(f"{name:<15} {val_exp1:<20.6f} {val_exp2:<20.6f} {val_exp3:<20.6f} {val_exp4:<20.6f}")

    def experiment_with_routes(self, exp_key, route_indices, exp_name, use_stage1_init=True, seed_offset=0):
        print(f"\n[{exp_key}] {exp_name}")
        route_indices = np.unique(np.asarray(route_indices, dtype=np.int64))
        print(f"  selected routes: {len(route_indices)}")

        self.set_seed(self.global_seed + seed_offset)
        self.model = self.create_model()

        if use_stage1_init and self.stage1_model_state is not None:
            self.model.load_state_dict(copy.deepcopy(self.stage1_model_state))

        stage2_result = self.train_model_with_selected_routes(
            route_indices,
            epochs=self.stage2_epochs,
            verbose=True,
            seed_offset=seed_offset,
        )
        if stage2_result is None:
            return None

        target_time = self.history_end
        all_route_indices = np.arange(self.num_routes, dtype=np.int64)
        predictions = self.predict_routes(all_route_indices, target_time)
        ground_truth = self.matrix_data[:, target_time]
        metrics = compute_metrics(predictions, ground_truth)

        predictions_file, completed_full_file, completed_missing_file = self._save_filled_outputs(
            exp_key,
            target_time,
            predictions,
        )

        if metrics is None:
            print("  no valid ground-truth values at target time, predictions were still saved")
            metrics = {}
        else:
            print(
                "  metrics | "
                f"MAE={metrics['mae']:.6f} | "
                f"MSE={metrics['mse']:.6f} | "
                f"RMSE={metrics['rmse']:.6f} | "
                f"MAPE={metrics['mape']:.6f}"
            )

        result = {
            "metrics": metrics,
            "selected_routes": route_indices,
            "training_route_count": int(len(route_indices)),
            "prediction_route_count": int(self.num_routes),
            "predictions_file": predictions_file,
            "completed_full_file": completed_full_file,
            "completed_missing_file": completed_missing_file,
        }
        return result

    def save_model(self, filename):
        model_path = os.path.join(self.top_level_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "config": {
                "embedding_dim": self.embedding_dim,
                "k": self.k,
                "c": self.c,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
        }
        torch.save(checkpoint, model_path)
        return model_path

    def save_config(self):
        config_path = os.path.join(self.save_dir, "config_and_seed.txt")
        lines = [
            "NTM representer configuration",
            "=" * 60,
            f"device: {device}",
            f"timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"matrix_shape: {self.matrix_data.shape}",
            f"embedding_dim: {self.embedding_dim}",
            f"k: {self.k}",
            f"c: {self.c}",
            f"lr: {self.lr}",
            f"weight_decay: {self.weight_decay}",
            f"epochs_per_step: {self.epochs_per_step}",
            f"stage2_epochs: {self.stage2_epochs}",
            f"loss_type: {self.loss_type}",
            f"patience: {self.patience}",
            f"val_split: {self.val_split}",
            f"history_start: {self.history_start}",
            f"history_end: {self.history_end}",
            f"target_time: {self.history_end}",
            f"sample_rate: {self.sample_rate}",
            f"min_train_samples: {self.min_train_samples}",
            f"use_representer: {self.use_representer}",
            f"use_similarity: {self.use_similarity}",
            f"representer_method: {self.representer_method}",
            f"route_selection_ratio: {self.route_selection_ratio}",
            f"dissimilar_threshold: {self.dissimilar_threshold}",
            f"similarity_threshold: {self.similarity_threshold}",
            f"global_seed: {self.global_seed}",
        ]
        with open(config_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return config_path

    def run_online_learning(self):
        print("=" * 80)
        print("NTM representer matrix completion")
        print("=" * 80)
        print(f"matrix shape       : {self.matrix_data.shape}")
        print(f"history range      : [{self.history_start}, {self.history_end - 1}]")
        print(f"target time        : {self.history_end}")
        print(f"embedding dim      : {self.embedding_dim}")
        print(f"k / c              : {self.k} / {self.c}")
        print(f"sample rate        : {self.sample_rate}")
        print(f"use representer    : {self.use_representer}")
        print(f"use similarity     : {self.use_similarity}")
        print(f"route select ratio : {self.route_selection_ratio}")
        print("=" * 80)

        self.set_seed(self.global_seed)
        self.model = self.create_model()

        print("\n[stage 1] training on full history")
        stage1_result = self.train_model_with_representer(epochs=self.epochs_per_step, verbose=True)
        if stage1_result is None:
            raise RuntimeError("stage-1 training failed because no valid training samples were found")

        self.save_model("trained_model.pth")

        if self.sample_importances is not None and self.save_numpy_outputs:
            np.save(os.path.join(self.save_dir, "sample_importances.npy"), self.sample_importances)

        route_importances = self.compute_route_importance()
        if route_importances is None:
            route_importances = np.ones(self.num_routes, dtype=np.float32)

        total_selected, _, _ = self._selection_counts()

        print("\n[stage 2] route selection experiments")
        selected_routes_exp1, top_routes_exp1, other_routes_exp1 = self.select_mixed_routes(
            route_importances, from_top_routes=False
        )
        result_exp1 = self.experiment_with_routes(
            "exp1",
            selected_routes_exp1,
            "important routes + dissimilar extra routes",
            use_stage1_init=True,
            seed_offset=10,
        )

        self.set_seed(self.global_seed + 20)
        random_route_indices = np.random.choice(self.num_routes, size=total_selected, replace=False)
        result_exp2 = self.experiment_with_routes(
            "exp2",
            random_route_indices,
            "fully random routes",
            use_stage1_init=False,
            seed_offset=20,
        )

        selected_routes_exp3, top_routes_exp3, other_routes_exp3 = self.select_mixed_routes(
            route_importances, from_top_routes=True
        )
        result_exp3 = self.experiment_with_routes(
            "exp3",
            selected_routes_exp3,
            "important routes + random routes from an importance pool",
            use_stage1_init=True,
            seed_offset=30,
        )

        top_routes_exp4 = np.argsort(route_importances)[-total_selected:]
        result_exp4 = self.experiment_with_routes(
            "exp4",
            top_routes_exp4,
            "top important routes only",
            use_stage1_init=True,
            seed_offset=40,
        )

        self.print_experiment_comparison(result_exp1, result_exp2, result_exp3, result_exp4)

        experiment_results = {
            "exp1": {
                "indices": selected_routes_exp1,
                "top_routes": top_routes_exp1,
                "other_routes": other_routes_exp1,
                "importances": route_importances[selected_routes_exp1],
                "metrics": result_exp1["metrics"] if result_exp1 else None,
                "predictions_file": result_exp1["predictions_file"] if result_exp1 else None,
                "completed_full_file": result_exp1["completed_full_file"] if result_exp1 else None,
                "completed_missing_file": result_exp1["completed_missing_file"] if result_exp1 else None,
            },
            "exp2": {
                "indices": random_route_indices,
                "importances": route_importances[random_route_indices],
                "metrics": result_exp2["metrics"] if result_exp2 else None,
                "predictions_file": result_exp2["predictions_file"] if result_exp2 else None,
                "completed_full_file": result_exp2["completed_full_file"] if result_exp2 else None,
                "completed_missing_file": result_exp2["completed_missing_file"] if result_exp2 else None,
            },
            "exp3": {
                "indices": selected_routes_exp3,
                "top_routes": top_routes_exp3,
                "other_routes": other_routes_exp3,
                "importances": route_importances[selected_routes_exp3],
                "metrics": result_exp3["metrics"] if result_exp3 else None,
                "predictions_file": result_exp3["predictions_file"] if result_exp3 else None,
                "completed_full_file": result_exp3["completed_full_file"] if result_exp3 else None,
                "completed_missing_file": result_exp3["completed_missing_file"] if result_exp3 else None,
            },
            "exp4": {
                "indices": top_routes_exp4,
                "importances": route_importances[top_routes_exp4],
                "metrics": result_exp4["metrics"] if result_exp4 else None,
                "predictions_file": result_exp4["predictions_file"] if result_exp4 else None,
                "completed_full_file": result_exp4["completed_full_file"] if result_exp4 else None,
                "completed_missing_file": result_exp4["completed_missing_file"] if result_exp4 else None,
            },
        }

        self.save_config()
        return experiment_results


def append_summary(summary, target_time, experiment_results):
    summary["target_times"].append(target_time)
    for exp_key in ("exp1", "exp2", "exp3", "exp4"):
        metrics = experiment_results.get(exp_key, {}).get("metrics")
        for metric_name in ("mae", "mse", "rmse", "mape"):
            value = metrics.get(metric_name, np.nan) if metrics else np.nan
            summary[f"{exp_key}_metrics"][metric_name].append(value)


def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(values)
    if not np.any(valid):
        return np.nan
    return float(values[valid].mean())


def print_overall_summary(all_results_summary):
    print("\n" + "=" * 80)
    print("All Timepoints Summary")
    print("=" * 80)

    target_times = all_results_summary["target_times"]
    if not target_times:
        print("No valid experiment results were collected.")
        return

    print(f"target time range      : {target_times[0]} - {target_times[-1]}")
    print(f"effective experiment # : {len(target_times)}")
    print("\nAverage Metrics Comparison")

    metrics = ["mae", "mse", "rmse", "mape"]
    metric_names = ["MAE", "MSE", "RMSE", "MAPE"]
    print(f"{'Metric':<15} {'Exp1':<20} {'Exp2':<20} {'Exp3':<20} {'Exp4':<20}")
    print("-" * 100)

    for metric, name in zip(metrics, metric_names):
        exp1_avg = safe_nanmean(all_results_summary["exp1_metrics"][metric])
        exp2_avg = safe_nanmean(all_results_summary["exp2_metrics"][metric])
        exp3_avg = safe_nanmean(all_results_summary["exp3_metrics"][metric])
        exp4_avg = safe_nanmean(all_results_summary["exp4_metrics"][metric])
        print(f"{name:<15} {exp1_avg:<20.6f} {exp2_avg:<20.6f} {exp3_avg:<20.6f} {exp4_avg:<20.6f}")


def build_target_times(args, num_time):
    if args.target_time is not None:
        target_times = [args.target_time]
    elif args.target_start is not None or args.target_end is not None:
        start = args.target_start if args.target_start is not None else 1
        end = args.target_end if args.target_end is not None else start
        target_times = list(range(start, end + 1))
    else:
        if num_time > 70:
            target_times = list(range(50, min(69, num_time - 1) + 1))
        else:
            target_times = [num_time - 1]

    valid_target_times = []
    for target_time in target_times:
        if target_time <= 0:
            raise ValueError("target_time must be larger than 0 so the model has historical context")
        if target_time >= num_time:
            raise ValueError(f"target_time={target_time} is out of range for num_time={num_time}")
        valid_target_times.append(int(target_time))
    return valid_target_times


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="NTM-based representer point route selection and matrix completion"
    )
    # Abilene_12_12_3000_normalized  Geant_23_23_3000_normalized PMU_28_28_normalized Seattle_28_28_normalized
    parser.add_argument(
        "--matrix-file",
        type=str,
        default="../dataset/Abilene_12_12_3000_normalized.npy",
        help="Path to the 2D route-time matrix (.npy).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Top-level output directory.")
    parser.add_argument("--target-time", type=int, default=None, help="Single target time to predict.")
    parser.add_argument("--target-start", type=int, default=None, help="Start target time, inclusive.")
    parser.add_argument("--target-end", type=int, default=None, help="End target time, inclusive.")

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--c", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--loss-type", type=str, default="mae", choices=["mae", "mse", "mae_mse"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--min-train-samples", type=int, default=100)
    parser.add_argument("--route-selection-ratio", type=float, default=0.15)
    parser.add_argument("--dissimilar-threshold", type=float, default=0.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-representer", action="store_true")
    parser.add_argument("--disable-similarity", action="store_true")
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    matrix_file = args.matrix_file
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"matrix file does not exist: {matrix_file}")

    matrix_data = np.load(matrix_file)
    if matrix_data.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {matrix_data.shape}")

    matrix_filename = os.path.basename(matrix_file).replace(".npy", "")
    if args.output_dir is None:
        output_dir = f"./Result_NTM_{matrix_filename}_{int(args.route_selection_ratio * 100)}"
    else:
        output_dir = args.output_dir
    summary_dir = f"{output_dir}_summary"
    os.makedirs(summary_dir, exist_ok=True)

    target_times = build_target_times(args, matrix_data.shape[1])

    print(f"device      : {device}")
    print(f"matrix file : {matrix_file}")
    print(f"matrix shape: {matrix_data.shape}")
    print(f"output dir  : {output_dir}")
    print(f"targets     : {target_times}")
    print("=" * 80)

    all_results_summary = {
        "target_times": [],
        "exp1_metrics": {"mae": [], "mse": [], "rmse": [], "mape": []},
        "exp2_metrics": {"mae": [], "mse": [], "rmse": [], "mape": []},
        "exp3_metrics": {"mae": [], "mse": [], "rmse": [], "mape": []},
        "exp4_metrics": {"mae": [], "mse": [], "rmse": [], "mape": []},
    }

    overall_start = time.time()
    for index, target_time in enumerate(target_times, start=1):
        print("\n" + "=" * 80)
        print(f"timepoint {index}/{len(target_times)} | target_time={target_time}")
        print("=" * 80)

        config = {
            "embedding_dim": args.embedding_dim,
            "k": args.k,
            "c": args.c,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs_per_step": args.epochs,
            "stage2_epochs": args.stage2_epochs,
            "loss_type": args.loss_type,
            "patience": args.patience,
            "val_split": args.val_split,
            "sample_rate": args.sample_rate,
            "min_train_samples": args.min_train_samples,
            "global_seed": args.seed,
            "use_representer": not args.disable_representer,
            "use_similarity": not args.disable_similarity,
            "representer_method": "gradient",
            "route_selection_ratio": args.route_selection_ratio,
            "dissimilar_threshold": args.dissimilar_threshold,
            "similarity_threshold": args.similarity_threshold,
            "save_numpy_outputs": False,
            "history_start": 0,
            "history_end": target_time,
            "save_dir": os.path.join(output_dir, f"t{target_time}"),
            "top_level_dir": output_dir,
        }

        learner = OnlineNTMRepresenterLearner(matrix_data, config)
        experiment_results = learner.run_online_learning()
        append_summary(all_results_summary, target_time, experiment_results)

    elapsed = time.time() - overall_start
    print_overall_summary(all_results_summary)
    print(f"\nelapsed sec: {elapsed:.2f}")


if __name__ == "__main__":
    main()
