from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import (
    EPISODE_COLUMN,
    HOUR_COLUMN,
    TARGET_COLUMN,
    add_derived_clinical_features,
)

SEQUENCE_DYNAMIC_NUMERIC_COLUMNS = (
    "hour_from_admission",
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "mobility_score",
    "nurse_alert",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "sepsis_risk_score",
    "mean_arterial_pressure",
    "pulse_pressure",
    "shock_index",
    "spo2_deficit",
)

SEQUENCE_DYNAMIC_CATEGORICAL_COLUMNS = ("oxygen_device",)
SEQUENCE_STATIC_NUMERIC_COLUMNS = ("age", "comorbidity_index")
SEQUENCE_STATIC_CATEGORICAL_COLUMNS = ("gender", "admission_type")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def prepare_sequence_frame(df: pd.DataFrame) -> pd.DataFrame:
    return add_derived_clinical_features(df)


def resolve_device(preferred_device: str = "auto") -> torch.device:
    if preferred_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred_device)


@dataclass
class SequencePreprocessor:
    dynamic_means: dict[str, float]
    dynamic_stds: dict[str, float]
    static_means: dict[str, float]
    static_stds: dict[str, float]
    categories: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequencePreprocessor":
        return cls(**payload)


def fit_sequence_preprocessor(train_frame: pd.DataFrame) -> SequencePreprocessor:
    dynamic_stats = train_frame[list(SEQUENCE_DYNAMIC_NUMERIC_COLUMNS)]
    static_stats = train_frame[list(SEQUENCE_STATIC_NUMERIC_COLUMNS)]

    dynamic_stds = dynamic_stats.std().replace(0.0, 1.0)
    static_stds = static_stats.std().replace(0.0, 1.0)

    categories = {}
    for column in (*SEQUENCE_DYNAMIC_CATEGORICAL_COLUMNS, *SEQUENCE_STATIC_CATEGORICAL_COLUMNS):
        categories[column] = sorted(train_frame[column].astype(str).unique().tolist())

    return SequencePreprocessor(
        dynamic_means=dynamic_stats.mean().to_dict(),
        dynamic_stds=dynamic_stds.to_dict(),
        static_means=static_stats.mean().to_dict(),
        static_stds=static_stds.to_dict(),
        categories=categories,
    )


def save_preprocessor(path: str | Path, preprocessor: SequencePreprocessor) -> None:
    Path(path).write_text(pd.Series(preprocessor.to_dict()).to_json(indent=2), encoding="utf-8")


def load_preprocessor(path: str | Path) -> SequencePreprocessor:
    return SequencePreprocessor.from_dict(pd.read_json(path, typ="series").to_dict())


def _encode_one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
    category_map = {category: index for index, category in enumerate(categories)}
    encoded = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row_index, value in enumerate(values.astype(str).tolist()):
        encoded[row_index, category_map.get(value, 0)] = 1.0
    return encoded


def build_episode_store(
    frame: pd.DataFrame,
    preprocessor: SequencePreprocessor,
) -> dict[int, dict[str, np.ndarray]]:
    store: dict[int, dict[str, np.ndarray]] = {}
    grouped = frame.groupby(EPISODE_COLUMN, sort=False)

    dynamic_mean = pd.Series(preprocessor.dynamic_means)
    dynamic_std = pd.Series(preprocessor.dynamic_stds)
    static_mean = pd.Series(preprocessor.static_means)
    static_std = pd.Series(preprocessor.static_stds)

    for episode_id, episode_df in grouped:
        dynamic_numeric = episode_df.loc[:, list(SEQUENCE_DYNAMIC_NUMERIC_COLUMNS)].astype("float32")
        dynamic_numeric = ((dynamic_numeric - dynamic_mean) / dynamic_std).to_numpy(dtype=np.float32)

        dynamic_parts = [dynamic_numeric]
        for column in SEQUENCE_DYNAMIC_CATEGORICAL_COLUMNS:
            dynamic_parts.append(
                _encode_one_hot(episode_df[column], preprocessor.categories[column]),
            )
        dynamic = np.concatenate(dynamic_parts, axis=1).astype(np.float32)

        static_row = episode_df.iloc[0]
        static_numeric = (
            (static_row.loc[list(SEQUENCE_STATIC_NUMERIC_COLUMNS)].astype("float32") - static_mean)
            / static_std
        ).to_numpy(dtype=np.float32)

        static_parts = [static_numeric]
        for column in SEQUENCE_STATIC_CATEGORICAL_COLUMNS:
            static_parts.append(
                _encode_one_hot(pd.Series([static_row[column]]), preprocessor.categories[column])[0],
            )
        static = np.concatenate(static_parts, axis=0).astype(np.float32)

        labels = (
            episode_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
            if TARGET_COLUMN in episode_df.columns
            else np.zeros(len(episode_df), dtype=np.float32)
        )
        hours = episode_df[HOUR_COLUMN].to_numpy(dtype=np.int16)

        store[int(episode_id)] = {
            "dynamic": dynamic,
            "static": static,
            "labels": labels,
            "hours": hours,
        }

    return store


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        episode_store: dict[int, dict[str, np.ndarray]],
        max_seq_len: int,
    ) -> None:
        self.episode_store = episode_store
        self.max_seq_len = max_seq_len
        self.sample_index: list[tuple[int, int, int, float]] = []

        for episode_id, payload in episode_store.items():
            hours = payload["hours"]
            labels = payload["labels"]
            for step_index in range(len(hours)):
                self.sample_index.append(
                    (episode_id, step_index, int(hours[step_index]), float(labels[step_index])),
                )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, step_index, hour, label = self.sample_index[index]
        payload = self.episode_store[episode_id]
        dynamic = payload["dynamic"]
        static = payload["static"]

        start_index = max(0, step_index - self.max_seq_len + 1)
        window = dynamic[start_index : step_index + 1]
        sequence_length = window.shape[0]

        sequence = np.zeros((self.max_seq_len, dynamic.shape[1]), dtype=np.float32)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        sequence[-sequence_length:] = window
        mask[-sequence_length:] = True

        return {
            "sequence": torch.from_numpy(sequence),
            "mask": torch.from_numpy(mask),
            "static": torch.from_numpy(static),
            "label": torch.tensor(label, dtype=torch.float32),
            "episode_id": torch.tensor(episode_id, dtype=torch.int32),
            "hour": torch.tensor(hour, dtype=torch.int16),
        }

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                EPISODE_COLUMN: [item[0] for item in self.sample_index],
                HOUR_COLUMN: [item[2] for item in self.sample_index],
                TARGET_COLUMN: [item[3] for item in self.sample_index],
            },
        )


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(values).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)


class GRUAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=True,
        )
        self.attention = AttentionPooling(hidden_size * 2)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear((hidden_size * 2) + 32, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(sequence)
        context = self.attention(outputs, mask)
        static_context = self.static_mlp(static)
        logits = self.classifier(torch.cat([context, static_context], dim=1))
        return logits.squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return tensor
        return tensor[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(tensor) + self.residual(tensor))


class TCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        channels: tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        blocks = []
        for level, out_channels in enumerate(channels):
            in_channels = input_dim if level == 0 else channels[level - 1]
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                ),
            )
        self.tcn = nn.Sequential(*blocks)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] + 32, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(sequence.transpose(1, 2)).transpose(1, 2)
        last_index = mask.long().sum(dim=1).clamp(min=1) - 1
        last_features = encoded[torch.arange(encoded.size(0), device=encoded.device), last_index]
        static_context = self.static_mlp(static)
        logits = self.classifier(torch.cat([last_features, static_context], dim=1))
        return logits.squeeze(-1)


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        max_seq_len: int,
        d_model: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(d_model)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0)
        encoded = self.input_projection(self.input_norm(sequence)) + self.position_embedding(positions)
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        pooled_features = self.pooling(encoded, mask)
        static_context = self.static_mlp(static)
        logits = self.classifier(torch.cat([pooled_features, static_context], dim=1))
        return logits.squeeze(-1)


def build_model(
    model_name: str,
    input_dim: int,
    static_dim: int,
    max_seq_len: int,
    config_override: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    model_name = model_name.lower()
    config_override = config_override or {}
    if model_name == "gru_attention":
        config = {"hidden_size": 96, "num_layers": 2, "dropout": 0.2}
        config.update(config_override)
        return GRUAttentionModel(input_dim=input_dim, static_dim=static_dim, **config), config
    if model_name == "tcn":
        config = {"channels": (64, 64, 64), "kernel_size": 3, "dropout": 0.15}
        config.update(config_override)
        return TCNModel(input_dim=input_dim, static_dim=static_dim, **config), config
    if model_name == "transformer_encoder":
        config = {
            "max_seq_len": max_seq_len,
            "d_model": 96,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.15,
        }
        config.update(config_override)
        return TransformerEncoderModel(input_dim=input_dim, static_dim=static_dim, **config), config
    raise ValueError(f"Unsupported model: {model_name}")


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if key in {"episode_id", "hour"}:
            moved[key] = value
        else:
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
    return moved


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    hours: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(moved["sequence"], moved["static"], moved["mask"])
            scores = torch.sigmoid(logits).float().cpu().numpy()
            predictions.append(scores)
            labels.append(batch["label"].cpu().numpy())
            episodes.append(batch["episode_id"].cpu().numpy())
            hours.append(batch["hour"].cpu().numpy())

    score_array = np.concatenate(predictions)
    label_array = np.concatenate(labels)
    episode_array = np.concatenate(episodes)
    hour_array = np.concatenate(hours)
    metrics = compute_binary_metrics(label_array, score_array)
    frame = pd.DataFrame(
        {
            EPISODE_COLUMN: episode_array.astype(int),
            HOUR_COLUMN: hour_array.astype(int),
            TARGET_COLUMN: label_array.astype(int),
            "risk_score": score_array.astype(np.float32),
        },
    )
    return metrics, frame


def train_sequence_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    holdout_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    pos_weight: float,
    scheduler_name: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_score = -np.inf
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / f"{model_name}.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            moved = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(moved["sequence"], moved["static"], moved["mask"])
                loss = criterion(logits, moved["label"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        val_metrics, val_predictions = evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **val_metrics,
            },
        )

        if val_metrics["pr_auc"] > best_score:
            best_score = val_metrics["pr_auc"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": copy.deepcopy(model.state_dict()),
                },
                checkpoint_path,
            )
            val_predictions.to_csv(output_dir / f"{model_name}_val_predictions.csv", index=False)
        else:
            wait += 1
            if wait >= patience:
                break

        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    best_val_metrics, val_predictions = evaluate_model(model, val_loader, device)
    holdout_metrics, holdout_predictions = evaluate_model(model, holdout_loader, device)

    pd.DataFrame(history).to_csv(output_dir / f"{model_name}_history.csv", index=False)
    val_predictions.to_csv(output_dir / f"{model_name}_val_predictions.csv", index=False)
    holdout_predictions.to_csv(output_dir / f"{model_name}_holdout_predictions.csv", index=False)

    return {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "holdout_metrics": holdout_metrics,
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "history_path": str((output_dir / f"{model_name}_history.csv").as_posix()),
        "val_prediction_path": str((output_dir / f"{model_name}_val_predictions.csv").as_posix()),
        "holdout_prediction_path": str((output_dir / f"{model_name}_holdout_predictions.csv").as_posix()),
    }
