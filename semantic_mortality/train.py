from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from .config import Config, RunPaths, save_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_from_config(device_value: str) -> torch.device:
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def _build_dataloaders(
    config: Config, batch_size_override: int | None = None
) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config.data
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    download = data_cfg.get("download", False)
    train_dataset = torchvision.datasets.Places365(
        root=data_cfg["dataset_root"],
        split=data_cfg["train_split"],
        small=data_cfg["small"],
        download=download,
        transform=train_tf,
    )
    val_dataset = torchvision.datasets.Places365(
        root=data_cfg["dataset_root"],
        split=data_cfg["val_split"],
        small=data_cfg["small"],
        download=download,
        transform=val_tf,
    )

    max_train_samples = data_cfg.get("max_train_samples")
    if max_train_samples and max_train_samples < len(train_dataset):
        indices = random.sample(range(len(train_dataset)), max_train_samples)
        train_dataset = Subset(train_dataset, indices)

    batch_size = batch_size_override or config.training["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def _build_model(config: Config) -> nn.Module:
    pretrained = config.training.get("pretrained", True)
    if pretrained:
        weights = torchvision.models.ResNet18_Weights.DEFAULT
    else:
        weights = None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 365)
    return model


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "loss": running_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_and_checkpoint(config: Config, paths: RunPaths) -> None:
    _set_seed(config.data.get("seed", 42))
    device = _device_from_config(config.training.get("device", "auto"))

    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, paths.config_path)

    train_loader, val_loader = _build_dataloaders(config)
    model = _build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training["lr"],
        momentum=config.training["momentum"],
        weight_decay=config.training["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training["step_size"],
        gamma=config.training["gamma"],
    )

    save_every_epochs = int(config.training.get("save_every", 1))
    save_every_iters = int(config.training.get("save_every_iters", 0))
    global_step = 0

    with paths.metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        writer.writeheader()

        for epoch in range(config.training["epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                global_step += 1

                if save_every_iters > 0 and global_step % save_every_iters == 0:
                    ckpt_path = paths.checkpoints_dir / f"iter_{global_step}.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_loss": loss.item(),
                            "train_acc": correct / max(total, 1),
                        },
                        ckpt_path,
                    )

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            val_metrics = _evaluate(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["acc"],
                    "lr": lr,
                }
            )
            f.flush()

            if save_every_epochs > 0 and epoch % save_every_epochs == 0:
                ckpt_path = paths.checkpoints_dir / f"epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["acc"],
                    },
                    ckpt_path,
                )
            scheduler.step()
