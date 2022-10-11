#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import flax
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import training.utils as utils


FrozenDict = Any
Model = Any
Tensor = torch.Tensor

@dataclass
class Trainer:
    model: Model
    model_config: FrozenDict
    data_config: FrozenDict
    optimizer_config: FrozenDict
    dir_config: FrozenDict
    steps: int
    rnd_seed: int

    def __post_init__(self):
        # freeze model config dictionary (i.e., make it immutable)
        self.model_config = flax.core.FrozenDict(self.model_config)
        self.loss_fun = nn.CrossEntropyLoss()
        self.logger = SummaryWriter(log_dir=self.dir_config.log_dir)
        self.train_metrics = list()
        self.test_metrics = list()
        self.device = self.model_config["device"]

    def init_optim(self) -> None:
        if self.optimizer_config.name == "adam":
            self.optim = getattr(torch.optim, "Adam")(
                self.model.parameters(), eps=1e-08, lr=self.optimizer_config.lr
            )
        elif self.optimizer_config.name == "adamw":
            self.optim = getattr(torch.optim, "AdamW")(
                self.model.parameters(), eps=1e-08, lr=self.optimizer_config.lr
            )
        elif self.optimizer_config.name == "sgd":
            self.optim = getattr(torch.optim, "SGD")(
                self.model.parameters(), lr=self.optimizer_config.lr, momentum=0.9
            )
        else:
            raise ValueError("\nUse Adam, AdamW or SGD for optimization process.\n")

    @staticmethod
    def collect_hits(
        cls_hits: Dict[int, List[int]], batch_hits: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        for cls, hits in batch_hits.items():
            cls_hits[cls].extend(hits)
        return cls_hits

    def train_epoch(self, batches: Iterator) -> Tuple[float, float]:
        """Step over the full training data in mini-batches of size B and perform SGD."""
        cls_hits = defaultdict(list)
        batch_losses = torch.zeros(len(batches))
        self.model.train()
        for step, batch in tqdm(enumerate(batches), desc="Training", leave=False):
            batch = tuple(t.to(self.device) for t in batch)
            X, y = batch
            self.optim.zero_grad()
            outputs = self.model.forward(X)

            self.loss = 0
            for out in outputs:
                self.loss += self.loss_fun(out, y)
            self.loss /= len(outputs)

            batch_hits = utils.class_hits(outputs[0], y)
            cls_hits = self.collect_hits(
                cls_hits=cls_hits,
                batch_hits=batch_hits,
            )
            self.loss.backward()
            self.optim.step()
            batch_losses[step] += self.loss.item()

        cls_accs = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
        avg_train_acc = torch.mean(torch.tensor(list(cls_accs.values())))
        avg_train_loss = batch_losses.mean()
        return (avg_train_loss, avg_train_acc)

    @torch.no_grad()
    def val_epoch(self, batches: Iterator) -> Tuple[float, float]:
        cls_hits = defaultdict(list)
        batch_losses = torch.zeros(len(batches))
        self.model.eval()
        for step, batch in tqdm(enumerate(batches), desc="Validation", leave=False):
            batch = tuple(t.to(self.device) for t in batch)
            X, y = batch
            out = self.model.forward(X, train=False)
            loss = self.loss_fun(out, y)
            batch_hits = utils.class_hits(out, y)
            cls_hits = self.collect_hits(
                cls_hits=cls_hits,
                batch_hits=batch_hits,
            )
            batch_losses[step] += loss.item()
        cls_accs = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
        avg_val_acc = torch.mean(torch.tensor(list(cls_accs.values())))
        avg_val_loss = batch_losses.mean()
        return (avg_val_loss, avg_val_acc)

    @torch.no_grad()
    def inference(self, X_test: Tensor, y_test: Tensor) -> Tuple[float, Dict[int, List[int]]]:
        cls_hits = defaultdict(list)
        self.model.eval()
        X = X_test.to(self.device)
        y = y_test.to(self.device)
        logits = self.model.forward(X, train=False)
        loss = self.loss_fun(logits, y)
        batch_hits = utils.class_hits(logits, y)
        cls_hits = self.collect_hits(cls_hits=cls_hits, batch_hits=batch_hits)
        return loss, cls_hits

    @torch.no_grad()
    def batch_inference(self, X_test: Tensor, y_test: Tensor) -> Tuple[float, Dict[int, List[int]]]:
        losses = []
        cls_hits = defaultdict(list)
        X = X_test.to(self.device)
        y = y_test.to(self.device)
        for i in range(math.ceil(X_test.shape[0] / self.data_config.batch_size)):
            X_i = X[i * self.data_config.batch_size : (i + 1) * self.data_config.batch_size]
            y_i = y[i * self.data_config.batch_size : (i + 1) * self.data_config.batch_size]
            logits = self.model.forward(X_i, train=False)
            loss = self.loss_fun(logits, y_i)
            batch_hits = utils.class_hits(logits, y_i)
            cls_hits = self.collect_hits(cls_hits=cls_hits, batch_hits=batch_hits)
            losses.append(loss)
        loss = torch.mean(losses)
        return loss, cls_hits

    def train(self, train_batches: Iterator, val_batches: Iterator) -> Tuple[dict, int]:
        self.init_optim()
        for epoch in tqdm(range(1, self.optimizer_config.epochs + 1), desc="Epoch"):
            train_performance = self.train_epoch(train_batches)
            val_performance = self.val_epoch(val_batches)
            self.train_metrics.append(train_performance)
            self.test_metrics.append(val_performance)
            train_loss, train_acc = train_performance
            val_loss, val_acc = val_performance

            # Tensorboard logging
            self.logger.add_scalar("train/loss", train_loss, global_step=epoch)
            self.logger.add_scalar("train/acc", train_acc, global_step=epoch)
            self.logger.add_scalar("val/loss", val_loss, global_step=epoch)
            self.logger.add_scalar("val/acc", val_acc, global_step=epoch)
            print(
                f"Epoch: {epoch:04d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n"
            )
            self.logger.flush()

            if epoch % self.steps == 0:
                self.save_checkpoint(epoch=epoch)

        metrics = self.merge_metrics(self.train_metrics, self.test_metrics)
        return metrics, epoch

    @staticmethod
    def merge_metrics(train_metrics, test_metrics) -> FrozenDict:
        return flax.core.FrozenDict(
            {"train_metrics": train_metrics, "test_metrics": test_metrics}
        )

    def save_checkpoint(self, epoch: int) -> None:
        # PyTorch convention is to save checkpoints as .tar files
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": copy.deepcopy(self.state_dict()),
            "optim_state_dict": copy.deepcopy(self.optim.state_dict()),
            "loss": self.loss,
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, f"model_epoch{epoch+1:04d}.tar"),
        )

    def load_checkpoint_(self) -> None:
        """Load model and optimizer params from previous checkpoint, if available."""
        if os.path.exists(self.checkpoint_dir):
            models = sorted(
                [
                    m.name
                    for m in os.scandir(self.checkpoint_dir)
                    if m.name.endswith("tar")
                ]
            )
            if len(models) > 0:
                try:
                    PATH = os.path.join(self.checkpoint_dir, models[-1])
                    checkpoint = torch.load(PATH, map_location=self.device)
                    self.load_state_dict(checkpoint["model_state_dict"])
                    self.optim.load_state_dict(checkpoint["optim_state_dict"])
                    self.start = checkpoint["epoch"]
                    self.loss = checkpoint["loss"]
                    self.train_accs = checkpoint["train_accs"]
                    self.val_accs = checkpoint["val_accs"]
                    self.train_losses = checkpoint["train_losses"]
                    self.val_losses = checkpoint["val_losses"]
                    print(
                        f"...Loaded model and optimizer params from previous run. Resuming training at epoch {self.start}.\n"
                    )
                except RuntimeError:
                    print(
                        "...Loading model and optimizer params failed. Check whether you are currently using a different set of model parameters."
                    )
                    print("...Starting model training from scratch.\n")
                    self.start = 0
                    self.train_accs, self.val_accs = [], []
                    self.train_losses, self.val_losses = [], []
            else:
                self.start = 0
                self.train_accs, self.val_accs = [], []
                self.train_losses, self.val_losses = [], []
        else:
            os.makedirs(self.checkpoint_dir)
            self.start = 0
            self.train_accs, self.val_accs = [], []
            self.train_losses, self.val_losses = [], []

    def __len__(self) -> int:
        return self.optimizer_config.epochs
