import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .contrastive_loss import ContrastiveLoss
from .triplet_loss import TripletLoss

Tensor = torch.Tensor


class GlocalProbe(pl.LightningModule):
    def __init__(
        self,
        features: Tensor,
        optim_cfg: Dict[str, Any],
        model_cfg: Dict[str, str],
        extractor: Any,
    ):
        super().__init__()
        self.features = torch.nn.Parameter(
            torch.from_numpy(features).to(torch.float),
            requires_grad=False,
        )
        self.feature_dim = self.features.shape[1]
        self.optim = optim_cfg["optim"]
        self.lr = optim_cfg["lr"]  # learning rate
        self.scale = optim_cfg["sigma"]  # width of the Gaussian at initialization time
        self.reg = optim_cfg["reg"]  # type of regularization
        self.lmbda = optim_cfg["lmbda"]  # strength of regularization
        self.temp = optim_cfg[
            "tau"
        ]  # temperature parameter for contrastive learning objective
        self.alpha = optim_cfg[
            "alpha"
        ]  # contribution of contrastive loss to overall loss
        self.use_bias = optim_cfg[
            "use_bias"
        ]  # whether or not to use a bias for the probe
        self.max_epochs = optim_cfg["max_epochs"]
        self.module = model_cfg["module"]
        self.out_path = optim_cfg["out_path"]
        self.things_mean = optim_cfg["things_mean"]
        self.things_std = optim_cfg["things_std"]

        self.global_loss_fun = TripletLoss(temperature=1.0)
        self.local_loss_fun = ContrastiveLoss(temperature=self.temp)
        self.teacher_extractor = extractor

        initialization = self.get_initialization()

        if self.use_bias:
            self.transform_w = torch.nn.Parameter(
                data=initialization[0],
                requires_grad=True,
            )
            self.transform_b = torch.nn.Parameter(
                data=initialization[1],
                requires_grad=True,
            )
        else:
            self.transform_w = torch.nn.Parameter(
                data=initialization,
                requires_grad=True,
            )

    def get_initialization(self) -> Tensor:
        """Initialize the transformation matrix."""
        # initialize the transformation matrix with values drawn from a tight Gaussian with very small width
        weights = torch.eye(self.feature_dim) * self.scale
        if self.use_bias:
            bias = torch.ones(self.feature_dim) * self.scale
            return weights, bias
        return weights

    def normalize_features(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Map ImageNet features onto the unit-sphere."""
        normalized_teacher_features = F.normalize(features, dim=1)
        student_features = features @ self.transform_w
        if self.use_bias:
            student_features += self.transform_b
        normalized_student_features = F.normalize(student_features, dim=1)
        return normalized_teacher_features, normalized_student_features

    def forward(self, things_objects: Tensor, imagenet_features: Tensor) -> Tensor:
        things_embedding = self.features @ self.transform_w
        if self.use_bias:
            things_embedding += self.transform_b
        batch_embeddings = things_objects @ things_embedding
        (
            normalized_teacher_imagenet_features,
            normalized_student_imagenet_features,
        ) = self.normalize_features(imagenet_features.to(torch.float))
        teacher_similarities = (
            normalized_teacher_imagenet_features
            @ normalized_teacher_imagenet_features.T
        )
        student_similarities = (
            normalized_student_imagenet_features
            @ normalized_student_imagenet_features.T
        )
        return batch_embeddings, teacher_similarities, student_similarities

    def global_prediction(self, things_objects: Tensor) -> Tensor:
        things_embedding = self.features @ self.transform_w
        if self.use_bias:
            things_embedding += self.transform_b
        batch_embeddings = things_objects @ things_embedding
        return batch_embeddings

    @staticmethod
    def convert_predictions(sim_predictions: Tensor) -> Tensor:
        """Convert similarity predictions into odd-one-out predictions."""
        first_conversion = torch.where(
            sim_predictions != 1, sim_predictions - 2, sim_predictions
        )
        ooo_predictions = torch.where(first_conversion < 0, 2, first_conversion)
        return ooo_predictions

    @staticmethod
    def compute_similarities(
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the similarity function (modeled as a dot product) to each pair in the triplet."""
        sim_i = torch.sum(anchor * positive, dim=1)
        sim_j = torch.sum(anchor * negative, dim=1)
        sim_k = torch.sum(positive * negative, dim=1)
        return (sim_i, sim_j, sim_k)

    @staticmethod
    def break_ties(probas: Tensor) -> Tensor:
        return torch.tensor(
            [
                -1
                if (
                    torch.unique(pmf).shape[0] != pmf.shape[0]
                    or torch.unique(pmf.round(decimals=2)).shape[0] == 1
                )
                else torch.argmax(pmf)
                for pmf in probas
            ]
        )

    def accuracy_(self, probas: Tensor, batching: bool = True) -> Tensor:
        choices = self.break_ties(probas)
        argmax = np.where(choices == 0, 1, 0)
        acc = argmax.mean() if batching else argmax.tolist()
        return acc

    def choice_accuracy(self, similarities: float) -> float:
        probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
        choice_acc = self.accuracy_(probas)
        return choice_acc

    @staticmethod
    def unbind(embeddings: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.unbind(
            torch.reshape(embeddings, (-1, 3, embeddings.shape[-1])), dim=1
        )

    @staticmethod
    def normalize(triplet: List[Tensor]) -> List[Tensor]:
        """Normalize object embeddings to have unit norm."""
        return list(map(lambda obj: F.normalize(obj, dim=1), triplet))

    def l2_regularization(self, alpha: float = 1.0) -> Tensor:
        """Apply combination of l2 and l1 regularization during training."""
        # NOTE: Frobenius norm in PyTorch is equivalent to torch.linalg.vector_norm(self.transform, ord=2, dim=(0, 1)))
        l2_reg = alpha * torch.linalg.norm(self.transform_w, ord="fro")
        l1_reg = (1 - alpha) * torch.linalg.vector_norm(
            self.transform_w, ord=1, dim=(0, 1)
        )
        complexity_loss = l2_reg + l1_reg
        return complexity_loss

    def eye_regularization(self) -> Tensor:
        """Regularization towards the identity matrix."""
        complexity_loss = torch.sum(
            (
                self.transform_w
                - torch.eye(self.feature_dim).to(self.transform_w.device)
                * torch.mean(torch.diag(self.transform_w))
            )
            ** 2
        )
        return complexity_loss

    def training_step(
        self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
    ) -> Tensor:
        things_objects, (imagenet_images, _) = batch
        imagenet_features = self.teacher_extractor.extract_features(
            batches=imagenet_images.unsqueeze(0),
            module_name=self.module,
            flatten_acts=True,
            output_type="tensor",
        )
        batch_embeddings, teacher_similarities, student_similarities = self(
            things_objects, imagenet_features
        )
        anchor, positive, negative = self.unbind(batch_embeddings)
        dots = self.compute_similarities(anchor, positive, negative)
        global_loss = self.global_loss_fun(dots)
        # apply l1 and l2 regularization during training to prevent overfitting to train objects
        if self.reg == "l2":
            # regularization towards 0
            complexity_loss = self.l2_regularization()
        else:  # regularization towards the identity
            complexity_loss = self.eye_regularization()
        locality_loss = self.local_loss_fun(teacher_similarities, student_similarities)
        loss = (
            (1 - self.alpha) * global_loss
            + self.alpha * locality_loss
            + self.lmbda * complexity_loss
        )
        acc = self.choice_accuracy(dots)
        self.log("triplet_acc", acc, on_epoch=True)
        self.log("triplet_loss", global_loss, on_epoch=True)
        self.log("local_loss", locality_loss, on_epoch=True)
        self.log("complexity_loss", complexity_loss, on_epoch=True)
        return loss

    def _save_transform_snapshot(self) -> None:
        if self.use_bias:
            with open(os.path.join(self.out_path, "transform_tmp.npz"), "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=self.transform_w.data.detach().cpu().numpy(),
                    bias=self.transform_b.data.detach().cpu().numpy(),
                    mean=self.things_mean,
                    std=self.things_std,
                )
        else:
            with open(os.path.join(self.out_path, "transform_tmp.npz"), "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=self.transform_w.data.detach().cpu().numpy(),
                    mean=self.things_mean,
                    std=self.things_std,
                )

    def validation_step(
        self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
    ) -> Dict[str, float]:
        global_loss, locality_loss, loss, acc = self._shared_eval_step(batch, batch_idx)
        # save snapshot of the transformation
        self._save_transform_snapshot()
        metrics = {
            "val_acc": acc,
            "val_overall_loss": loss,
            "val_triplet_loss": global_loss,
            "val_contrastive_loss": locality_loss,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(
        self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
    ) -> Dict[str, float]:
        global_loss, locality_loss, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "test_acc": acc,
            "test_overall_loss": loss,
            "test_triplet_loss": global_loss,
            "test_contrastive_loss": locality_loss,
        }
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(
        self, batch: Tuple[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
    ) -> Tuple[float, float, float, float]:
        things_objects, (imagenet_images, _) = batch
        imagenet_features = self.teacher_extractor.extract_features(
            batches=imagenet_images.unsqueeze(0),
            module_name=self.module,
            flatten_acts=True,
            output_type="tensor",
        )
        batch_embeddings, teacher_similarities, student_similarities = self(
            things_objects, imagenet_features
        )
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        global_loss = self.global_loss_fun(similarities)
        locality_loss = self.local_loss_fun(teacher_similarities, student_similarities)
        loss = global_loss + locality_loss
        acc = self.choice_accuracy(similarities)
        return global_loss, locality_loss, loss, acc

    def predict_step(self, batch: Tensor, batch_idx: int):
        things_objects = batch
        batch_embeddings = self.global_prediction(things_objects)
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        sim_predictions = torch.argmax(
            F.softmax(torch.stack(similarities, dim=1), dim=1), dim=1
        )
        ooo_predictions = self.convert_predictions(sim_predictions)
        return ooo_predictions

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        if self.optim.lower() == "adam":
            optimizer = getattr(torch.optim, self.optim.capitalize())
            optimizer = optimizer(self.parameters(), lr=self.lr)
        elif self.optim.lower() == "sgd":
            optimizer = getattr(torch.optim, self.optim.upper())
            optimizer = optimizer(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(
                "\nUse Adam or SGD for learning a linear transformation of a network's feature space.\n"
            )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, last_epochs=-1, verbose=True)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=self.max_epochs, last_epoch=-1, verbose=True
        )
        return [optimizer], [scheduler]


class GlocalFeatureProbe(pl.LightningModule):
    def __init__(
        self,
        features: Tensor,
        optim_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.features = torch.nn.Parameter(
            torch.from_numpy(features).to(torch.float),
            requires_grad=False,
        )
        self.feature_dim = self.features.shape[1]
        self.optim = optim_cfg["optim"]
        self.lr = optim_cfg["lr"]  # learning rate
        self.scale = optim_cfg["sigma"]  # width of the Gaussian at initialization time
        self.reg = optim_cfg["reg"]  # type of regularization
        self.lmbda = optim_cfg["lmbda"]  # strength of regularization
        self.temp = optim_cfg[
            "tau"
        ]  # temperature parameter for contrastive learning objective
        self.alpha = optim_cfg[
            "alpha"
        ]  # contribution of contrastive loss to overall loss
        self.use_bias = optim_cfg[
            "use_bias"
        ]  # whether or not to use a bias for the probe
        self.max_epochs = optim_cfg["max_epochs"]
        self.out_path = optim_cfg["out_path"]
        self.things_mean = optim_cfg["things_mean"]
        self.things_std = optim_cfg["things_std"]

        self.global_loss_fun = TripletLoss(temperature=1.0)
        self.local_loss_fun = ContrastiveLoss(temperature=self.temp)

        initialization = self.get_initialization()

        if self.use_bias:
            self.transform_w = torch.nn.Parameter(
                data=initialization[0],
                requires_grad=True,
            )
            self.transform_b = torch.nn.Parameter(
                data=initialization[1],
                requires_grad=True,
            )
        else:
            self.transform_w = torch.nn.Parameter(
                data=initialization,
                requires_grad=True,
            )

    def get_initialization(self) -> Tensor:
        """Initialize the transformation matrix."""
        # initialize the transformation matrix with values drawn from a tight Gaussian with very small width
        weights = torch.eye(self.feature_dim) * self.scale
        if self.use_bias:
            bias = torch.ones(self.feature_dim) * self.scale
            return weights, bias
        return weights

    def normalize_features(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """Map ImageNet features onto the unit-sphere."""
        normalized_teacher_features = F.normalize(features, dim=1)
        student_features = features @ self.transform_w
        if self.use_bias:
            student_features += self.transform_b
        normalized_student_features = F.normalize(student_features, dim=1)
        return normalized_teacher_features, normalized_student_features

    def forward(self, things_objects: Tensor, imagenet_features: Tensor) -> Tensor:
        things_embedding = self.features @ self.transform_w
        if self.use_bias:
            things_embedding += self.transform_b
        batch_embeddings = things_objects @ things_embedding
        (
            normalized_teacher_imagenet_features,
            normalized_student_imagenet_features,
        ) = self.normalize_features(imagenet_features)
        teacher_similarities = (
            normalized_teacher_imagenet_features
            @ normalized_teacher_imagenet_features.T
        )
        student_similarities = (
            normalized_student_imagenet_features
            @ normalized_student_imagenet_features.T
        )
        return batch_embeddings, teacher_similarities, student_similarities

    def global_prediction(self, things_objects: Tensor) -> Tensor:
        things_embedding = self.features @ self.transform_w
        if self.use_bias:
            things_embedding += self.transform_b
        batch_embeddings = things_objects @ things_embedding
        return batch_embeddings

    @staticmethod
    def convert_predictions(sim_predictions: Tensor) -> Tensor:
        """Convert similarity predictions into odd-one-out predictions."""
        first_conversion = torch.where(
            sim_predictions != 1, sim_predictions - 2, sim_predictions
        )
        ooo_predictions = torch.where(first_conversion < 0, 2, first_conversion)
        return ooo_predictions

    @staticmethod
    def compute_similarities(
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the similarity function (modeled as a dot product) to each pair in the triplet."""
        sim_i = torch.sum(anchor * positive, dim=1)
        sim_j = torch.sum(anchor * negative, dim=1)
        sim_k = torch.sum(positive * negative, dim=1)
        return (sim_i, sim_j, sim_k)

    @staticmethod
    def break_ties(probas: Tensor) -> Tensor:
        return torch.tensor(
            [
                -1
                if (
                    torch.unique(pmf).shape[0] != pmf.shape[0]
                    or torch.unique(pmf.round(decimals=2)).shape[0] == 1
                )
                else torch.argmax(pmf)
                for pmf in probas
            ]
        )

    def accuracy_(self, probas: Tensor, batching: bool = True) -> Tensor:
        choices = self.break_ties(probas)
        argmax = np.where(choices == 0, 1, 0)
        acc = argmax.mean() if batching else argmax.tolist()
        return acc

    def choice_accuracy(self, similarities: float) -> float:
        probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
        choice_acc = self.accuracy_(probas)
        return choice_acc

    @staticmethod
    def unbind(embeddings: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.unbind(
            torch.reshape(embeddings, (-1, 3, embeddings.shape[-1])), dim=1
        )

    @staticmethod
    def normalize(triplet: List[Tensor]) -> List[Tensor]:
        """Normalize object embeddings to have unit norm."""
        return list(map(lambda obj: F.normalize(obj, dim=1), triplet))

    def l2_regularization(self, alpha: float = 1.0) -> Tensor:
        """Apply combination of l2 and l1 regularization during training."""
        # NOTE: Frobenius norm in PyTorch is equivalent to torch.linalg.vector_norm(self.transform, ord=2, dim=(0, 1)))
        l2_reg = alpha * torch.linalg.norm(self.transform_w, ord="fro")
        l1_reg = (1 - alpha) * torch.linalg.vector_norm(
            self.transform_w, ord=1, dim=(0, 1)
        )
        complexity_loss = l2_reg + l1_reg
        return complexity_loss

    def eye_regularization(self) -> Tensor:
        """Regularization towards the identity matrix."""
        complexity_loss = torch.sum(
            (
                self.transform_w
                - torch.eye(self.feature_dim).to(self.transform_w.device)
                * torch.mean(torch.diag(self.transform_w))
            )
            ** 2
        )
        return complexity_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        things_objects, imagenet_features = batch
        batch_embeddings, teacher_similarities, student_similarities = self(
            things_objects, imagenet_features
        )
        anchor, positive, negative = self.unbind(batch_embeddings)
        dots = self.compute_similarities(anchor, positive, negative)
        global_loss = self.global_loss_fun(dots)
        # apply l1 and l2 regularization during training to prevent overfitting to train objects
        if self.reg == "l2":
            # regularization towards 0
            complexity_loss = self.l2_regularization()
        else:  # regularization towards the identity
            complexity_loss = self.eye_regularization()
        locality_loss = self.local_loss_fun(teacher_similarities, student_similarities)
        loss = (
            (1 - self.alpha) * global_loss
            + self.alpha * locality_loss
            + self.lmbda * complexity_loss
        )
        acc = self.choice_accuracy(dots)
        self.log("triplet_acc", acc, on_epoch=True)
        self.log("triplet_loss", global_loss, on_epoch=True)
        self.log("local_loss", locality_loss, on_epoch=True)
        self.log("complexity_loss", complexity_loss, on_epoch=True)
        return loss

    def _save_transform_snapshot(self) -> None:
        if self.use_bias:
            with open(os.path.join(self.out_path, "transform_tmp.npz"), "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=self.transform_w.data.detach().cpu().numpy(),
                    bias=self.transform_b.data.detach().cpu().numpy(),
                    mean=self.things_mean,
                    std=self.things_std,
                )
        else:
            with open(os.path.join(self.out_path, "transform_tmp.npz"), "wb") as f:
                np.savez_compressed(
                    file=f,
                    weights=self.transform_w.data.detach().cpu().numpy(),
                    mean=self.things_mean,
                    std=self.things_std,
                )

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, float]:
        global_loss, locality_loss, loss, acc = self._shared_eval_step(batch, batch_idx)
        # save snapshot of the transformation
        self._save_transform_snapshot()
        metrics = {
            "val_acc": acc,
            "val_overall_loss": loss,
            "val_triplet_loss": global_loss,
            "val_contrastive_loss": locality_loss,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, float]:
        global_loss, locality_loss, loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "test_acc": acc,
            "test_overall_loss": loss,
            "test_triplet_loss": global_loss,
            "test_contrastive_loss": locality_loss,
        }
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tuple[float, float, float, float]:
        things_objects, imagenet_features = batch
        batch_embeddings, teacher_similarities, student_similarities = self(
            things_objects, imagenet_features
        )
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        global_loss = self.global_loss_fun(similarities)
        locality_loss = self.local_loss_fun(teacher_similarities, student_similarities)
        loss = global_loss + locality_loss
        acc = self.choice_accuracy(similarities)
        return global_loss, locality_loss, loss, acc

    def predict_step(self, things_objects: Tensor, batch_idx: int):
        batch_embeddings = self.global_prediction(things_objects)
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        sim_predictions = torch.argmax(
            F.softmax(torch.stack(similarities, dim=1), dim=1), dim=1
        )
        ooo_predictions = self.convert_predictions(sim_predictions)
        return ooo_predictions

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        if self.optim.lower() == "adam":
            optimizer = getattr(torch.optim, self.optim.capitalize())
            optimizer = optimizer(self.parameters(), lr=self.lr)
        elif self.optim.lower() == "sgd":
            optimizer = getattr(torch.optim, self.optim.upper())
            optimizer = optimizer(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(
                "\nUse Adam or SGD for learning a linear transformation of a network's feature space.\n"
            )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, last_epochs=-1, verbose=True)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=self.max_epochs, last_epoch=-1, verbose=True
        )
        return [optimizer], [scheduler]
