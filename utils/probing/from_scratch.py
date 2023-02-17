from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .contrastive_loss import ContrastiveLoss
from .triplet_loss import TripletLoss

Tensor = torch.Tensor


class FromScratch(pl.LightningModule):
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
        self.reg = optim_cfg["reg"]  # type of regularization
        self.lmbda = optim_cfg["lmbda"]  # strength of regularization
        self.alpha = optim_cfg[
            "alpha"
        ]  # contribution of supervized loss to overall loss
        self.batch_size = optim_cfg[
            "batch_size"
        ]
        self.max_epochs = optim_cfg[
            "max_epochs"
        ]
        self.module = model_cfg["module"]
        self.model_name = model_cfg["model"]
        extractor = get_extractor(
                model_name=self.model_name,
                source="pytorch",
                device="cuda",
                pretrained=False,
                model_parameters=None,
        )
        self.model = extractor.model

        self.similarity_loss_fun = TripletLoss(temperature=1.0)
        self.classification_loss_fun = torch.nn.CrossEntropyLoss()

    def forward(self, things_batch: Tensor, imagenet_batch: Tensor) -> Tensor:
        # TODO: use a hook to also enable the use of the penultimate layer
        things_ebmbeddings = self.model(things_batch)
        imagenet_logits = self.model(imagenet_batch)
        return things_ebmbeddings, imagenet_logits

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

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        things_batch, imagenet_batch = batch
        images, labels = imagenet_batch

        # TODO: we want to change the "extract_features" method
        # such that we get back a torch.Tensor rather than a np.ndarray,
        # because the line below is unncessary computational overhead
        imagenet_features = torch.from_numpy(imagenet_features)
        batch_embeddings, teacher_similarities, student_similarities = self(
            things_batch, imagenet_features
        )
        anchor, positive, negative = self.unbind(batch_embeddings)
        dots = self.compute_similarities(anchor, positive, negative)
        c_entropy = self.similarity_loss_fun(dots)
        classification_loss = self.classification_loss_fun(imagenet_logits, labels)
        loss = c_entropy + self.alpha * locality_loss + self.lmbda * complexity_loss
        acc = self.choice_accuracy(dots)
        self.log("train_loss", c_entropy, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, things_batch: Tensor, batch_idx: int):
        loss, acc = self._shared_eval_step(things_batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, things_batch: Tensor, batch_idx: int):
        loss, acc = self._shared_eval_step(things_batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, things_batch: Tensor, batch_idx: int):
        batch_embeddings = self.global_prediction(things_batch)
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        loss = self.similarity_loss_fun(similarities)
        acc = self.choice_accuracy(similarities)
        return loss, acc

    def predict_step(self, things_batch: Tensor, batch_idx: int):
        batch_embeddings = self(things_batch)
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
        return optimizer
