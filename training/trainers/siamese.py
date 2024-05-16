# ---INFO-----------------------------------------------------------------------
"""
Utilities to train models in siamese configuration.
"""

__all__ = [
    "ConstrastiveSimilarityTrainer",
]


# ---DEPEDENCIES----------------------------------------------------------------
import typing as tp
import torch as pt
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
)

from ..losses import ContrastiveSimilarityLoss


# ---SRC------------------------------------------------------------------------
class ConstrastiveSimilarityTrainer:
    def __init__(
        self,
        eeg_encoder: pt.nn.Module,
        img_encoder: tp.Any,
        optimizer: pt.optim.Optimizer = None,
        loss_fn: tp.Any = None,
        train_img_encoder: bool = False,
        device: str = "cuda",
        log_dir: str = None,
    ):
        self.eeg_encoder = eeg_encoder
        self.img_encoder = img_encoder

        self.eeg_encoder.to(device)
        if hasattr(img_encoder, "to"):
            img_encoder.to(device)

        params_e = list(eeg_encoder.parameters())
        params_i = list(img_encoder.parameters()) if train_img_encoder else []
        self.optimizer = optimizer or pt.optim.Adam(
            params_e + params_i, lr=1e-3, weight_decay=1e-5
        )
        self.loss_fn = loss_fn or ContrastiveSimilarityLoss()

        self.trn_engine = Engine(self._trn_step)
        self.val_engine = Engine(self._val_step)
        self.device = device
        self.log_dir = log_dir or "."
        self.logger = TensorboardLogger(log_dir=self.log_dir)

    def _trn_step(self, engine, batch) -> float:
        self.eeg_encoder.train()
        if hasattr(self.img_encoder, "train"):
            self.img_encoder.train()
        self.optimizer.zero_grad()

        eeg, pos_img, neg_img = batch
        for t in [eeg, pos_img, neg_img]:
            t.to(self.device)
        eeg_emb = self.eeg_encoder(eeg)
        pos_img_emb = self.img_encoder(pos_img)
        neg_img_emb = self.img_encoder(neg_img)

        loss = self.loss_fn(eeg_emb, pos_img_emb, neg_img_emb).mean()
        loss.backward()
        self.optimizer.step()
        metrics = {"loss": loss.item()}

        return metrics

    def _val_step(self, engine, batch) -> float:
        self.eeg_encoder.eval()
        if hasattr(self.img_encoder, "eval"):
            self.img_encoder.eval()

        with pt.inference_mode():
            eeg, pos_img, neg_img = batch
            for t in [eeg, pos_img, neg_img]:
                t.to(self.device)
            eeg_emb = self.eeg_encoder(eeg)
            pos_img_emb = self.img_encoder(pos_img)
            neg_img_emb = self.img_encoder(neg_img)

            loss = self.loss_fn(eeg_emb, pos_img_emb, neg_img_emb).mean()
        metrics = {"loss": loss.item()}

        return metrics

    def run(
        self,
        label: str,
        trn_loader: tp.Any,
        val_loader: tp.Any,
        max_epochs: int = 100,
        ckpt_eeg_encoder: bool = True,
        ckpt_img_encoder: bool = True,
    ):
        trn_pbar = ProgressBar(persist=False)
        val_pbar = ProgressBar(persist=False)
        for p, e in [(trn_pbar, self.trn_engine), (val_pbar, self.val_engine)]:
            p.attach(
                e,
                metric_names="all",
                output_transform=lambda x: x,
                event_name=Events.ITERATION_COMPLETED,
            )

        for flag, prefix, model in [
            (ckpt_eeg_encoder, "eeg_encoder", self.eeg_encoder),
            (ckpt_img_encoder, "img_encoder", self.img_encoder),
        ]:
            if flag:
                ckpt_handler = ModelCheckpoint(
                    self.log_dir,
                    prefix,
                    score_function=lambda engine: -engine.state.metrics["loss"],
                    score_name="loss",
                    n_saved=1,
                    create_dir=True,
                    save_as_state_dict=True,
                    require_empty=False,
                )
                self.trn_engine.add_event_handler(
                    Events.EPOCH_COMPLETED(every=5),
                    ckpt_handler,
                    {label: model},
                )

        for tag, engine in [
            (f"{label}/trn", self.trn_engine),
            (f"{label}/val", self.val_engine),
        ]:
            tb_handler = OutputHandler(
                tag=tag,
                metric_names="all",
                output_transform=lambda x: x,
            )
            engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                tb_handler,
            )

        self.trn_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda _: self.val_engine.run(val_loader, max_epochs=1),
        )

        self.trn_engine.run(trn_loader, max_epochs=max_epochs)

        trn_pbar.close()
        val_pbar.close()
        self.trn_engine.remove_event_handler(trn_pbar)
        self.val_engine.remove_event_handler(val_pbar)
        self.trn_engine.remove_event_handler(ckpt_handler)
        self.trn_engine.remove_event_handler(tb_handler)
        self.val_engine.remove_event_handler(tb_handler)
        self.logger.close()
