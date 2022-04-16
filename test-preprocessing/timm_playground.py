import argparse
from pathlib import Path
from pyexpat import model
from statistics import mode
from black import out
from fastNLP import CrossEntropyLoss, EarlyStopCallback
from pyrsistent import b

import numpy as np
import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torch
import torchmetrics
from timm.scheduler import CosineLRScheduler

from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS
from timm.data.parsers.parser import Parser
from pathlib import Path

from torchmetrics import MetricCollection, Accuracy, Precision, Recall

from pytorch_accelerated.callbacks import TrainerCallback, EarlyStoppingCallback

from experiment_log import ExperimentLog

def create_datasets(image_size, data_mean, data_std, train_path, val_path, log):
    auto_augment = "rand-m7-mstd0.5-inc1"
    log.auto_augment = auto_augment

    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment= auto_augment,
    )

    eval_transforms = timm.data.create_transform(
        input_size=image_size, mean=data_mean, std=data_std
    )

    train_dataset = timm.data.dataset.ImageDataset(
        train_path, transform=train_transforms
    )
    eval_dataset = timm.data.dataset.ImageDataset(val_path, transform=eval_transforms)

    return train_dataset, eval_dataset

best_since = 0

class SaveBestModelCallback(TrainerCallback):
    def __init__(
        self,
        log,
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
    ):
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.save_path = log.model_name+".pt"
        self.reset_on_train = reset_on_train
        self.log = log

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)

        if self.best_metric is None:
            self.best_metric = current_metric
            trainer.save_checkpoint(
                save_path=self.save_path,
                checkpoint_kwargs={self.watch_metric: self.best_metric},
            )

            self.log.accuracy = str(trainer.run_history.get_latest_metric("accuracy"))
            self.log.precision = str(trainer.run_history.get_latest_metric("precision"))
            self.log.recall = str(trainer.run_history.get_latest_metric("recall"))
            
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self.best_metric = current_metric
                trainer.save_checkpoint(
                    save_path=self.save_path,
                    checkpoint_kwargs={"loss": self.best_metric},
                )

                self.log.accuracy = str(trainer.run_history.get_latest_metric("accuracy"))
                self.log.precision = str(trainer.run_history.get_latest_metric("precision"))
                self.log.recall = str(trainer.run_history.get_latest_metric("recall"))


    def on_training_run_end(self, trainer, **kwargs):
        trainer.print(
            f"Loading checkpoint with {self.watch_metric}: {self.best_metric}"
        )
        trainer.load_checkpoint(self.save_path)

    
class TimmMixupTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_args, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = timm.data.Mixup(**mixup_args)

        self.accuracy = torchmetrics.Accuracy(num_classes=num_classes, average="macro")
        self.precision = torchmetrics.Precision(num_classes=num_classes, average="macro")
        self.recall = torchmetrics.Recall(num_classes=num_classes, average="macro")

        self.ema_accuracy = torchmetrics.Accuracy(num_classes=num_classes)
        self.ema_model = None

    def create_scheduler(self):
        return timm.scheduler.CosineLRScheduler(
            self.optimizer,
            t_initial=self.run_config.num_epochs,
            cycle_decay=0.5,
            lr_min=1e-6,
            t_in_epochs=True,
            warmup_t=3,
            warmup_lr_init=1e-4,
            cycle_limit=1,
        )

    def training_run_start(self):
        # Model EMA requires the model without a DDP wrapper and before sync batchnorm conversion
        self.ema_model = timm.utils.ModelEmaV2(
            self._accelerator.unwrap_model(self.model), decay=0.9
        )
        if self.run_config.is_distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        # mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        # return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

        model_outputs = self.model(xb)
        train_loss = self.loss_func(model_outputs, yb)

        return {
            "loss": train_loss,
            "model_outputs": model_outputs,
            "batch_size": yb.size(0),
        }

    def train_epoch_end(
        self,
    ):
        self.ema_model.update(self.model)
        self.ema_model.eval()

        if hasattr(self.optimizer, "sync_lookahead"):
            self.optimizer.sync_lookahead()

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            self.accuracy.update(outputs.argmax(-1), yb)
            self.precision.update(outputs.argmax(-1), yb)
            self.recall.update(outputs.argmax(-1), yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()

        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric("precision", self.precision.compute().cpu())
        self.run_history.update_metric("recall", self.recall.compute().cpu())

        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.ema_accuracy.reset()


def main(data_path):
    log = ExperimentLog()
    # Set training arguments, hardcoded here for clarity
    image_size = (224, 224)
    lr = 5e-3
    smoothing = 0.1
    mixup = 0.3
    cutmix = 1.0
    batch_size = 32
    bce_target_thresh = 0.2
    num_epochs = 50
    early_stop = 5

    log.image_size = str(image_size)
    log.learning_rate = str(lr)
    log.smoothing = str(smoothing)
    log.mixup = str(mixup)
    log.cutmix = str(cutmix)
    log.batch_size = str(batch_size)
    log.bce_target_thresh = str(bce_target_thresh)
    log.epoch = str(num_epochs)
    log.early_stop = str(early_stop)

    data_path = Path(data_path)
    train_path = data_path / "train"
    val_path = data_path / "valid"
    num_classes = len(list(train_path.iterdir()))

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )

    model_name = "efficientnet_b0"
    drop_path_rate = 0.05
    log.model_name = model_name
    log.drop_path_rate = str(drop_path_rate)
    # Create model using timm
    model = timm.create_model(
        model_name=model_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path_rate
    )

    # Load data config associated with the model to use in data augmentation pipeline
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = (0.4124, 0.3856, 0.3493)
    data_std = (0.2798, 0.2703, 0.2726)
    log.data_mean = str(data_mean)
    log.data_std = str(data_std)
    # Create training and validation datasets
    train_dataset, eval_dataset = create_datasets(
        train_path=train_path,
        val_path=val_path,
        image_size=image_size,
        data_mean=data_mean,
        data_std=data_std,
        log= log
    )

    optimizer_name = "lookahead_AdamW"
    weight_decay = 0.01
    log.optimizer = optimizer_name
    log.weight_decay = str(weight_decay)
    # Create optimizer
    optimizer = timm.optim.create_optimizer_v2(
        model, opt= optimizer_name, lr=lr, weight_decay=weight_decay
    )

    # As we are using Mixup, we can use BCE during training and CE for evaluation
    # train_loss_fn = timm.loss.BinaryCrossEntropy(target_threshold=bce_target_thresh, smoothing=smoothing)
    
    # train_loss_fn = timm.loss.LabelSmoothingCrossEntropy(smoothing=smoothing)
    train_loss_fn = torch.nn.CrossEntropyLoss()
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    log.loss_name = str(train_loss_fn)

    # Create trainer and start training
    trainer = TimmMixupTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=train_loss_fn,
        eval_loss_fn=validate_loss_fn,
        mixup_args=mixup_args,
        num_classes=num_classes,
        callbacks=[
            *DEFAULT_CALLBACKS,
            SaveBestModelCallback(
                watch_metric="accuracy", 
                greater_is_better=True, 
                log =log),
            EarlyStoppingCallback(
                early_stopping_patience=early_stop, 
                early_stopping_threshold=0.001,
                watch_metric="accuracy",
                greater_is_better=True)
            ],
    )
    
    trainer.train(
        per_device_batch_size=batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        create_scheduler_fn=trainer.create_scheduler,
    )

    log.write_experiment_details(log.model_name+"-"+str(log.accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    main(args.data_dir)