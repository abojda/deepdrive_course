import pytorch_lightning as pl


class CollectTrainingMetrics(pl.Callback):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics[self.metric_name].item()
        self.metric_history.append(metric)


class CollectValidationMetrics(pl.Callback):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric_history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics[self.metric_name].item()
        self.metric_history.append(metric)
