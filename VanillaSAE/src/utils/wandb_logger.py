import wandb
import torch


class WandbLogger:

    def __init__(self, project_name="vanilla-sae", experiment_name=None, config=None):
        """Initialize wandb."""
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config
        )
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to wandb."""
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def finish(self):
        """Finish wandb run."""
        if self.run:
            self.run.finish()