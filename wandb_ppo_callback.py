import wandb
from stable_baselines3.common.callbacks import BaseCallback


class SimpleBrittleStarCallback(BaseCallback):
    """
    Simple callback for logging brittle star environment metrics to Weights & Biases.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Log info when episode ends
        if self.locals.get("dones")[0]:
            # Get info from environment
            info = self.locals.get("infos")[0]

            # Log only the essential metrics
            wandb.log({
                "steps_taken": info.get("steps_taken", 0),
                "best_distance": info.get("best_distance", 0),
                "reward": info.get("reward", 0),
                "best_improvement": info.get("best_improvement", 0),
                "is_terminated": int(info.get("is_terminated", False)),
                "is_truncated": int(info.get("is_truncated", False))
            })

        return True