import csv

import wandb
from typing import Dict, List

class BaseLogger:
    """Base class for loggers."""
    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the logger."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def batch_log(self, metrics_dict: Dict[str, List[float]]) -> None:
        """Log a batch of metrics to the logger."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def finish(self) -> None:
        """Finish the logging process."""
        pass

    def log_figure(self, fig, name: str) -> None:
        """Log a figure to the logger."""
        raise NotImplementedError("This method should be overridden by subclasses.")

class CSVLogger(BaseLogger):
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)

    def batch_log(self, metrics_dict: Dict[str, List[float]]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved, where each key corresponds to a metric name
                and each value is a list of floats representing the metric values.
        """
        n_logs = len(next(iter(metrics_dict.values())))

        rows = [
            {key: values[i] for key, values in metrics_dict.items()}
            for i in range(n_logs)
        ]
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            writer.writerows(rows)



class WandBLogger(BaseLogger):
    def __init__(self, cfg: Dict):
        """Initialize the WandB logger."""
        import wandb
        try:
            # Get QDax commit hash
            cmd = "pip freeze | grep qdax"
            output = os.popen(cmd).read()
            # Format qdax @ git+https://github.com/adaptive-intelligent-robotics/QDax.git@dcdc098fee1dad99f264e80f31208ccfd4a06a12
            qdax_commit = output.split("@")[-1].strip()
            cfg["qdax_commit"] = qdax_commit
            # Add link to commit
            cfg["qdax_commit_link"] = f"https://github.com/adaptive-intelligent-robotics/QDax/commit/{qdax_commit}"
        except:
            print("Could not get QDax commit hash. Skipping.")

        self.wandb_run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], config=cfg)

    def log(self, metrics: Dict[str, float]) -> None:
        """Log metrics to WandB."""
        self.wandb_run.log(metrics)

    def batch_log(self, metrics_dict: Dict[str, List[float]]) -> None:
        """Log a batch of metrics to WandB."""
        n_logs = len(next(iter(metrics_dict.values())))
        for i in range(n_logs):
            metrics = {key: values[i] for key, values in metrics_dict.items()}
            self.log(metrics)

    def finish(self):
        """Finish the WandB run."""
        self.wandb_run.finish()


class CombinedLogger(BaseLogger):
    def __init__(self, loggers: List[BaseLogger]):
        """Initialize the combined logger."""
        self.loggers = loggers
        print(f"Using loggers: {[type(logger).__name__ for logger in loggers]}")

    def log(self, metrics: Dict[str, float]) -> None:
        """Log metrics to all loggers."""
        for logger in self.loggers:
            logger.log(metrics)

    def batch_log(self, metrics_list: Dict[str, List[float]]) -> None:
        """Log a batch of metrics to all loggers."""
        # Compute the number of logs
        n_logs = len(next(iter(metrics_list.values())))
        # print(f"Logging {n_logs} metrics to all loggers.")
        if n_logs == 0:
            print("No metrics to log.")
            return
        for logger in self.loggers:
            logger.batch_log(metrics_list)

    def finish(self) -> None:
        """Finish the logging process for all loggers."""
        for logger in self.loggers:
            logger.finish()
        print("Finished logging for all loggers.")