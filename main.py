import hydra
from omegaconf import DictConfig, OmegaConf

from qdax_bench.main_func import run as bench_run

def main_func(cfg: DictConfig) -> None:
    bench_run(cfg)

if __name__ == "__main__":
    main = hydra.main(version_base=None, config_path="qdax_bench/configs", config_name="config")(main_func)
    main()
