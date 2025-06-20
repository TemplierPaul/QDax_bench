import hydra
from qdax_bench.main_func import run

if __name__ == "__main__":
    main = hydra.main(version_base=None, config_path="qdax_bench/configs", config_name="config")(run)
    main()
