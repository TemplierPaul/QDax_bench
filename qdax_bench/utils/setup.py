
import qdax_bench.utils.setup_control as setup_control
import qdax_bench.utils.setup_optim as setup_optim
import qdax_bench.utils.setup_bbob as setup_bbob


def setup_pga(config):
    if config["task"]["setup_type"] in [ "brax", "kheperax" ]:
        return setup_control.setup_pga(config)
    else:
        raise NotImplementedError(
            f"Setup type {config['task']['setup_type']} not supported"
        )

def setup_qd(config):
    if config["task"]["setup_type"] in [ "brax", "kheperax" ]:
        return setup_control.setup_qd(config)
    elif config["task"]["setup_type"] == "optim":
        return setup_optim.setup_qd(config)
    elif config["task"]["setup_type"] == "bbob":
        return setup_bbob.setup_qd(config)
    else:
        raise NotImplementedError(
            f"Setup type {config['task']['setup_type']} not supported"
        )
