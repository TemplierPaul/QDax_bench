from typing import Any, List, Optional, Union

from brax.v1.envs import Env, _envs
from brax.v1.envs.wrappers import (
    AutoResetWrapper,
    EpisodeWrapper,
    EvalWrapper,
    VectorWrapper,
)

from qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.environments.init_state_wrapper import FixedInitialStateWrapper
from qdax.environments.wrappers import CompletedEvalWrapper

from qdax.environments import _qdax_envs, _qdax_custom_envs

def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    legacy_spring=True,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in _envs.keys():
        env = _envs[env_name](legacy_spring=legacy_spring, **kwargs)
    elif env_name in _qdax_envs.keys():
        env = _qdax_envs[env_name](**kwargs)
    elif env_name in _qdax_custom_envs.keys():
        base_env_name = _qdax_custom_envs[env_name]["env"]
        if base_env_name in _envs.keys():
            env = _envs[base_env_name](legacy_spring=legacy_spring, **kwargs)
        elif base_env_name in _qdax_envs.keys():
            env = _qdax_envs[base_env_name](**kwargs)  # type: ignore
    else:
        raise NotImplementedError("This environment name does not exist!")

    if env_name in _qdax_custom_envs.keys():
        # roll with qdax wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore

    if episode_length is not None:
        env = EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = VectorWrapper(env, batch_size)
    if fixed_init_state:
        # retrieve the base env
        if env_name not in _qdax_custom_envs.keys():
            base_env_name = env_name
        # wrap the env
        env = FixedInitialStateWrapper(env, base_env_name=base_env_name)  # type: ignore
    if auto_reset:
        env = AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if eval_metrics:
        env = EvalWrapper(env)
        env = CompletedEvalWrapper(env)

    return env
