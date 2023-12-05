import copy
from ray import cloudpickle
import logging
import math
import numpy as np
from ray import tune
from ray.tune.search.repeater import _TrialGroup
from ray.tune.search import Repeater, Searcher
from typing import Optional, Dict
import ConfigSpace
from hpbandster.optimizers.config_generators.bohb import BOHB
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.bohb.bohb_search import _BOHBJobWrapper
from typing import Dict, List, Optional, Union
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict
from ray.tune.search.sample import (
    Categorical,
    Domain,
    Float,
    Integer,
    LogUniform,
    Normal,
    Quantized,
    Uniform,
)

TRIAL_INDEX = "__trial_index__"
"""str: A constant value representing the repeat index of the trial."""

class BOHBRepeater(Searcher):

    def __init__(
        self,
        space: Optional[Union[Dict, "ConfigSpace.ConfigurationSpace"]] = None,
        bohb_config: Optional[Dict] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        seed: Optional[int] = None,
        max_concurrent: int = 0,
        repeat: int = 5,
        set_index: bool = True
    ):
        assert (
            BOHB is not None
        ), """HpBandSter must be installed!
            You can install HpBandSter with the command:
            `pip install hpbandster ConfigSpace`."""
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
        self.trial_to_params = {}
        self._metric = metric

        self._bohb_config = bohb_config

        if isinstance(space, dict) and space:
            resolved_vars, domain_vars, grid_vars = parse_spec_vars(space)
            if domain_vars or grid_vars:
                logger.warning(
                    UNRESOLVED_SEARCH_SPACE.format(par="space", cls=type(self))
                )
                space = self.convert_search_space(space)

        self._space = space
        self._seed = seed

        self.repeat = repeat
        self._set_index = set_index
        self._groups = []
        self._trial_id_to_group = {}
        self._current_group = None

        self.running = set()
        self.paused = set()

        self._max_concurrent = max_concurrent
        self._points_to_evaluate = points_to_evaluate

        super(BOHBRepeater, self).__init__(
            metric=self._metric,
            mode=mode,
        )

        if self._space:
            self._setup_bohb()

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        self._max_concurrent = max_concurrent
        return True

    def _setup_bohb(self):
        from hpbandster.optimizers.config_generators.bohb import BOHB

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        if self._mode == "max":
            self._metric_op = -1.0
        elif self._mode == "min":
            self._metric_op = 1.0

        if self._seed is not None:
            self._space.seed(self._seed)

        self.running = set()
        self.paused = set()

        bohb_config = self._bohb_config or {}
        self.bohber = BOHB(self._space, **bohb_config)

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], config: Dict, **spec
    ) -> bool:
        if self._space:
            return False
        space = self.convert_search_space(config)
        self._space = space

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        self._setup_bohb()
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if not self._space:
            raise RuntimeError(
                UNDEFINED_SEARCH_SPACE.format(
                    cls=self.__class__.__name__, space="space"
                )
            )

        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(
                    cls=self.__class__.__name__, metric=self._metric, mode=self._mode
                )
            )

        max_concurrent = (
            self._max_concurrent if self._max_concurrent > 0 else float("inf")
        )
        if len(self.running) >= max_concurrent:
            return None

        if self._current_group is None or self._current_group.full():
            config, _ = self.bohber.get_config(None)
            self._current_group = _TrialGroup(trial_id, copy.deepcopy(config), max_trials=self.repeat)
            self._groups.append(self._current_group)
            index_in_group = 0
        else:
            index_in_group = self._current_group.count()
            self._current_group.add(trial_id)

        config = self._current_group.config.copy()
        if self._set_index:
            config[TRIAL_INDEX] = index_in_group
        self._trial_id_to_group[trial_id] = self._current_group

        self.running.add(trial_id)
        return unflatten_list_dict(config)

    def on_trial_result(self, trial_id: str, result: Dict):
        if trial_id not in self.paused:
            self.running.add(trial_id)
        if "hyperband_info" not in result:
            logger.warning(
                "BOHB Info not detected in result. Are you using "
                "HyperBandForBOHB as a scheduler?"
            )
        elif "budget" in result.get("hyperband_info", {}):
            hbs_wrapper = self.to_wrapper(trial_id, result)
            self.bohber.new_result(hbs_wrapper)

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ):
        if trial_id not in self._trial_id_to_group:
            logger.error(
                "Trial {} not in group; cannot report score. "
                "Seen trials: {}".format(trial_id, list(self._trial_id_to_group))
            )
        trial_group = self._trial_id_to_group[trial_id]
        if not result or self._metric not in result:
            score = np.nan
        else:
            score = result[self._metric]
        trial_group.report(trial_id, score)

        if trial_group.finished_reporting():
            scores = np.nanmin(trial_group.scores())
            #self.searcher.on_trial_complete(
            #    trial_group.primary_trial_id,
            #    result={self.searcher.metric: np.nanmin(scores)},
            #    **kwargs
            #)

        #del self.trial_to_params[trial_id]
        self.paused.discard(trial_id)
        self.running.discard(trial_id)

    def to_wrapper(self, trial_id: str, result: Dict) -> _BOHBJobWrapper:
        #print(self._trial_id_to_group[trial_id].config)
        return _BOHBJobWrapper(
            self._metric_op * result[self.metric],
            result["hyperband_info"]["budget"],
            self._trial_id_to_group[trial_id].config,
        )

    # BOHB Specific.
    # TODO(team-ml): Refactor alongside HyperBandForBOHB
    def on_pause(self, trial_id: str):
        self.paused.add(trial_id)
        self.running.remove(trial_id)

    def on_unpause(self, trial_id: str):
        self.paused.remove(trial_id)
        self.running.add(trial_id)

    @staticmethod
    def convert_search_space(spec: Dict) -> "ConfigSpace.ConfigurationSpace":
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                "Grid search parameters cannot be automatically converted "
                "to a TuneBOHB search space."
            )

        # Flatten and resolve again after checking for grid search.
        spec = flatten_dict(spec, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        def resolve_value(
            par: str, domain: Domain
        ) -> ConfigSpace.hyperparameters.Hyperparameter:
            quantize = None

            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                quantize = sampler.q
                sampler = sampler.sampler

            if isinstance(domain, Float):
                if isinstance(sampler, LogUniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    return ConfigSpace.UniformFloatHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=True
                    )
                elif isinstance(sampler, Uniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    return ConfigSpace.UniformFloatHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=False
                    )
                elif isinstance(sampler, Normal):
                    return ConfigSpace.hyperparameters.NormalFloatHyperparameter(
                        par, mu=sampler.mean, sigma=sampler.sd, q=quantize, log=False
                    )

            elif isinstance(domain, Integer):
                if isinstance(sampler, LogUniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    else:
                        # Tune search space integers are exclusive
                        upper -= 1
                    return ConfigSpace.UniformIntegerHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=True
                    )
                elif isinstance(sampler, Uniform):
                    lower = domain.lower
                    upper = domain.upper
                    if quantize:
                        lower = math.ceil(domain.lower / quantize) * quantize
                        upper = math.floor(domain.upper / quantize) * quantize
                    else:
                        # Tune search space integers are exclusive
                        upper -= 1
                    return ConfigSpace.UniformIntegerHyperparameter(
                        par, lower=lower, upper=upper, q=quantize, log=False
                    )

            elif isinstance(domain, Categorical):
                if isinstance(sampler, Uniform):
                    return ConfigSpace.CategoricalHyperparameter(
                        par, choices=domain.categories
                    )

            raise ValueError(
                "TuneBOHB does not support parameters of type "
                "`{}` with samplers of type `{}`".format(
                    type(domain).__name__, type(domain.sampler).__name__
                )
            )

        cs = ConfigSpace.ConfigurationSpace()
        for path, domain in domain_vars:
            par = "/".join(str(p) for p in path)
            value = resolve_value(par, domain)
            cs.add_hyperparameter(value)

        return cs

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as outputFile:
            cloudpickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = cloudpickle.load(inputFile)
        self.__dict__.update(save_object)
