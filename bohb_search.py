"""BOHB (Bayesian Optimization with HyperBand)"""

import copy
import logging
import math

# use cloudpickle instead of pickle to make BOHB obj
# pickleable
from ray import cloudpickle
from typing import Dict, List, Optional, Union

from ray.tune.result import DEFAULT_METRIC
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
from ray.tune.search import (
    UNRESOLVED_SEARCH_SPACE,
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
    Searcher,
)
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict

try:
    import ConfigSpace
    from hpbandster.optimizers.config_generators.bohb import BOHB
except ImportError:
    BOHB = ConfigSpace = None

logger = logging.getLogger(__name__)


class _BOHBJobWrapper:
    """Mock object for HpBandSter to process."""

    def __init__(self, loss: float, budget: float, config: Dict):
        self.result = {"loss": loss}
        self.kwargs = {"budget": budget, "config": config.copy()}
        self.exception = None
