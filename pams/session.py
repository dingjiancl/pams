import random
from typing import Any
from typing import Dict
from typing import Optional

from .logs.base import Logger


class Session:
    """Session management class."""

    def __init__(
        self,
        session_id: int,
        prng: random.Random,
        session_start_time: int,
        simulator: "Simulator",  # type: ignore
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """initialization.

        Args:
            session_id (int): session ID.
            prng (random.Random): pseudo random number generator for this session.
            session_start_time (int): start time of this session.
            simulator (Simulator): this is used for accessing simulation environment.
            name (str): session name.
            logger (Logger, Optional): logger for correcting various outputs in one simulation.
                                       logger is usually shared to all classes.
                                       Please note that logger is usually not thread-safe and non-blocking.

        Returns:
            None

        Note:
             `prng` should not be shared with other classes and be used only in this class.
             It is because sometimes agent process runs one of parallelized threads.
        """
        self.session_id: int = session_id
        self.name: str = name
        self.prng: random.Random = prng
        self.sim: "Simulator" = simulator  # type: ignore
        self.logger: Optional[Logger] = logger

        self.iteration_steps: int = 0
        self.max_high_frequency_orders: int = 1
        self.max_normal_orders: int = 1
        self.with_order_execution: bool = False
        self.with_order_placement: bool = False
        self.with_print: bool = True
        self.high_frequency_submission_rate: float = 1.0
        self.session_start_time: int = session_start_time

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        """setup session configuration from setting format.

        Args:
            settings (Dict[str, Any]): session configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "iterationSteps", "withOrderPlacement", "withOrderExecution", and "withPrint".
                                       This can include the parameter "maxNormalOrders", "maxHighFrequencyOrders", and "hifreqSubmitRate".

        Returns:
            None
        """
        if "iterationSteps" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have iterationSteps"
            )
        self.iteration_steps = int(settings["iterationSteps"])
        if "withOrderPlacement" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withOrderPlacement"
            )
        if not isinstance(settings["withOrderPlacement"], bool):
            raise ValueError("withOrderPlacement must be boolean")
        self.with_order_placement = settings["withOrderPlacement"]
        if "withOrderExecution" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withOrderExecution"
            )
        if not isinstance(settings["withOrderExecution"], bool):
            raise ValueError("withOrderExecution must be boolean")
        self.with_order_execution = settings["withOrderExecution"]
        if "withPrint" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withPrint"
            )
        if not isinstance(settings["withPrint"], bool):
            raise ValueError("withPrint must be boolean")
        self.with_print = settings["withPrint"]
        if "maxNormalOrders" in settings:
            self.max_normal_orders = settings["maxNormalOrders"]
        if "maxHighFrequencyOrders" in settings:
            self.max_high_frequency_orders = settings["maxHighFrequencyOrders"]
        if "hifreqSubmitRate" in settings:
            self.max_high_frequency_orders = settings["hifreqSubmitRate"]
