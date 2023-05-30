import random
import warnings
from typing import Any
from typing import Dict
from typing import Optional

from .logs.base import Logger


# 一个模拟的会话（一次模拟实验的期间）
class Session:
    """Session management class."""

    # 构造函数，初始化Session实例的属性
    def __init__(
        self,
        session_id: int,  # 会话ID
        prng: random.Random,  # 随机数生成器
        session_start_time: int,  # 会话开始的时间戳
        simulator: "Simulator",  # type: ignore # 该会话所在的Simulator实例
        name: str,  # 会话名称，用于识别一个会话
        logger: Optional[Logger] = None,  # 该会话的记录器实例
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
        self.simulator: "Simulator" = simulator  # type: ignore  # NOQA
        self.logger: Optional[Logger] = logger

        self.iteration_steps: int = 0
        self.max_high_frequency_orders: int = 1
        self.max_normal_orders: int = 1
        self.with_order_execution: bool = False
        self.with_order_placement: bool = False
        self.with_print: bool = True
        self.high_frequency_submission_rate: float = 1.0
        self.session_start_time: int = session_start_time

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.session_id}, name={self.name}, "
            f"iteration_steps={self.iteration_steps}, session_start_time={self.session_start_time}, "
            f"max_normal_orders={self.max_normal_orders}, max_high_frequency_orders={self.max_high_frequency_orders}, "
            f"with_order_placement={self.with_order_placement}, with_order_execution={self.with_order_execution}, "
            f"high_frequency_submission_rate={self.high_frequency_submission_rate}, with_print={self.with_print}, "
            f"logger={self.logger.__str__()}>"
        )

    # 从输入的settings参数中读取一些参数，设置属性值
    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """setup session configuration from setting format.

        Args:
            settings (Dict[str, Any]): session configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "iterationSteps", "withOrderPlacement", "withOrderExecution", and "withPrint".
                                       This can include the parameter "maxNormalOrders", "maxHighFrequencyOrders", and "highFrequencySubmitRate".

        Returns:
            None
        """
        if "iterationSteps" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have iterationSteps"
            )
        if not isinstance(settings["iterationSteps"], int):
            raise ValueError("iterationSteps must be int")
        self.iteration_steps = int(settings["iterationSteps"])  # 每次迭代Simulator时，会话的步数（即会话持续的时间）
        if "withOrderPlacement" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withOrderPlacement"
            )
        if not isinstance(settings["withOrderPlacement"], bool):
            raise ValueError("withOrderPlacement must be boolean")
        self.with_order_placement = settings["withOrderPlacement"]  # 是否允许订单提交到市场
        if "withOrderExecution" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withOrderExecution"
            )
        if not isinstance(settings["withOrderExecution"], bool):
            raise ValueError("withOrderExecution must be boolean")
        self.with_order_execution = settings["withOrderExecution"]  # 是否执行订单撮合
        if "withPrint" not in settings:
            raise ValueError(
                "for each element in simulation.sessions must have withPrint"
            )
        if not isinstance(settings["withPrint"], bool):
            raise ValueError("withPrint must be boolean")
        self.with_print = settings["withPrint"]  # 是否输出会话信息
        if "maxNormalOrders" in settings:
            # TODO: check non-negative
            self.max_normal_orders = settings["maxNormalOrders"]  # 允许提交到市场的最大普通订单数量
        if "maxHighFrequencyOrders" in settings:
            # TODO: check non-negative
            self.max_high_frequency_orders = settings["maxHighFrequencyOrders"]  # 允许提交到市场的最大高频订单数量
            if "maxHifreqOrders" in settings:
                raise ValueError(
                    "maxHifreqOrders is replaced to maxHighFrequencyOrders in pams. Please delete it."
                )
        elif "maxHifreqOrders" in settings:
            # TODO: check non-negative
            warnings.warn(
                "maxHifreqOrders is replaced to maxHighFrequencyOrders in pams."
            )
            self.max_high_frequency_orders = settings["maxHifreqOrders"]  # 高频订单提交率
        # TODO: check malOrders + maxHighFrequencyOrders >= 1
        if "highFrequencySubmitRate" in settings:
            # TODO: check non-negative
            self.high_frequency_submission_rate = settings["highFrequencySubmitRate"]
            if "hifreqSubmitRate" in settings:
                raise ValueError(
                    "hifreqSubmitRate is replaced to highFrequencySubmitRate in pams. Please delete it."
                )
        elif "hifreqSubmitRate" in settings:
            warnings.warn(
                "hifreqSubmitRate is replaced to highFrequencySubmitRate in pams."
            )
            # TODO: check non-negative
            self.max_high_frequency_orders = settings["hifreqSubmitRate"]
