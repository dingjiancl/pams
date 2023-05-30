import json
import os
import random
import time
from abc import ABC
from abc import abstractmethod
from io import TextIOBase
from io import TextIOWrapper
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO
from typing import Type
from typing import Union

from ..agents import Agent
from ..agents import HighFrequencyAgent
from ..logs.base import Logger
from ..simulator import Simulator


class Runner(ABC):  # Runner类是ABC（抽象基类（Abstract Base Class））类的子类
    """Runner of the market simulator class (Abstract class).

    .. seealso::
        - :class:`pams.runners.SequentialRunner`
    """

    # Runner类的初始化方法
    def __init__(
        self,
        settings: Union[Dict, TextIOBase, TextIO, TextIOWrapper, os.PathLike, str],  # 当前Runner实例的设置
        prng: Optional[random.Random] = None,  # 当前Runner实例使用的随机数生成器
        logger: Optional[Logger] = None,  # 当前Runner实例使用的日志器
        simulator_class: Type[Simulator] = Simulator,  # 当前Runner实例使用的模拟器对象
    ):
        """initialize.

        Args:
            settings (Union[Dict, TextIOBase, TextIO, TextIOWrapper, os.PathLike, str]): runner configuration.
                You can set python dictionary, a file pointer, or a file path.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None
        """
        # ----给self.settings（字典类型）赋值----
        self.settings: Dict
        if isinstance(settings, Dict):  # 传入的settings为字典类型时
            self.settings = settings
        elif (
            isinstance(settings, TextIOBase)
            or isinstance(settings, TextIO)
            or isinstance(settings, TextIOWrapper)  # 传入的settings为文件指针类型时
        ):
            self.settings = json.load(fp=settings)
        else:
            self.settings = json.load(fp=open(settings, mode="r"))  # 传入的settings为文件路径时
        # ----给self.settings（字典类型）赋值</>----

        # 创建当前Runner实例使用的随机数生成器对象
        self._prng: random.Random = prng if prng is not None else random.Random()
        # 冒号：类型注释（Type Annotation），若不符合则类型检查工具会给出警告，但程序不会报错（并不影响程序运行）

        # 当前Runner实例使用的日志器
        self.logger = logger

        # 当前Runner实例使用的模拟器对象
        # 传入：随机数生成器；种子：self._prng生成的一个随机整数
        self.simulator: Simulator = simulator_class(
            prng=random.Random(self._prng.randint(0, 2**31))
        )
        # 创建空列表（Type：可存储任意类型的类对象）
        self.registered_classes: List[Type] = []

    # Runner类的主方法
    # 程序初始化与Simulator的运行，并输出程序初始化时间和执行时间
    def main(self) -> None:
        """main process. The process is executed while measuring time."""
        setup_start_time_ns = time.time_ns()
        self._setup()  # 设置
        start_time_ns = time.time_ns()
        self._run()  # 运行
        end_time_ns = time.time_ns()
        print(
            "# INITIALIZATION TIME " + str((start_time_ns - setup_start_time_ns) / 1e9)
        )
        print("# EXECUTION TIME " + str((end_time_ns - start_time_ns) / 1e9))

    def class_register(self, cls: Type) -> None:
        """register class. This method is used for user-defined classes.

        Usually, user-defined classes, i.e., the classes you implemented for your original simulation, cannot be referred from
        pams package, especially from simulation runners. Therefore, the class registration to the runner is necessary.

        Args:
            cls (Type): class to register.

        Returns:
            None
        """
        self.registered_classes.append(cls)

    @abstractmethod
    def _setup(self) -> None:
        """internal usage class for setup. This method should be implemented in descendants."""
        pass

    @abstractmethod
    def _run(self) -> None:
        """internal usage class for simulation running. This method should be implemented in descendants.

        Usually the process in this methods should be control simulation flow and parallelization.
        """
        pass

    @staticmethod
    def judge_hft_or_not(agent: Agent) -> bool:
        """determine if the agent is type of the :class:`pams.agents.HighFrequencyAgent`.

        Args:
            agent (Agent): agent instance.

        Returns:
            bool: whether the agent class is the :class:`pams.agents.HighFrequencyAgent` or not.
        """
        return isinstance(agent, HighFrequencyAgent)
