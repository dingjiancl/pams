import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..logs.base import CancelLog
from ..logs.base import ExecutionLog
from ..logs.base import Logger
from ..logs.base import OrderLog
from ..market import Market
from ..order import Cancel
from ..order import Order
from ..utils.json_random import JsonRandom


# 名为Agent的抽象基类，代表了市场中的交易代理
class Agent(ABC):
    """Agent class (abstract class)

    Once you define the agent class inheriting this agent ABC class, simulator automatically generate agents using the class you define.
    Therefore, you don't need to call __init__ or any other method and simulator automatically call them if it is necessary.

    `submit_orders(self, markets: List[Market])` is required to be implemented.

    .. seealso::
        - :class:`pams.agents.FCNAgent`: FCNAgent
    """

    # 初始化函数
    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """agent initialization. Usually call from simulator automatically.

        Args:
            agent_id (int): agent id. This is also required to generate orders to identify who submit orders.
            prng (random.Random): pseudo random number generator for this agent.
            simulator (Simulator): this is used for accessing simulation environment.
            name (str): this is unique name for this agent.
                        Automatically generated by simulator and just set it as the name of this agent.
            logger (Logger, Optional): logger for correcting various outputs in one simulation.
                                       logger is usually shared to all classes.
                                       Please note that logger is usually not thread-safe and non-blocking.

        Returns:
            None

        Note:
             `prng` should not be shared with other classes and be used only in this class.
             It is because sometimes agent process runs one of parallelized threads.
        """
        self.agent_id: int = agent_id
        self.name: str = name
        self.asset_volumes: Dict[int, int] = {}
        self.cash_amount: float = 0
        self.prng: random.Random = prng
        self.simulator: "Simulator" = simulator  # type: ignore  # NOQA
        self.logger: Optional[Logger] = logger

    # 代理类的字符串表示形式
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.agent_id}, name={self.name}, "
            f"logger={self.logger}>"
        )

    # 初始化代理人在可访问市场的资产数量
    def setup(self, settings: Dict[str, Any], accessible_markets_ids: List[int], *args, **kwargs) -> None:  # type: ignore
        """agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration.  Usually, automatically set from json config of simulator.
                                       This must include the parameters "cashAmount" and "assetVolume".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        if "cashAmount" not in settings:
            raise ValueError("cashAmount is required property of agent settings")
        self.cash_amount = JsonRandom(prng=self.prng).random(
            json_value=settings["cashAmount"]
        )
        if "assetVolume" not in settings:
            raise ValueError("cashAmount is required property of agent settings")

        # 在代理人可访问的市场上，为该代理人设置可用资产数量
        for market_id in accessible_markets_ids:
            self.set_market_accessible(market_id=market_id)  # 将市场标记为可访问的状态（将代理人在指定市场上的资产数量设置为0）
            # 根据JSON中的"assetVolume"生成随机的资产数量
            volume = int(
                JsonRandom(prng=self.prng).random(json_value=settings["assetVolume"])
            )
            # 将生成的资产数量分配给代理人
            self.set_asset_volume(market_id=market_id, volume=volume)

    # 获取代理人在指定市场上持有的资产数量
    def get_asset_volume(self, market_id: int) -> int:
        """getter of the asset volume held by the agent.

        Args:
            market_id (int): market ID.

        Returns:
            int: asset volume for the specified market ID.
        """
        if not self.is_market_accessible(market_id=market_id):
            raise ValueError(f"market {market_id} is not accessible")
        return self.asset_volumes[market_id]

    # 获取代理人的现金数量
    def get_cash_amount(self) -> float:
        """getter of the cash amount held by the agent.

        Returns:
            float: cash amount held by this agent.
        """
        return self.cash_amount

    # 获取代理人的伪随机数生成器
    def get_prng(self) -> random.Random:
        """getter of the pseudo random number generator.

        Returns:
            random.Random: pseudo random number generator for this agent.
        """
        return self.prng

    # 检查代理人在指定市场上是否可访问
    def is_market_accessible(self, market_id: int) -> bool:
        """determine if the market ID is included in the asset volume

        Args:
            market_id (int): market ID.

        Returns:
            bool: whether the asset volume held by the agent contains the specified market or not.
        """
        return market_id in self.asset_volumes

    # 代理向市场提交订单成功时回调
    def submitted_order(self, log: OrderLog) -> None:
        """call back when an order submission is accepted by a market.

        Args:
            log (OrderLog): log for order submission

        Returns:
            None
        """
        pass

    # 代理人的订单被市场执行时回调
    def executed_order(self, log: ExecutionLog) -> None:
        """call back when a submitted order is executed in a market.

        Args:
            log (ExecutionLog): log for order execution

        Returns:
            None
        """
        pass

    # 代理人的取消订单被市场接受时回调
    def canceled_order(self, log: CancelLog) -> None:
        """call back when cancel order is accepted by a market.

        Args:
            log (CancelLog): log for order cancellation

        Returns:
            None
        """
        pass

    # 设置代理人在指定市场上持有的资产数量
    def set_asset_volume(self, market_id: int, volume: int) -> None:
        """setter of the asset volume held by agent.

        Args:
            market_id (int): market ID.
            volume (int): volume to be set for the market

        Returns:
            None
        """
        if not self.is_market_accessible(market_id=market_id):
            raise ValueError(f"market {market_id} is not accessible")
        if not isinstance(volume, int):
            raise ValueError("volume have to be int")
        self.asset_volumes[market_id] = volume

    # 设置代理人的现金数量
    def set_cash_amount(self, cash_amount: float) -> None:
        """setter of the cash amount held by agent.

        Args:
            cash_amount (float): cash amount held by this agent.

        Returns:
            None
        """
        self.cash_amount = cash_amount

    # 将代理人在指定市场上的资产数量设置为0
    def set_market_accessible(self, market_id: int) -> None:
        """set the specified market volume to 0.

        Args:
            market_id (int): market ID.

        Returns:
            None
        """
        if self.is_market_accessible(market_id=market_id):
            raise ValueError(f"market {market_id} is already accessible")
        self.asset_volumes[market_id] = 0

    # 代理人提交订单的抽象方法
    @abstractmethod
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders (abstract method). This method automatically called from runners.

        This method is called only when this agent has a chance to submit orders.
        Therefore, it is not guaranteed that this method is called at all the step of simulation.

        Args:
            markets (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.

        Note:
            You should implement this method if you inherit this agent.
        """
        pass

    # 增加或减少代理人在指定市场上持有的资产数量
    def update_asset_volume(self, market_id: int, delta: int) -> None:
        """increasing or decreasing the asset volume.

        Args:
            market_id (int): market ID.
            delta (int): amount of change in the asset volume.

        Returns:
            None
        """
        if not self.is_market_accessible(market_id=market_id):
            raise ValueError(f"market {market_id} is not accessible")
        if not isinstance(delta, int):
            raise ValueError("delta have to be int")
        self.asset_volumes[market_id] += delta

    # 增加或减少代理的现金数量
    def update_cash_amount(self, delta: float) -> None:
        """increasing or decreasing the cash amount.

        Args:
            delta (float): amount of change in the cash amount.

        Returns:
            None
        """
        self.cash_amount += delta