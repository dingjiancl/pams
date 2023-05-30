import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..index_market import IndexMarket
from ..logs import Logger
from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from .high_frequency_agent import HighFrequencyAgent


# 套利代理人：在现货市场和其指数市场之间进行套利交易的代理人
class ArbitrageAgent(HighFrequencyAgent):
    """Arbitrage Agent class.

    This class inherits from the HighFrequencyAgent class.
    Arbitrage agent mainly aimed to take arbitrage chance between spot markets and its index market.

    Note:
        Currently, index markets must have the same weight for each constitutional stock.
    """

    def __init__(
        self,
        agent_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id, prng=prng, simulator=simulator, name=name, logger=logger
        )

        self.order_volume: int = 1
        self.order_threshold_price: float = 1.0
        self.order_time_length: int = 1

    # 对代理人进行初始化设置
    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],  # 反映agent的设置
        accessible_markets_ids: List[int],  # 可访问的市场
        *args,
        **kwargs,
    ) -> None:
        """agent setup. Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "orderVolume", "orderThresholdPrice".
                                       This can include the parameter "orderTimeLength".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        # 初始化代理人在可访问市场的资产数量
        super(ArbitrageAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )
        # 从输入参数中获取订单数量、阈值价格和订单时间长度
        # 将这些参数保存在代理人对象中以备使用
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for ArbitrageAgent")
        if not isinstance(settings["orderVolume"], int):
            raise ValueError("orderVolume have to be int")
        self.order_volume = settings["orderVolume"]
        if "orderThresholdPrice" not in settings:
            raise ValueError("orderThresholdPrice is required for ArbitrageAgent")
        self.order_threshold_price = settings["orderThresholdPrice"]
        if "orderTimeLength" in settings:
            if not isinstance(settings["orderTimeLength"], int):
                raise ValueError("orderTimeLength have to be int")
            self.order_time_length = settings["orderTimeLength"]

    # 提交订单并进行套利交易
    def _submit_orders(self, market: Market) -> List[Union[Order, Cancel]]:
        """internal sub routine for submitting orders by market.

        Args:
            market (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = []
        # 检查是否为指数市场and是否可以访问
        if not isinstance(market, IndexMarket):
            return orders
        if not self.is_market_accessible(market_id=market.market_id):
            return orders

        # 获取IndexMarket对象和所有的SpotMarket对象
        index: IndexMarket = market
        spots: List[Market] = index.get_components()  # 返回_components列表中的所有市场
        if not index.is_running or not index.is_all_markets_running():
            return orders
        market_index: float = index.get_index()  # 获取当前指数市场的指数价格
        market_price: float = index.get_market_price()  # 获取当前指数市场的实时价格

        # 检查所有组成SpotMarket的组件是否具有相同的outstanding_shares
        if len(set(map(lambda x: x.outstanding_shares, spots))) > 1:
            raise NotImplementedError(
                "currently, the components must have the same outstanding shares"
            )

        # ----价格差>阈值时，进行套利交易----

        # 如果IndexMarket的价格低于其指数
        if (
            market_price < market_index
            and market_index - market_price > self.order_threshold_price
        ):
            # 买入IndexMarket
            index_order_volume = len(spots) * self.order_volume
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=index.market_id,
                    is_buy=True,
                    kind=LIMIT_ORDER,
                    volume=index_order_volume,
                    price=index.get_market_price(),
                    ttl=self.order_time_length,
                )
            )
            # 以市场价格买入所有的SpotMarket
            for m in spots:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=m.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=self.order_volume,
                        price=m.get_market_price(),
                        ttl=self.order_time_length,
                    )
                )
        # 如果IndexMarket的价格高于其指数
        if (
            market_price > market_index
            and market_price - market_index > self.order_threshold_price
        ):
            # 卖出IndexMarket
            index_order_volume = len(spots) * self.order_volume
            orders.append(
                Order(
                    agent_id=self.agent_id,
                    market_id=index.market_id,
                    is_buy=False,
                    kind=LIMIT_ORDER,
                    volume=index_order_volume,
                    price=index.get_market_price(),
                    ttl=self.order_time_length,
                )
            )
            # 以市场价格卖出所有的SpotMarket
            for m in spots:
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=m.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=self.order_volume,
                        price=m.get_market_price(),
                        ttl=self.order_time_length,
                    )
                )
        # ----价格差>阈值时，进行套利交易</>----
        # 返回订单列表（之前已将新建的订单添加到订单列表中）
        return orders

    # 在所有市场中进行套利交易（调用_submit_orders()）
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders to take arbitrage chance.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            orders.extend(self._submit_orders(market=market))
        return orders

    """
    list.extend():
    将一个列表的元素添加到另一个列表中
    具体的，将参数列表中的每个元素添加到原始列表的末尾
    """

    def __repr__(self) -> str:
        """string representation of FCN agent class.

        Returns:
            str: string representation of this class.
        """
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.agent_id}, rnd={self.prng}, "
            f"order_volume={self.order_volume}, order_threshold_price={self.order_threshold_price}, "
            f"order_time_length={self.order_time_length}>"
        )