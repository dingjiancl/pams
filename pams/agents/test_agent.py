from typing import List
from typing import Union

from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from .base import Agent


# 用于测试market的agent：随机生成订单，没有任何策略
class TestAgent(Agent):
    """TestAgent class

    This agent is mainly used for generating orders with zero intelligence and just aims for market tests.
    Different from Agent class, this agent is implementing its `submit_orders` methods.

    Note:
        Please do not use this agent for your research.
    """

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders. This method is usually called by simulator or runners.

        Args:
             markets (list[Market]): market objects. Usually it is set by simulator or runners automatically.

        Returns:
            list[Order, Cancel]: orders including cancel orders.

        Note:
            TestAgent is mainly introduced for testing.
            Therefore, we do not intend that you modify it.
            If you want to use similar agents for your research purpose, please make user-defined class instead of using TestAgent.
        """
        margin_scale: float = 10.0
        volume_scale: int = 100
        time_length_scale: int = 100
        buy_chance: float = 0.4
        sell_chance: float = 0.4

        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            # 如果该代理可以访问该市场
            if self.is_market_accessible(market_id=market.market_id):
                # 生成一个随机价格
                price: float = market.get_market_price() + (
                    self.prng.random() * 2 * margin_scale - margin_scale
                )
                # 生成随机的订单数量
                volume: int = self.prng.randint(1, volume_scale)
                # 订单生命周期的范围
                time_length: int = self.prng.randint(1, time_length_scale)
                p = self.prng.random()  # 随机生成一个浮点数p
                # 生成的买卖订单，并添加到orders列表中
                if p < buy_chance + sell_chance:
                    orders.append(
                        Order(
                            agent_id=self.agent_id,
                            market_id=market.market_id,
                            is_buy=p < buy_chance,
                            kind=LIMIT_ORDER,
                            volume=volume,
                            price=price,
                            ttl=time_length,
                        )
                    )
        # 返回orders列表
        return orders
