from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..index_market import IndexMarket
from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from .high_frequency_agent import HighFrequencyAgent


class ArbitrageAgent(HighFrequencyAgent):
    """Arbitrage Agent class.

    This class inherits from the HighFrequencyAgent class.

    Note:
        Please also see :class:`pams.agents.ArbitrageAgent`
    """

    order_volume: int = 1
    order_threshold_price: float = 1.0
    order_time_length: int = 1

    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs
    ) -> None:
        """agent setup.

        Args:
            settings (Dict[str, Any]): agent configuration.
                                       This must include the parameters "orderVolume", "orderThresholdPrice".
                                       This can include the parameter "orderTimeLength".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        super(ArbitrageAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )
        if "orderVolume" not in settings:
            raise ValueError("orderVolume is required for ArbitrageAgent")
        self.order_volume = settings["orderVolume"]
        if "orderThresholdPrice" not in settings:
            raise ValueError("orderThresholdPrice is required for ArbitrageAgent")
        self.order_threshold_price = settings["orderThresholdPrice"]
        if "orderTimeLength" in settings:
            self.order_time_length = settings["orderTimeLength"]

    def _submit_orders(self, market: Market) -> List[Union[Order, Cancel]]:
        """submit orders by market.

        Args:
            market (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = []
        if not isinstance(market, IndexMarket):
            return orders
        if not self.is_market_accessible(market_id=market.market_id):
            return orders
        index: IndexMarket = market
        spots: List[Market] = index.get_components()
        if not index.is_running or not index.is_all_markets_running():
            return orders
        market_index: float = index.get_index()
        market_price: float = index.get_market_price()

        if len(set(map(lambda x: x.outstanding_shares, spots))) > 1:
            raise AssertionError(
                "currently, the components must have the same outstanding shares"
            )

        if (
            market_price < market_index
            and market_index - market_price > self.order_threshold_price
        ):
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
        if (
            market_price > market_index
            and market_price - market_index > self.order_threshold_price
        ):
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
        return orders

    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders.

        Args:
            markets (List[Market]): markets to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        orders: List[Union[Order, Cancel]] = []
        for market in markets:
            orders.extend(self._submit_orders(market=market))
        return orders
