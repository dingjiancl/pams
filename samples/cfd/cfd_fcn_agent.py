import math
import random
from pams.agents import Agent, HighFrequencyAgent
from typing import Any, Dict, List
from pams.market import Market
from pams.order import MARKET_ORDER, LIMIT_ORDER
from cfd_market import CFDOrder, CFDMarket

class CFDMarketFCNAgent(Agent):
    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        super(CFDMarketFCNAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )
        if "leverage_rate" not in settings:
            raise ValueError("leverage_rate is required for CFDSimpleAgent")
        self.leverage_rate = settings["leverage_rate"]
        self.positions: List[Dict[str, Any]] = []
        # Ex: [{"is_buy": True, "price":300.0, "volume": 1, "leverage_rate":0.1}, ...]

    def submit_orders_by_market(self, market: Market) -> List[CFDOrder]:
        orders: List[CFDOrder] = []

        if not self.is_market_accessible(market_id=market.market_id):
            return orders

        if not isinstance(market, CFDMarket):
            return orders
        
        cfd_market = market
        underlying_market = cfd_market.underlying_market
        
        # Decide whether to close the position based on price of the position
        for pos_dic in self.positions:
            is_buy = pos_dic["is_buy"]
            target_order_id = pos_dic["order_id"]
            price = pos_dic["price"]
            volume = pos_dic["volume"]
            leverage_rate = pos_dic["leverage_rate"]
            
            # price of the position > expected future price: investors liquidate their long positions
            if is_buy:
                if price > self.get_expected_future_price(underlying_market):
                    orders.append(
                    CFDOrder(
                        agent_id=self.agent_id,
                        market_id=cfd_market.market_id,
                        is_buy=bool(1-is_buy),
                        kind=MARKET_ORDER,
                        volume=volume,
                        price=None,
                        ttl=self.time_window_size,
                        is_to_close_position=True,
                        target_order_id=target_order_id,
                        target_order_price=price,
                        leverage_rate=leverage_rate
                    )
                )
            # price of the position < expected future price: investors liquidate their short positions
            else:
                if price < self.get_expected_future_price(underlying_market):
                    orders.append(
                    CFDOrder(
                        agent_id=self.agent_id,
                        market_id=cfd_market.market_id,
                        is_buy=bool(1-is_buy),
                        kind=MARKET_ORDER,
                        volume=volume,
                        price=None,
                        ttl=self.time_window_size,
                        is_to_close_position=True,
                        target_order_id=target_order_id,
                        target_order_price=price,
                        leverage_rate=leverage_rate
                    )
                )
        
        # Decide whether to open a position based on market price

        # market price < expected future price: investors open a long position
        market_price = underlying_market.get_market_price()
        if market_price < self.get_expected_future_price(underlying_market):
            orders.append(
            CFDOrder(
                agent_id=self.agent_id,
                market_id=cfd_market.market_id,
                is_buy=True,
                kind=MARKET_ORDER,
                volume=volume,
                price=None,
                ttl=self.time_window_size,
                is_to_close_position=False,
                target_order_id=target_order_id,
                target_order_price=price,
                leverage_rate=leverage_rate
            )
        )
            
        # market price > expected future price: investors open a short position
        if market_price > self.get_expected_future_price(underlying_market):
            orders.append(
            CFDOrder(
                agent_id=self.agent_id,
                market_id=cfd_market.market_id,
                is_buy=False,
                kind=MARKET_ORDER,
                volume=volume,
                price=None,
                ttl=self.time_window_size,
                is_to_close_position=False,
                target_order_id=target_order_id,
                target_order_price=price,
                leverage_rate=leverage_rate
            )
        )

        return orders
    
    def submit_orders(self, markets: List[Market]) -> List[CFDOrder]:
        """submit orders based on FCN-based calculation.
        """
        orders: List[List[CFDOrder]] = sum(
            [self.submit_orders_by_market(market=market) for market in markets], []
        )
        return orders
    
    def get_expected_future_price(self, market:Market) -> float:
        """calculate expected future price.
                """
        time: int = market.get_time()
        time_window_size: int = min(time, self.time_window_size)
        assert time_window_size >= 0
        assert self.fundamental_weight >= 0.0
        assert self.chart_weight >= 0.0
        assert self.noise_weight >= 0.0

        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return = fundamental_scale * math.log(
            market.get_fundamental_price() / market.get_market_price()
        )
        assert self.is_finite(fundamental_log_return)  # To determine if it is a valid (non-NaN and finite) floating-point number

        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_mean_log_return = chart_scale * math.log(
            market.get_market_price() / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_mean_log_return)

        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)

        expected_log_return: float = (  # Expected logarithmic return
                                             1.0 / (self.fundamental_weight + self.chart_weight + self.noise_weight)
                                     ) * (
                                             self.fundamental_weight * fundamental_log_return
                                             + self.chart_weight
                                             * chart_mean_log_return
                                             * (1 if self.is_chart_following else -1)
                                             + self.noise_weight * noise_log_return
                                     )
        assert self.is_finite(expected_log_return)

        # Expected future price
        expected_future_price: float = market.get_market_price() * math.exp(
            expected_log_return * self.time_window_size
        )
        assert self.is_finite(expected_future_price)

        return expected_future_price