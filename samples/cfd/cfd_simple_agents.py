from pams.agents import Agent, HighFrequencyAgent
from typing import Any, Dict, List
from pams.market import Market
from pams.order import MARKET_ORDER, LIMIT_ORDER
from cfd_market import CFDOrder, CFDMarket

class CFDSimpleAgent(Agent):
    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        super(CFDSimpleAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )
        if "leverage_rate" not in settings:
            raise ValueError("leverage_rate is required for CFDSimpleAgent")
        self.leverage_rate = settings["leverage_rate"]
        self.positions: List[Dict[str, Any]] = []
        # Ex: [{"is_buy": True, "price":300.0, "volume": 1, "leverage_rate":0.1}, ...]

    def submit_orders(self, markets: List[Market]) -> List[CFDOrder]:
        for market in markets:
            if self.is_market_accessible(market_id=market.market_id):
                cfd_market = market
        orders: List[CFDOrder] = []
        for pos_dic in self.positions:
            is_buy = pos_dic["is_buy"]
            target_order_id = pos_dic["order_id"]
            price = pos_dic["price"]
            volume = pos_dic["volume"]
            leverage_rate = pos_dic["leverage_rate"]
            orders.append(
                CFDOrder(
                    agent_id=self.agent_id,
                    market_id=cfd_market.market_id,
                    is_buy=bool(1-is_buy),
                    kind=MARKET_ORDER,
                    volume=volume,
                    price=None,
                    ttl=int(1e+10),
                    is_to_close_position=True,
                    target_order_id=target_order_id,
                    target_order_price=price,
                    leverage_rate=leverage_rate
                )
            )
        submit_new_order: bool = bool(self.prng.randint(0,1))
        if not submit_new_order:
            return orders
        is_buy: bool = bool(self.prng.randint(0,1))
        orders.append(
            CFDOrder(
                agent_id=self.agent_id,
                market_id=cfd_market.market_id,
                is_buy=is_buy,
                kind=MARKET_ORDER,
                volume=1,
                price=None,
                ttl=int(1e+10),
                is_to_close_position=False,
                target_order_price=None,
                leverage_rate=self.leverage_rate
            )
        )
        return orders

class CFDSimpleMarketMakerAgent(HighFrequencyAgent):
    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        super(CFDSimpleMarketMakerAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )
        if "bid_ask_spread" not in settings:
            raise ValueError("bid_ask_spread is required for CFDSimpleMarketMakerAgent")
        self.bid_ask_spread: float = settings["bid_ask_spread"]
        self.positions: List[Dict[str, Any]] = []

    def submit_orders(self, markets: List[Market]) -> List[CFDOrder]:
        estimated_index: float = 0.0
        asset_num: int = len(markets) - 1
        for market in markets:
            if market.__class__ != CFDMarket:
                estimated_index += market.get_market_price() / asset_num
            else:
                cfd_market_id = market.market_id
        p_ask = estimated_index + self.bid_ask_spread / 2
        p_bid = estimated_index - self.bid_ask_spread / 2
        orders: List[CFDOrder] = [
            CFDOrder(
                agent_id=self.agent_id,
                market_id=cfd_market_id,
                is_buy=True,
                kind=LIMIT_ORDER,
                volume=100,
                price=p_bid,
                ttl=1
            ),
            CFDOrder(
                agent_id=self.agent_id,
                market_id=cfd_market_id,
                is_buy=False,
                kind=LIMIT_ORDER,
                volume=100,
                price=p_ask,
                ttl=1
            )
        ]
        return orders
