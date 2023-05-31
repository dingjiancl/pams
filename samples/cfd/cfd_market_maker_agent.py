import math
import random
from pams.agents import Agent, HighFrequencyAgent
from typing import Any, Dict, List
from pams.market import Market
from pams.order import MARKET_ORDER, LIMIT_ORDER
from cfd_market import CFDOrder, CFDMarket
from typing import Union
from pams.order import Order
from pams.order import Cancel


class CFDMarketMakerAgent(HighFrequencyAgent):
    def setup(  # type: ignore
        self,
        settings: Dict[str, Any],
        accessible_markets_ids: List[int],
        *args,
        **kwargs,
    ) -> None:
        super(CFDMarketMakerAgent, self).setup(
            settings, accessible_markets_ids, *args, **kwargs
        )

        if "bidOrderVolume" not in settings:
            raise ValueError("bidOrderVolume is required for MarketMakerAgent")
        if not isinstance(settings["bidOrderVolume"], int):
            raise ValueError("bidOrderVolume have to be int")
        self.bid_order_volume = settings["bidOrderVolume"]
        if "askOrderVolume" not in settings:
            raise ValueError("askOrderVolume is required for MarketMakerAgent")
        if not isinstance(settings["askOrderVolume"], int):
            raise ValueError("askOrderVolume have to be int")
        self.ask_order_volume = settings["askOrderVolume"]

        if "bidSpread" not in settings:
            raise ValueError("bidSpread is required for MarketMakerAgent")
        self.bid_spread = settings["bidSpread"]
        if "askSpread" not in settings:
            raise ValueError("askSpread is required for MarketMakerAgent")
        self.ask_spread = settings["askSpread"]

        if "orderTimeLengthSpread" in settings:
            if not isinstance(settings["orderTimeLengthSpread"], int):
                raise ValueError("orderTimeLengthSpread have to be int")
            self.order_time_length_spread = settings["orderTimeLengthSpread"]
        if "orderTimeLengthHedge" in settings:
            if not isinstance(settings["orderTimeLengthHedge"], int):
                raise ValueError("orderTimeLengthHedge have to be int")
            self.order_time_length_hedge = settings["orderTimeLengthHedge"]

        if "rateHedgeBuy" not in settings:
            raise ValueError("rateHedgeBuy is required for MarketMakerAgent")
        self.rate_hedge_buy = settings["rateHedgeBuy"]
        if "rateHedgeSell" not in settings:
            raise ValueError("rateHedgeSell is required for MarketMakerAgent")
        self.rate_hedge_sell = settings["rateHedgeSell"]

        self.positions: List[Dict[str, Any]] = []

    def submit_orders(self, markets: List[Market]) -> List[CFDOrder]:
        
        orders: List[Union[Order, Cancel]] = []
        
        cfds: CFDMarket = [market for market in markets if isinstance(market, CFDMarket)]
        underlyings: Market = [market for market in markets if isinstance(market, Market)]
        
        for cfd in cfds:
            if cfd.underlying_market not in underlyings:
                raise Exception("The list of CFD markets must correspond one-to-one with the list of underlying markets.")
            underlying = cfd.underlying_market
            if not cfd.is_running or not underlying.is_running:
                return orders

            # ----Spread + hedge----

            # Spread
            if (
                True  # Todo: Conditions for MM agent to place buy orders in CFD market
            ):
                bid_price = underlying.get_market_price() - self.bid_spread
                bid_volume = self.bid_order_volume
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=cfd.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=bid_volume,
                        price=bid_price,
                        ttl=self.order_time_length_spread,
                    )
                )

            if (
                    True  # Todo: Conditions for MM agent to place sell orders in CFD market
            ):
                ask_price = underlying.get_market_price() + self.ask_spread
                ask_volume = self.ask_order_volume
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=cfd.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=ask_volume,
                        price=ask_price,
                        ttl=self.order_time_length_spread,
                    )
                )

            # Hedging price fluctuations risk
            if (
                    True  # Todo: Conditions for MM agent to hedge by buy orders in CFD market
            ):
                hedge_buy_price = underlying.get_market_price()
                hedge_buy_volume_ = self.rate_hedge_buy * ask_volume
                hedge_buy_volume = math.floor(hedge_buy_volume_)  # Round down
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=underlying.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=hedge_buy_volume,
                        price=hedge_buy_price,
                        ttl=self.order_time_length_hedge,
                    )
                )

            if (
                    True  # Todo: Conditions for MM agent to hedge by sell orders in CFD market
            ):
                hedge_sell_price = underlying.get_market_price()
                hedge_sell_volume_ = self.rate_hedge_sell * bid_volume
                hedge_sell_volume = math.floor(hedge_sell_volume_)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=underlying.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=hedge_sell_volume,
                        price=hedge_sell_price,
                        ttl=self.order_time_length_hedge,
                    )
                )

            # ----Spread + hedge</>----

            return orders