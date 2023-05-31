import heapq
import random
from pams.agents import Agent
from pams.index_market import Market
from pams.logs import OrderLog, ExecutionLog
from pams.logs.base import ExecutionLog, Logger, OrderLog
from pams.order import Order, OrderKind, LIMIT_ORDER, MARKET_ORDER
from pams.simulator import Simulator
from typing import Any, List, Optional, Dict

class CFDOrder(Order):
    """Contract for Difference (CFD) Order class.

    This order type must be submitted only to CFD market.
    """
    def __init__(
        self,
        agent_id: int,
        market_id: int,
        is_buy: bool,
        kind: OrderKind,
        volume: int,
        placed_at: Optional[int] = None,
        price: Optional[float] = None,
        order_id: Optional[int] = None,
        ttl: Optional[int] = None,
        is_to_close_position: Optional[bool] = None,
        target_order_id: Optional[int] = None,
        target_order_price: Optional[float] = None,
        leverage_rate: Optional[float] = None
    ):
        """initialization.

        Args:
            agent_id (int): agent ID.
            market_id (int): market ID.
            is_buy (bool): whether the order is buy order or not.
            kind (:class:`pams.order.OrderKind`): kind of order.
            volume (int): order volume.
            placed_at (int, Optional): time step that the order is placed. (Set by market. Please do not set it in agent)
            price (float, Optional): order price.
            order_id (int, Optional): order ID. (Set by market. Please do not set it in agent)
            ttl (int, Optional): time to order expiration.
            is_to_close_position (bool):  whether the order is to close position.
            target_order_id (int): the target position ID to close.
            target_order_price (bool): the target position price to close.
            leverage_rate (float): leverage rate.
        """
        super().__init__(
            agent_id,
            market_id,
            is_buy,
            kind,
            volume,
            placed_at,
            price,
            order_id,
            ttl
        )
        self.is_to_close_position = is_to_close_position
        self.target_order_id = target_order_id
        self.target_order_price = target_order_price
        self.leverage_rate = leverage_rate

class CFDExecutionLog(ExecutionLog):
    """Contract for Difference (CFD) Execution type log class.

    This log is usually generated when an order is executed on CFD markets.
    """
    def __init__(
        self,
        market_id: int,
        time: int,
        buy_agent_id: int,
        sell_agent_id: int,
        buy_order_id: int,
        sell_order_id: int,
        price: float,
        volume: int,
        is_to_close_position: Optional[bool] = None,
        target_order_id: Optional[int] = None,
        target_order_price: Optional[float] = None,
        leverage_rate: Optional[float] = None,
        is_buyside_mm: Optional[bool] = None,
    ):
        """initialize.

        Args:
            market_id (int): market ID.
            time (int): time to execute.
            buy_agent_id (int): buyer agent ID.
            sell_agent_id (int): seller agent ID.
            buy_order_id (int): buy order ID.
            sell_order_id (int): sell order ID.
            price (float): executed price.
            volume (int): executed volume.
            is_to_close_position (bool): whether the CFD market order is to close position.
            target_order_price (float): the target position price to close.
            leverage_rate (float): leverage rate.
            is_buyside_mm (bool): whether the buy side of the order is submitted by MM.
        """
        super().__init__(
            market_id,
            time,
            buy_agent_id,
            sell_agent_id,
            buy_order_id,
            sell_order_id,
            price,
            volume
        )
        self.is_to_close_position = is_to_close_position
        self.target_order_id = target_order_id
        self.target_order_price = target_order_price
        self.leverage_rate = leverage_rate
        self.is_buyside_mm = is_buyside_mm

class CFDSimulator(Simulator):
    """Contract for Difference (CFD) Simulator class.

    The simulator contains CFD Market in self.markets.
    """
    def _update_agents_for_execution(self, execution_logs: List[Any]) -> None:
        """update agents for execution. (Usually, this is called from runner.)

        execution_logs may include CFDExecutionLog, that must be executed by CFD manner.

        There are 4 cases of execution scenarios.
        1. buy_agent is MM. sell_agent wants to open the position.
            sell_agent pays (p_bid * volume * leverage_rate) as margin.
            
            Investors wants to open a short position, 
            pays (p_bid * volume * leverage_rate) to the CFDmarket as margin.

        2. buy_agent is MM. sell_agent wants to close the position.
            sell_agent receives the margin and the difference from MM.
            margin = target_prder_price * volume * leverage_rate
            difference = (p_bid - target_order_price) * volume
            
            Investors wants to close a short position, 
            receive margin + difference

        3. sell_agent is MM. buy_agent wants to open the position.
            buy_agent pays (p_ask * volume * leverage_rate) to MM as margin.
            
            Investors wants to open a long position,
            pays (p_ask * volume * leverage_rate) to the CFDMarket as margin

        4. sell_agent is MM. buy_agent wants to close the position.
            buy_agent receives the margin and the difference from MM.
            difference = - (p_ask - target_order_price) * volume
            
            Investors wants to close a short position,
            receive margin + difference


        Args:
            execution_logs (List[ExecutionLog | CFDExecutionLog]): execution logs.

        Returns:
            None
        """
        for log in execution_logs:
            buy_agent: Agent = self.id2agent[log.buy_agent_id]
            sell_agent: Agent = self.id2agent[log.sell_agent_id]
            price: float = log.price
            volume: int = log.volume
            market_id: int = log.market_id
            is_cfd = (log.__class__ == CFDExecutionLog)
            if is_cfd:
                is_buy_mm = log.is_buyside_mm
                is_buy_client = bool(1 - is_buy_mm)
                is_to_close_position = log.is_to_close_position
                leverage_rate = log.leverage_rate
                if is_buy_mm:
                    mm_agent = buy_agent
                    mm_order_id = log.buy_order_id
                    client_agent = sell_agent
                    client_order_id = log.sell_order_id
                else:
                    mm_agent = sell_agent
                    mm_order_id = log.sell_order_id
                    client_agent = buy_agent
                    client_order_id = log.buy_order_id
                if is_to_close_position:
                    target_order_id = log.target_order_id
                    target_order_price = log.target_order_price
                    is_target_in_position: bool = False
                    for p in client_agent.positions:
                        if target_order_id == p["order_id"]:
                            is_target_in_position = True
                            if leverage_rate != p["leverage_rate"]:
                                raise ValueError(
                                    f"leverage_rate must be set to the same value as when the position is opened: " +
                                    f'{leverage_rate} != {p["leverage_rate"]}'
                                )
                    if not is_target_in_position:
                        raise ValueError(
                            f"target order: {target_order_id} is not in positions of agent {client_agent.agent_id}."
                        )
                    diff = (price - target_order_price) * volume
                    margin = target_order_price * volume * leverage_rate
                    if not is_buy_mm:
                        diff = - diff
                    client_agent.cash_amount += (diff + margin)
                    mm_agent.cash_amount -= diff
                    client_agent.positions.remove(
                        {
                            "order_id": target_order_id,
                            "is_buy": bool(1 - is_buy_client),
                            "price": target_order_price,
                            "volume": volume,
                            "leverage_rate": leverage_rate
                        }
                    )
                else:
                    margin = price * volume * leverage_rate
                    client_agent.cash_amount -= margin
                    client_agent.positions.append(
                        {
                            "order_id": client_order_id,
                            "is_buy": is_buy_client,
                            "price": price,
                            "volume": volume,
                            "leverage_rate": leverage_rate
                        }
                    )
            else:
                buy_agent.cash_amount -= price * volume
                sell_agent.cash_amount += price * volume
                buy_agent.asset_volumes[market_id] += volume
                sell_agent.asset_volumes[market_id] -= volume

class CFDMarket(Market):
    """Contract for Difference (CFD) Market class."""
    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: CFDSimulator,
        name: str,
        logger: Optional[Logger] = None
    ) -> None:
        """initialization.

        ask_limit_orders is a list to store unexecuted sell limit orders submitted by MM agents.
        bid_limit_orders is a list to store unexecuted buy limit orders submitted by MM agents.
        ask_market_orders is a list to store unexecuted sell market orders submitted by normal agents.
        bid_market_orders is a list to store unexecuted buy market orders submitted by normal agents.

        Args:
            market_id (int): market ID.
            prng (random.Random): pseudo random number generator for this market.
            simulator (:class:`pams.Simulator`): cfd simulator that executes this market.
            name (str): market name.
            logger (Logger, Optional): logger.

        Returns:
            None
        """
        super().__init__(market_id, prng, simulator, name, logger)
        self.ask_limit_orders: List[CFDOrder] = []
        self.bid_limit_orders: List[CFDOrder] = []
        self.ask_market_orders: List[CFDOrder] = []
        self.bid_market_orders: List[CFDOrder] = []

    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        if "underlyingMarket" not in settings:
            raise ValueError("underlyingMarket is required")
        self.underlying_market_name = settings["underlyingMarket"]
        self.underlying_market = self.simulator.name2market[settings["underlyingMarket"]]

    def _add_order(self, order: CFDOrder) -> OrderLog:
        """add order. (Usually, only triggered by runner)

        Timit orders must besubmitted by MM and
        market orders must be submitted by normal agents.
        Market orders are sorted by their order IDs and
        limit orders are sorted by their limit prices.

        Args:
            order (:class:CFDOrder): order.

        Returns:
            :class:`pams.logs.base.OrderLog`: order log.
        """
        order_log = super()._add_order(order)
        if order.kind == MARKET_ORDER:
            if order.is_to_close_position:
                if (
                    order.target_order_id is None
                    or order.target_order_price is None
                    or order.leverage_rate is None
                ):
                    raise ValueError(
                        "target_order_id, target_order_price, and leverage_rate are required to close positions."
                    )
            if order.is_buy:
                heapq.heappush(self.bid_market_orders, (order.order_id, order))
            else:
                heapq.heappush(self.ask_market_orders, (order.order_id, order))
        else:
            if order.is_buy:
                heapq.heappush(self.bid_limit_orders, (-order.price, order))
            else:
                heapq.heappush(self.ask_limit_orders, (order.price, order))
        return order_log

    def _execute_orders(self, price: float, volume: int,
                        buy_order: CFDOrder, sell_order: CFDOrder) -> CFDExecutionLog:
        """execute orders. (Internal method)

        Identify which side of order (bid/ask) is submitted by MM (=limit order) and
        whether the order by the ordinal agent (=market order) is to close position.

        If the market order is to close position, keep the target order price to
        calculate the difference.

        Args:
            price (float): price.
            volume (int): volume.
            buy_order (:class:CFDOrder): buy order.
            sell_order (:class:CFDOrder): sell order.

        Returns:
            :class:CFDExecutionLog: execution log.
        """
        execution_log = super()._execute_orders(price, volume, buy_order, sell_order)
        if buy_order.kind == MARKET_ORDER:
            if sell_order.kind != LIMIT_ORDER:
                raise ValueError("Both buy and sell orders are market order")
            is_to_close_position = buy_order.is_to_close_position
            target_order_id = buy_order.target_order_id
            target_order_price = buy_order.target_order_price
            leverage_rate = buy_order.leverage_rate
            is_buyside_mm = False
        else:
            if sell_order.kind != MARKET_ORDER:
                raise ValueError("Both buy and sell orders are limit order")
            is_to_close_position = sell_order.is_to_close_position
            target_order_id = sell_order.target_order_id
            target_order_price = sell_order.target_order_price
            leverage_rate = sell_order.leverage_rate
            is_buyside_mm = True
        log = CFDExecutionLog(
            market_id=execution_log.market_id,
            time=execution_log.time,
            buy_agent_id=execution_log.buy_agent_id,
            sell_agent_id=execution_log.sell_agent_id,
            buy_order_id=execution_log.buy_order_id,
            sell_order_id=execution_log.sell_order_id,
            price=execution_log.price,
            volume=execution_log.volume,
            is_to_close_position=is_to_close_position,
            target_order_id=target_order_id,
            target_order_price=target_order_price,
            leverage_rate=leverage_rate,
            is_buyside_mm=is_buyside_mm,
        )
        return log

    def _execution(self) -> List[CFDExecutionLog]:
        """execute for market. (Usually, only triggered by runner)

        If unexecuted market orders remain, they are executed against
        the opposite limit orders with the best limit price.

        Returns:
            List[:class:CFDExecutionLog]: execution logs.
        """
        executionlog_list: List[CFDExecutionLog] = []
        while True:
            if len(self.bid_market_orders) == 0 and len(self.ask_market_orders) == 0:
                break
            if not len(self.ask_market_orders) == 0:
                _, ask_market_order = heapq.heappop(self.ask_market_orders)
                _, bid_limit_order = heapq.heappop(self.bid_limit_orders)
                volume = min(ask_market_order.volume, bid_limit_order.volume)
                executionlog_list.append(
                    self._execute_orders(
                        price=bid_limit_order.price,
                        volume=volume,
                        buy_order=bid_limit_order,
                        sell_order=ask_market_order
                    )
                )
                if ask_market_order.volume != 0:
                    heapq.heappush(self.ask_market_orders,
                                (ask_market_order.order_id, ask_market_order))
                if bid_limit_order.volume != 0:
                    heapq.heappush(self.bid_limit_orders,
                                (-bid_limit_order.price, bid_limit_order))
            if not len(self.bid_market_orders) == 0:
                _, bid_market_order = heapq.heappop(self.bid_market_orders)
                _, ask_limit_order = heapq.heappop(self.ask_limit_orders)
                volume = min(bid_market_order.volume, ask_limit_order.volume)
                executionlog_list.append(
                    self._execute_orders(
                        price=ask_limit_order.price,
                        volume=volume,
                        buy_order=bid_market_order,
                        sell_order=ask_limit_order
                    )
                )
                if bid_market_order.volume != 0:
                    heapq.heappush(self.bid_market_orders,
                                (bid_market_order.order_id, bid_market_order))
                if ask_limit_order.volume != 0:
                    heapq.heappush(self.ask_limit_orders,
                                (ask_limit_order.price, ask_limit_order))
        return executionlog_list