import heapq
import math
import random
import warnings
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import cast

from .logs.base import CancelLog
from .logs.base import ExecutionLog
from .logs.base import Log
from .logs.base import Logger
from .logs.base import OrderLog
from .order import Cancel
from .order import Order
from .order_book import OrderBook

T = TypeVar("T")


class Market:
    """Market class.

    .. seealso::
        - :class:`pams.index_market.IndexMarket`: IndexMarket
    """

    # 初始化方法，接受指定参数并初始化一些属性
    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ) -> None:
        """initialization.

        Args:
            market_id (int): market ID.
            prng (random.Random): pseudo random number generator for this market.
            simulator (:class:`pams.Simulator`): simulator that executes this market.
            name (str): market name.
            logger (Logger, Optional): logger.

        Returns:
            None
        """
        self.market_id: int = market_id
        self._prng = prng
        self.logger: Optional[Logger] = logger
        self._is_running: bool = False
        self.tick_size: float = 1.0
        self.chunk_size = 100  # 每个时间单位市场可以处理的最大订单数量
        self.sell_order_book: OrderBook = OrderBook(is_buy=False)
        self.buy_order_book: OrderBook = OrderBook(is_buy=True)
        self.time: int = -1
        self._market_prices: List[Optional[float]] = []
        self._last_executed_prices: List[Optional[float]] = []
        self._mid_prices: List[Optional[float]] = []
        self._fundamental_prices: List[Optional[float]] = []
        self._executed_volumes: List[int] = []
        self._executed_total_prices: List[float] = []
        self._n_buy_orders: List[int] = []
        self._n_sell_orders: List[int] = []
        self._next_order_id: int = 0
        self.simulator: "Simulator" = simulator  # type: ignore  # NOQA
        self.name: str = name
        self.outstanding_shares: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.market_id}, name={self.name}, "
            f"tick_size={self.tick_size}, outstanding_shares={self.outstanding_shares}>"
        )

    # 设置方法setup，根据settings来设置市场的一些属性
    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore  # NOQA
        """setup market configuration from setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameters "tickSize" and either "marketPrice" or "fundamentalPrice".
                                       This can include the parameter "outstandingShares".

        Returns:
            None
        """
        if "tickSize" not in settings:
            raise ValueError("tickSize is required")
        self.tick_size = settings["tickSize"]
        if "outstandingShares" in settings:
            if not isinstance(settings["outstandingShares"], int):
                raise ValueError("outstandingShares must be int")
            self.outstanding_shares = settings["outstandingShares"]
        if "marketPrice" in settings:
            self._market_prices = [float(settings["marketPrice"])]
        elif "fundamentalPrice" in settings:
            self._market_prices = [float(settings["fundamentalPrice"])]
        else:
            raise ValueError("fundamentalPrice or marketPrice is required for market")

    # ----一些获取市场状态的方法（如市场价格、成交量、成交总价、买方和卖方订单簿等）----

    # 从Simulator记录的一系列时间点的参数中提取出相应的参数值，返回一个包含这些参数值的列表
    # 从parameters列表中按照给定时间序列times提取数据（提取多个元素）
    def _extract_sequential_data_by_time(
        self,
        times: Union[Iterable[int], None],
        # Union:类型注解工具，表示可接受多种类型中的一个；这里表示可以接受Iterable[int]或None两种类型
        # Iterable[int]: 表示可以被迭代的对象，其中的元素都是整数
        parameters: List[Optional[T]],
        allow_none: bool = False,
    ) -> List[Optional[T]]:  # Optional[T]：可以为T类型或None
        """extract sequential parameters by time. (Internal method)

        Args:
            times (Union[Iterable[int], None]): range of time steps.
            parameters (List[Optional[T]]): referenced parameters.
            allow_none (bool): whether a None result can be returned.

        Returns:
            List[Optional[T]]: extracted parameters.
        """
        if times is None:
            times = range(self.time + 1)  # 默认提取当前时间之前所有的时间点
        if sum([t > self.time for t in times]) > 0:  # 若存在任何一个时间点在当前时间之后
            raise AssertionError("Cannot refer the future parameters")  # 引发断言错误：无法引用未来的参数
        result = [parameters[t] for t in times]  # 将times对应的参数值放入result中
        if not allow_none and None in result:  # 对result进行None值检查
            raise AssertionError
        return result

    # 从parameters列表中按照给定时间time提取数据（提取一个元素）
    def _extract_data_by_time(
        self,
        time: Union[int, None],
        parameters: List[Optional[T]],
        allow_none: bool = False,
    ) -> Optional[T]:
        """extract a parameter by time. (Internal method)

        Args:
            time (Union[int, None]): time step.
            parameters (List[Optional[T]]): referenced parameters.
            allow_none (bool): whether a None result can be returned.

        Returns:
            Optional[T]: extracted parameter.
        """
        if time is None:
            time = self.time
        if time > self.time:
            raise AssertionError("Cannot refer the future parameters")
        result = parameters[time]
        if not allow_none and result is None:
            raise AssertionError
        return result

    def get_time(self) -> int:
        """get time step."""
        return self.time

    # 获取市场价格序列
    def get_market_prices(
        self, times: Union[Iterable[int], None] = None  # 接受一个可迭代的整数序列times
    ) -> List[float]:
        """get market prices.

        Args:
            times (Union[Iterable[int], None]): range of time steps.

        Returns:
            List[float]: extracted sequential data.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(
                times, self._market_prices, allow_none=False
            ),
        )

    # 获取某时间的市场价格
    def get_market_price(self, time: Union[int, None] = None) -> float:
        """get market price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: extracted data.
        """
        return cast(
            float,
            self._extract_data_by_time(time, self._market_prices, allow_none=False),
        )

    def get_mid_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[Optional[float]]:
        """get middle prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[Optional[float]]: middle prices.
        """
        return self._extract_sequential_data_by_time(
            times, self._mid_prices, allow_none=True
        )

    def get_mid_price(self, time: Union[int, None] = None) -> Optional[float]:
        """get middle price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float, Optional: middle price.
        """
        return self._extract_data_by_time(time, self._mid_prices, allow_none=True)

    def get_last_executed_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[Optional[float]]:
        """get prices executed last steps.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[Optional[float]]: prices.
        """
        return self._extract_sequential_data_by_time(
            times, self._last_executed_prices, allow_none=True
        )

    def get_last_executed_price(self, time: Union[int, None] = None) -> Optional[float]:
        """get price executed last step.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float, Optional: price.
        """
        return self._extract_data_by_time(
            time, self._last_executed_prices, allow_none=True
        )

    def get_fundamental_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[float]:
        """get fundamental prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[float]: fundamental prices.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(times, self._fundamental_prices),
        )

    def get_fundamental_price(self, time: Union[int, None] = None) -> float:
        """get fundamental price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: fundamental price.
        """
        return cast(float, self._extract_data_by_time(time, self._fundamental_prices))

    def get_executed_volumes(
        self, times: Union[Iterable[int], None] = None
    ) -> List[int]:
        """get executed volumes.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: volumes.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._executed_volumes)
            ),
        )

    def get_executed_volume(self, time: Union[int, None] = None) -> int:
        """get executed volume.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: volume.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._executed_volumes)
            ),
        )

    def get_executed_total_prices(
        self, times: Union[Iterable[int], None] = None
    ) -> List[float]:
        """get executed total prices.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[float]: total prices.
        """
        return cast(
            List[float],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._executed_total_prices)
            ),
        )

    def get_executed_total_price(self, time: Union[int, None] = None) -> float:
        """get executed total price.

        Args:
            time (Union[int, None]): time step.

        Returns:
            float: total price.
        """
        return cast(
            float,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._executed_total_prices)
            ),
        )

    def get_n_buy_orders(self, times: Union[Iterable[int], None] = None) -> List[int]:
        """get the number of buy orders.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: number of buy orders.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._n_buy_orders)
            ),
        )

    def get_n_buy_order(self, time: Union[int, None] = None) -> int:
        """get the number of buy order.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: number of buy order.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._n_buy_orders)
            ),
        )

    def get_n_sell_orders(self, times: Union[Iterable[int], None] = None) -> List[int]:
        """get the number of sell orders.

        Args:
            times (Union[Iterable[int], None]): time steps.

        Returns:
            List[int]: number of sell orders.
        """
        return cast(
            List[int],
            self._extract_sequential_data_by_time(
                times, cast(List[Optional[int]], self._n_sell_orders)
            ),
        )

    def get_n_sell_order(self, time: Union[int, None] = None) -> int:
        """get the number of sell order.

        Args:
            time (Union[int, None]): time step.

        Returns:
            int: number of sell order.
        """
        return cast(
            int,
            self._extract_data_by_time(
                time, cast(List[Optional[int]], self._n_sell_orders)
            ),
        )

    # ----一些获取市场状态的方法（如市场价格、成交量、成交总价、买方和卖方订单簿等）</>----

    # 用None值或0补全（价格、数量等）列表，以便将它们扩展到给定时间
    # 以确保所有列表都有足够的长度来容纳该时间点的数据
    def _fill_until(self, time: int) -> None:
        if len(self._mid_prices) >= time + 1:
            return
        # 需要填充到的长度（确保列表的长度是chunk_size的整数倍）
        length = (time // self.chunk_size + 1) * self.chunk_size
        # 将以下的数据列表长度扩展到length
        # 对于前四个列表（价格数据），添加None元素
        # 对于后四个列表（交易量和订单数据），添加0元素。
        self._market_prices = self._market_prices + [
            None for _ in range(length - len(self._market_prices))
        ]
        self._mid_prices = self._mid_prices + [
            None for _ in range(length - len(self._mid_prices))
        ]
        self._last_executed_prices = self._last_executed_prices + [
            None for _ in range(length - len(self._last_executed_prices))
        ]
        self._fundamental_prices = self._fundamental_prices + [
            None for _ in range(length - len(self._fundamental_prices))
        ]
        self._executed_volumes = self._executed_volumes + [
            0 for _ in range(length - len(self._executed_volumes))
        ]
        self._executed_total_prices = self._executed_total_prices + [
            0 for _ in range(length - len(self._executed_total_prices))
        ]
        self._n_buy_orders = self._n_buy_orders + [
            0 for _ in range(length - len(self._n_buy_orders))
        ]
        self._n_sell_orders = self._n_sell_orders + [
            0 for _ in range(length - len(self._n_sell_orders))
        ]

    # 返回指定时间之前的成交量加权平均价
    def get_vwap(self, time: Optional[int] = None) -> float:
        """get VWAP.

        Args:
            time (int, Optional): time step.

        Returns:
            float: VWAP.
        """
        if time is None:
            time = self.time
        if time > self.time:
            raise AssertionError("Cannot refer the future parameters")
        if sum(self._executed_volumes[: time + 1]) == 0:
            return float("nan")
        return sum(self._executed_total_prices[: time + 1]) / sum(
            self._executed_volumes[: time + 1]
        )

    @property  # @property：将方法转化为只读属性（可直接通过market.is_running而不是is_running()来调用）
    def is_running(self) -> bool:
        """get whether this market is running or not.

        Returns:
            bool: whether this market is running or not.
        """
        return self._is_running

    # 获取当前买单订单簿中最优买价
    def get_best_buy_price(self) -> Optional[float]:
        """get the best buy price.

        Returns:
            float, Optional: the best buy price.
        """
        return self.buy_order_book.get_best_price()

    def get_best_sell_price(self) -> Optional[float]:
        """get the best sell price.

        Returns:
            float, Optional: the best sell price.
        """
        return self.sell_order_book.get_best_price()

    # 获取当前卖单订单簿中所有订单价格和对应数量的字典
    def get_sell_order_book(self) -> Dict[Optional[float], int]:
        """get sell order book.

        Returns:
            Dict[Optional[float], int]: sell order book.
        """
        return self.sell_order_book.get_price_volume()

    def get_buy_order_book(self) -> Dict[Optional[float], int]:
        """get buy order book.

        Returns:
            Dict[Optional[float], int]: buy order book.
        """
        return self.buy_order_book.get_price_volume()

    def convert_to_tick_level_rounded_lower(self, price: float) -> int:
        """convert price to tick level rounded lower.

        Args:
            price (float): price.

        Returns:
            int: price for tick level rounded lower.
        """
        return math.floor(price / self.tick_size)

    def convert_to_tick_level_rounded_upper(self, price: float) -> int:
        """convert price to tick level rounded upper.

        Args:
            price (float): price.

        Returns:
            int: price for tick level rounded upper.
        """
        return math.ceil(price / self.tick_size)

    # 根据买卖方向对价格取整
    def convert_to_tick_level(self, price: float, is_buy: bool) -> int:
        """convert price to tick level. If it is buy order price, it is rounded lower. If it is sell order price, it is rounded upper.

        Args:
            price (float): price.
            is_buy (bool): buy order or not.

        Returns:
            int: price for tick level.
        """
        if is_buy:
            return self.convert_to_tick_level_rounded_lower(price=price)
        else:
            return self.convert_to_tick_level_rounded_upper(price=price)

    def convert_to_price(self, tick_level: int) -> float:
        """convert tick to price.

        Args:
            tick_level (int): tick level.

        Returns:
            float: price.
        """
        return self.tick_size * tick_level

    # 设置市场时间及该时间的价格
    def _set_time(self, time: int, next_fundamental_price: float) -> None:
        """set time step. (Usually, only triggered by simulator)

        Args:
            time (int): time step.
            next_fundamental_price (float): next fundamental price.

        Returns:
            None
        """
        # ----更新时间----
        self.time = time
        self.buy_order_book._set_time(time)
        self.sell_order_book._set_time(time)
        # ----更新时间</>----
        self._fill_until(time=time)  # 如果当前存储数据的数组长度不足time，则通过添加None扩充数组，以便存储当前时间点的数据
        self._fundamental_prices[self.time] = next_fundamental_price  # 数组self._fundamental_prices：所有时间点的基本价格
        if self.time > 0:
            # ----更新当前时间点的最后成交价、中间价格和市场价格----
            executed_prices: List[float] = cast(  # cast()：类型转换函数；将一个对象显式地转换为另一个类型
                List[float],
                list(
                    filter(  # 过滤掉数组中的None值
                        lambda x: x is not None, self._last_executed_prices[: self.time]
                    )
                ),
            )
            self._last_executed_prices[self.time] = (
                executed_prices[-1] if sum(executed_prices) > 0 else None
            )
            mid_prices: List[float] = cast(
                List[float],
                list(filter(lambda x: x is not None, self._mid_prices[: self.time])),
            )
            self._mid_prices[self.time] = (
                mid_prices[-1] if sum(mid_prices) > 0 else None
            )
            market_prices: List[float] = cast(
                List[float],
                list(filter(lambda x: x is not None, self._market_prices[: self.time])),
            )
            self._market_prices[self.time] = (
                market_prices[-1] if sum(market_prices) > 0 else None
            )
            # ----更新当前时间点的最后成交价、中间价格和市场价格</>----

            # ----更新当前时间点的市场价格----
            if self.is_running:
                if self._last_executed_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._last_executed_prices[
                        self.time
                    ]
                elif self._mid_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._mid_prices[self.time]
            # ----更新当前时间点的市场价格</>----

    # 完成模拟交易市场时间向前推进一步的操作
    # （更新当前市场时间与当前市场价格）
    # 更新市场的时间，并同步更新相关的价格、订单簿和其他信息
    def _update_time(self, next_fundamental_price: float) -> None:
        """update time. (Usually, only triggered by simulator)

        Args:
            next_fundamental_price (float): next fundamental price.

        Returns:
            None
        """
        self.time += 1
        self.buy_order_book._set_time(self.time)
        self.sell_order_book._set_time(self.time)
        self._fill_until(time=self.time)
        self._fundamental_prices[self.time] = next_fundamental_price
        if self.time > 0:
            # ----将上一个时刻的最近成交价、中间价格、市场价格分别赋值给当前时刻----
            self._last_executed_prices[self.time] = self._last_executed_prices[
                self.time - 1
            ]
            self._mid_prices[self.time] = self._mid_prices[self.time - 1]
            self._market_prices[self.time] = self._market_prices[self.time - 1]
            # ----将上一个时刻的最近成交价、中间价格、市场价格分别赋值给当前时刻</>----

            # 若上个时刻的最近成交价不为空，则当前时刻市场价格=上一个时刻的最近成交价
            # 否则，当前时刻市场价格=上一个时刻的中间价格
            if self.is_running:
                if self._last_executed_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._last_executed_prices[
                        self.time - 1
                    ]
                elif self._mid_prices[self.time - 1] is not None:
                    self._market_prices[self.time] = self._mid_prices[self.time - 1]
        # 如果当前时间self.time等于零，且当前市场价格为空，则将当前市场价格设为基本价格next_fundamental_price
        else:
            if self._market_prices[self.time] is None:
                self._market_prices[self.time] = next_fundamental_price

    # 取消一个订单
    def _cancel_order(self, cancel: Cancel) -> CancelLog:
        """cancel order. (Usually, only triggered by simulator)

        Args:
            cancel (:class:`pams.order.Cancel`): cancel class.

        Returns:
            :class:`pams.logs.base.CancelLog`: cancel log.
        """
        if self.market_id != cancel.order.market_id:  # 检查待取消的Order对象是否属于该市场
            raise ValueError("this cancel order is for a different market")
        if cancel.order.order_id is None or cancel.order.placed_at is None:  # 检查Order对象是否已经提交
            raise ValueError("the order is not submitted before")
        # 根据Order对象的is_buy属性，从买单或卖单订单簿中找到对应的订单簿，并调用该订单簿的cancel方法来撤销该订单
        (self.buy_order_book if cancel.order.is_buy else self.sell_order_book).cancel(
            cancel=cancel
        )
        if cancel.placed_at is None:
            raise AssertionError
        self._update_market_price()  # 更新该市场的价格

        log: CancelLog = CancelLog(  # 构造一个CancelLog对象，包含了撤销的订单信息
            order_id=cancel.order.order_id,
            market_id=cancel.order.market_id,
            cancel_time=cancel.placed_at,
            order_time=cancel.order.placed_at,
            agent_id=cancel.order.agent_id,
            is_buy=cancel.order.is_buy,
            kind=cancel.order.kind,
            volume=cancel.order.volume,
            price=cancel.order.price,
            ttl=cancel.order.ttl,
        )
        if self.logger is not None:  # 如果有logger，则将该日志写入日志文件中
            log.read_and_write(logger=self.logger)
        # 返回CancelLog对象
        return log

    # 更新市场价格
    def _update_market_price(self) -> None:
        """update market price. (Internal method)"""
        # 获取最佳的买入价和卖出价
        best_buy_price: Optional[float] = self.get_best_buy_price()
        best_sell_price: Optional[float] = self.get_best_sell_price()
        # 如果两者都存在，则计算中间价
        if best_buy_price is None or best_sell_price is None:
            self._mid_prices[self.time] = None
        else:
            self._mid_prices[self.time] = (
                (best_sell_price + best_buy_price) / 2.0
                if best_sell_price is not None and best_buy_price is not None
                else None
            )
        # 设置市场价格，存储在self._market_prices字典中
        # （最后执行的价格 or 中间价格）
        if self.is_running:
            if self._last_executed_prices[self.time] is not None:
                self._market_prices[self.time] = self._last_executed_prices[self.time]
            elif self._mid_prices[self.time] is not None:
                self._market_prices[self.time] = self._mid_prices[self.time]

    # 执行订单
    def _execute_orders(  # 传入价格、交易量以及要被执行的买单和卖单
        self, price: float, volume: int, buy_order: Order, sell_order: Order
    ) -> ExecutionLog:
        """execute orders. (Internal method)

        Args:
            price (float): price.
            volume (int): volume.
            buy_order (:class:`pams.order.Order`): buy order.
            sell_order (:class:`pams.order.Order`): sell order.

        Returns:
            :class:`pams.logs.base.CancelLog`: execution log.
        """
        # ----一些错误判断----
        # 例：市场没有在运行、传入买卖单不是该市场的订单、交易量小于等于0等
        if not self.is_running:
            raise AssertionError("market is not running")
        if buy_order.market_id != self.market_id:
            raise ValueError("buy order is not for this market")
        if sell_order.market_id != self.market_id:
            raise ValueError("sell order is not for this market")

        if buy_order.placed_at is None:
            raise ValueError("buy order is not submitted yet")
        if sell_order.placed_at is None:
            raise ValueError("sell order is not submitted yet")

        if volume <= 0:
            raise AssertionError
        # ----一些错误判断</>----

        # 创建一个ExecutionLog对象，以存储执行的相关信息
        # （包括：市场ID、时间、买家agent ID、卖家agent ID、买单ID、卖单ID、价格、交易量）
        log: ExecutionLog = ExecutionLog(
            market_id=self.market_id,
            time=self.time,
            buy_agent_id=buy_order.agent_id,
            sell_agent_id=sell_order.agent_id,
            buy_order_id=cast(int, buy_order.order_id),
            sell_order_id=cast(int, sell_order.order_id),
            price=price,
            volume=volume,
        )

        # 将订单从买单簿和卖单簿中删除
        self.buy_order_book.change_order_volume(order=buy_order, delta=-volume)
        self.sell_order_book.change_order_volume(order=sell_order, delta=-volume)

        # 记录最后执行的价格、执行的总交易量以及执行的总价格
        self._last_executed_prices[self.time] = price
        self._executed_volumes[self.time] += volume
        self._executed_total_prices[self.time] += volume * price

        # 更新市场价格
        self._update_market_price()

        # ToDo: Agent modification will be handled in simulator
        # 如果指定了logger，则将该执行记录写入日志中
        if self.logger is not None:
            log.read_and_write(logger=self.logger)
        # 返回执行记录对象
        return log

    # 向市场添加订单
    def _add_order(self, order: Order) -> OrderLog:
        """add order. (Usually, only triggered by runner)

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            :class:`pams.logs.base.OrderLog`: order log.
        """
        if order.market_id != self.market_id:  # 判断订单是否属于该市场
            raise ValueError("order is not for this market")
        if order.placed_at is not None:
            raise ValueError("the order is already submitted")
        if order.order_id is not None:  # 订单的order_id是否为空
            raise ValueError("the order is already submitted")

        # 若订单的价格不为空，则判断其是否符合市场的tick_size
        # 若不符合则发出警告，并将其转化为符合市场的价格
        if order.price is not None and order.price % self.tick_size != 0:
            warnings.warn(
                "order price does not accord to the tick size. price will be modified"
            )
            order.price = (
                self.convert_to_tick_level(price=order.price, is_buy=order.is_buy)
                * self.tick_size
            )

        # 生成订单的唯一标识order_id，并将其赋值给订单
        order.order_id = self._next_order_id
        self._next_order_id += 1
        # 根据订单的is_buy属性判断该是买单还是卖单，并添加到相应的订单薄中
        (self.buy_order_book if order.is_buy else self.sell_order_book).add(order=order)
        if order.placed_at != self.time:
            raise AssertionError
        self._update_market_price()  # 更新市场价格

        if order.is_buy:  # 如果是买单，则将该时间点的买单数量_n_buy_orders加1
            self._n_buy_orders[self.time] += 1
        else:  # 如果是卖单，则将该时间点的卖单数量_n_sell_orders加1
            self._n_sell_orders[self.time] += 1

        # 生成该订单的日志对象OrderLog，并将其记录到日志中
        log: OrderLog = OrderLog(
            order_id=order.order_id,
            market_id=order.market_id,
            time=cast(int, order.placed_at),
            agent_id=order.agent_id,
            is_buy=order.is_buy,
            kind=order.kind,
            volume=order.volume,
            price=order.price,
            ttl=order.ttl,
        )
        if self.logger is not None:
            log.read_and_write(logger=self.logger)

        # 返回该订单的日志对象
        return log

    # 判断是否还存在可执行的买单和卖单
    def remain_executable_orders(self) -> bool:
        """check if there are remain executable orders in this market.

        Returns:
            bool: whether some orders is executable or not.
        """
        # 检查卖单薄和买单薄中是否都有至少一个订单（否则无法交易）
        if len(self.sell_order_book) == 0:
            return False
        if len(self.buy_order_book) == 0:
            return False

        sell_best: Order = cast(Order, self.sell_order_book.get_best_order())
        buy_best: Order = cast(Order, self.buy_order_book.get_best_order())
        if sell_best.price is not None or buy_best.price is not None:
            if sell_best.price is not None and buy_best.price is not None:  # 如果卖单薄和买单薄中的最优订单价格均已确定
                # 判断是否存在可交易的订单
                return sell_best.price <= buy_best.price
            else:
                return True
        else:
            sell_book: Dict[
                Optional[float], int
            ] = self.sell_order_book.get_price_volume()
            buy_book: Dict[
                Optional[float], int
            ] = self.buy_order_book.get_price_volume()
            # 检查卖单薄和买单薄中是否存在没有价格的订单
            if None not in sell_book or None not in buy_book:
                raise AssertionError
            if sell_book[None] != buy_book[None]:  # 判断两个薄中没有价格的订单的数量是否相等
                # 若不相等，则说明存在可交易的订单
                if sell_book[None] < buy_book[None]:
                    additional_required_orders = buy_book[None] - sell_book[None]
                    sell_book.pop(None)
                    return len(sell_book) >= additional_required_orders
                else:
                    additional_required_orders = sell_book[None] - buy_book[None]
                    buy_book.pop(None)
                    return len(buy_book) >= additional_required_orders
            else:  # 进一步比较卖单薄和买单薄中的最优价格
                sell_book.pop(None)
                buy_book.pop(None)
                if len(sell_book) == 0 or len(buy_book) == 0:
                    return False
                # 如果卖单薄的最优价格<=等于买单薄的最优价格，则返回True，否则False
                return min(list(cast(Dict[float, int], sell_book).keys())) <= max(
                    list(cast(Dict[float, int], buy_book).keys())
                )

    # 执行市场中的交易
    def _execution(self) -> List[ExecutionLog]:
        """execute for market. (Usually, only triggered by runner)

        Returns:
            List[:class:`pams.logs.base.ExecutionLog`]: execution logs.
        """
        if not self.remain_executable_orders():  # 检查是否有可执行的订单，若没有则直接返回空列表
            return []
        pending: List[Tuple[int, Order, Order]] = []  # 定义空列表pending以存储待执行的订单

        popped_buy_orders: List[Order] = []
        popped_sell_orders: List[Order] = []

        buy_order: Order = heapq.heappop(self.buy_order_book.priority_queue)
        popped_buy_orders.append(buy_order)
        sell_order: Order
        buy_order_volume_tmp: int = buy_order.volume  # 追踪当前买单剩余可交易量
        sell_order_volume_tmp: int = 0  # 追踪当前卖单剩余可交易量
        price: Optional[float] = None
        # ----定义一些变量用于后续的执行过程</>----

        # ----往pending中添加元素：成功撮合的买单和卖单，以及其成交的数量(volume)----
        while True:
            # 若存在未完成的买单或卖单，则抛出异常
            if buy_order_volume_tmp != 0 and sell_order_volume_tmp != 0:
                raise AssertionError
            # 如果买单的剩余数量为零，则取出下一个买单
            if buy_order_volume_tmp == 0:
                if len(self.buy_order_book.priority_queue) == 0:
                    break
                buy_order = heapq.heappop(self.buy_order_book.priority_queue)
                popped_buy_orders.append(buy_order)
                buy_order_volume_tmp = buy_order.volume
                if buy_order_volume_tmp == 0:
                    raise AssertionError
            if sell_order_volume_tmp == 0:
                if len(self.sell_order_book.priority_queue) == 0:
                    break
                sell_order = heapq.heappop(self.sell_order_book.priority_queue)
                popped_sell_orders.append(sell_order)
                sell_order_volume_tmp = sell_order.volume
                if sell_order_volume_tmp == 0:
                    raise AssertionError
            # 如果买单价格小于卖单价格，表示无法进行撮合交易，循环终止
            if (
                buy_order.price is not None
                and sell_order.price is not None
                and buy_order.price < sell_order.price
            ):
                break

            # 成交量：两个订单中可成交量的较小值
            volume = min(buy_order_volume_tmp, sell_order_volume_tmp)
            if volume == 0:
                raise AssertionError
            # 从每个订单的剩余量中减去这个可成交量，并确保不会导致任意一个订单的剩余量变成负数
            buy_order_volume_tmp -= volume
            sell_order_volume_tmp -= volume
            if buy_order_volume_tmp < 0:
                raise AssertionError
            if sell_order_volume_tmp < 0:
                raise AssertionError

            # ----确定成交价格----
            # 至少有一个订单的价格为None时
            if buy_order.price is None or sell_order.price is None:
                # 若买卖订单价格都为None，则直接添加到pending列表中
                if buy_order.price is None and sell_order.price is None:
                    pending.append((volume, buy_order, sell_order))
                # 若存在价格不为None的订单，则价格以其为准，并添加到pending列表中
                else:
                    price = (
                        buy_order.price
                        if buy_order.price is not None
                        else sell_order.price
                    )
                    pending.append((volume, buy_order, sell_order))
            # 若买卖订单的价格都不为None，则比较下单时间，价格以先下单者的价格为准
            else:
                if buy_order.placed_at == sell_order.placed_at:
                    if buy_order.order_id is None or sell_order.order_id is None:
                        raise AssertionError
                    if buy_order.order_id < sell_order.order_id:
                        price = buy_order.price
                    elif buy_order.order_id > sell_order.order_id:
                        price = sell_order.price
                    else:
                        raise AssertionError
                else:
                    price = (
                        buy_order.price
                        if cast(int, buy_order.placed_at)
                        < cast(int, sell_order.placed_at)
                        else sell_order.price
                    )
                # 将可成交量与买卖订单添加到pending列表中，以便后续进行交易的执行
                pending.append((volume, buy_order, sell_order))
            # ----确定成交价格</>----
        # ----往pending中添加元素：成功撮合的买单和卖单，以及其成交的数量(volume)</>----

        # 如果价格price为空值，则报错
        if price is None:
            raise AssertionError
        # TODO: faster impl
        self.buy_order_book.priority_queue = [
            *popped_buy_orders,
            *self.buy_order_book.priority_queue,
        ]
        self.sell_order_book.priority_queue = [
            *popped_sell_orders,
            *self.sell_order_book.priority_queue,
        ]
        heapq.heapify(self.buy_order_book.priority_queue)
        heapq.heapify(self.sell_order_book.priority_queue)
        # 遍历pending中的订单并逐一执行这些订单
        # 执行情况记录在logs列表中
        logs: List[ExecutionLog] = list(
            map(
                lambda x: self._execute_orders(
                    price=cast(float, price),
                    volume=x[0],
                    buy_order=x[1],
                    sell_order=x[2],
                ),
                pending,
            )
        )
        if self.remain_executable_orders():
            raise AssertionError
        # 若日志记录器(logger)存在，则将所有的订单执行日志一次性写入到日志文件中
        if self.logger is not None:
            self.logger.bulk_write(logs=cast(List[Log], logs))
        # 返回所有的订单执行日志
        return logs

    # 更改市场的基本价格（通过比例因子scale）
    def change_fundamental_price(self, scale: float) -> None:
        """change fundamental price.

        Args:
            scale (float): scale.

        Returns:
            None
        """
        # 获取当前时间和当前基本价格
        time: int = self.time
        current_fundamental: float = self.get_fundamental_price(time=time)
        # 计算新的基本价格
        new_fundamental: float = current_fundamental * scale
        # 将该价格存储到基本价格记录中
        self._fundamental_prices[time] = new_fundamental
        # 保存到市场对应的基本价格序列中
        self.simulator.fundamentals.prices[self.market_id][time] = new_fundamental
        # 更新基本价格生成器的状态（由于该更改将影响到以后生成的所有价格）
        self.simulator.fundamentals._generated_until = time
