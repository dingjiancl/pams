import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from ..logs.base import Logger
from ..market import Market
from ..order import LIMIT_ORDER
from ..order import Cancel
from ..order import Order
from ..utils.json_random import JsonRandom
from .base import Agent

MARGIN_FIXED = 0
MARGIN_NORMAL = 1


class FCNAgent(Agent):
    """FCN (Fundamental, Chartist, Noise) Agent class

    This class inherits from the :class:`pams.agents.Agent` class.

    An order decision mechanism proposed in Chiarella & Iori (2004).
    It employs two simple margin-based random tradings. Given an expected future price p, submit an order of price

    - "fixed" : :math:`p (1 ± k)` where :math:`0 \leq k \leq 1`
    - "normal" : :math:`p + N(0, k)` where :math:`k > 0`

    References:
        - Chiarella, C., & Iori, G. (2002). A simulation analysis of the microstructure of double auction markets.
          Quantitative Finance, 2(5), 346–353. https://doi.org/10.1088/1469-7688/2/5/303
    """  # NOQA

    fundamental_weight: float
    chart_weight: float
    margin_type: int
    mean_reversion_time: int
    noise_scale: float
    noise_weight: float
    order_margin: float
    time_window_size: int

    # 采用父类Agent的构造方法
    def __init__(
        self,
        agent_id: int,
        prng: random.Random,  # 随机数生成器
        simulator: "Simulator",  # type: ignore  # 市场模拟器
        name: str,  # 该代理人的名称
        logger: Optional[Logger] = None,  # 可选的日志记录器
    ):
        super().__init__(agent_id, prng, simulator, name, logger)
        self.is_chart_following = True

    # 判断给定的x是否是一个合法的（非NaN且有限的）浮点数
    def is_finite(self, x: float) -> bool:
        """determine if it is a valid value.

        Args:
            x (float): value.

        Return:
            bool: whether or not it is a valid (not NaN, finite) value.
        """
        return not math.isnan(x) and not math.isinf(x)

    # 对FCNAgent进行初始化设置
    def setup(
        self,
        settings: Dict[str, Any],  # 反映agent的设置
        accessible_markets_ids: List[int],  # 可访问的市场
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """agent setup.  Usually be called from simulator/runner automatically.

        Args:
            settings (Dict[str, Any]): agent configuration. This can include the parameters "fundamentalWeight", "chartWeight",
                                       "noiseWeight", "noiseScale", "timeWindowSize", "orderMargin", "marginType",
                                       and "meanReversionTime".
            accessible_markets_ids (List[int]): list of market IDs.

        Returns:
            None
        """
        super().setup(settings=settings, accessible_markets_ids=accessible_markets_ids)  # 初始化代理人在可访问市场的资产数量

        # ----设置FCNAgent实例的属性（使用生成的随机数）----
        json_random: JsonRandom = JsonRandom(prng=self.prng)
        self.fundamental_weight = json_random.random(
            json_value=settings["fundamentalWeight"]
        )
        self.chart_weight = json_random.random(json_value=settings["chartWeight"])
        self.noise_weight = json_random.random(json_value=settings["noiseWeight"])
        self.noise_scale = json_random.random(json_value=settings["noiseScale"])
        self.time_window_size = int(
            json_random.random(json_value=settings["timeWindowSize"])  # 用于计算技术面的时间窗口大小
        )
        self.order_margin = json_random.random(json_value=settings["orderMargin"])  # 限价单的价格浮动范围
        if settings.get("marginType") in [None, "fixed"]:  # marginType: 限价单的价格调整方式
            self.margin_type = MARGIN_FIXED
        elif settings.get("marginType") == "normal":
            self.margin_type = MARGIN_NORMAL
        else:
            raise ValueError(
                "marginType have to be normal or fixed (not specified is also allowed.)"
            )
        if "meanReversionTime" in settings:  # 基本面回归到平均值所需的时间
            self.mean_reversion_time = int(
                json_random.random(json_value=settings["meanReversionTime"])
            )
        else:
            self.mean_reversion_time = self.time_window_size
        # ----设置FCNAgent实例的属性（使用生成的随机数）</>----

    # 在所有可用市场中提交订单
    # 接收参数：所有可访问市场的列表
    def submit_orders(self, markets: List[Market]) -> List[Union[Order, Cancel]]:
        """submit orders based on FCN-based calculation.

        .. seealso::
            - :func:`pams.agents.Agent.submit_orders`
        """
        orders: List[Union[Order, Cancel]] = sum(
            [self.submit_orders_by_market(market=market) for market in markets], []
        )
        return orders

    # 在指定的市场中提交订单
    # 接收参数：一个market对象，表示要交易的市场
    def submit_orders_by_market(self, market: Market) -> List[Union[Order, Cancel]]:
        """submit orders by market (internal usage).

        Args:
            market (Market): market to order.

        Returns:
            List[Union[Order, Cancel]]: order list.
        """
        # 检查该市场是否可访问，若不可则直接返回一个空列表
        orders: List[Union[Order, Cancel]] = []
        if not self.is_market_accessible(market_id=market.market_id):
            return orders

        time: int = market.get_time()
        time_window_size: int = min(time, self.time_window_size)
        assert time_window_size >= 0
        assert self.fundamental_weight >= 0.0
        assert self.chart_weight >= 0.0
        assert self.noise_weight >= 0.0

        # ----计算预期未来价格----
        fundamental_scale: float = 1.0 / max(self.mean_reversion_time, 1)
        fundamental_log_return = fundamental_scale * math.log(
            market.get_fundamental_price() / market.get_market_price()
        )
        assert self.is_finite(fundamental_log_return)  # 判断是否是一个合法的（非NaN且有限的）浮点数

        chart_scale: float = 1.0 / max(time_window_size, 1)
        chart_mean_log_return = chart_scale * math.log(
            market.get_market_price() / market.get_market_price(time - time_window_size)
        )
        assert self.is_finite(chart_mean_log_return)

        noise_log_return: float = self.noise_scale * self.prng.gauss(mu=0.0, sigma=1.0)
        assert self.is_finite(noise_log_return)

        expected_log_return: float = (  # 预期对数收益率
            1.0 / (self.fundamental_weight + self.chart_weight + self.noise_weight)
        ) * (
            self.fundamental_weight * fundamental_log_return
            + self.chart_weight
            * chart_mean_log_return
            * (1 if self.is_chart_following else -1)
            + self.noise_weight * noise_log_return
        )
        assert self.is_finite(expected_log_return)

        # 预期未来价格
        expected_future_price: float = market.get_market_price() * math.exp(
            expected_log_return * self.time_window_size
        )
        assert self.is_finite(expected_future_price)
        # ----计算预期未来价格</>----

        # ----根据不同的self.margin_type计算订单价格，并将新建的订单添加到订单列表中，并返回----

        # ----如果使用的是MARGIN_FIXED（生成的订单的价格为定值）----
        if self.margin_type == MARGIN_FIXED:
            assert 0.0 <= self.order_margin <= 1.0  # 对self.order_margin进行范围限定

            order_volume: int = 1  # 订单数量设为1

            # 市场价格 < 预期未来价格时，下买单（限价单）
            # 价格为 预期未来价格 * (1 - self.order_margin)
            if expected_future_price > market.get_market_price():
                order_price = expected_future_price * (1 - self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            # 市场价格 > 预期未来价格时，下卖单（限价单）
            # 价格为 预期未来价格 * (1 + self.order_margin)
            if expected_future_price < market.get_market_price():
                order_price = expected_future_price * (1 + self.order_margin)
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
        # ----如果使用的是MARGIN_FIXED（生成的订单的价格为定值）</>----

        # ----如果使用的是MARGIN_NORMAL（生成的订单的价格为一个随机值）----
        if self.margin_type == MARGIN_NORMAL:
            assert self.order_margin >= 0.0  # 确保正态分布的sigma合法
            # 订单价格 = 期望未来价格 + 价格波动量
            order_price = (
                expected_future_price
                + self.prng.gauss(mu=0.0, sigma=1.0) * self.order_margin
            )
            order_volume = 1
            assert order_price >= 0.0  # 订单价格不可为负数
            assert order_volume > 0
            # 市场价格 < 预期未来价格时，下买单（限价单）
            if expected_future_price > market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=True,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
            # 市场价格 > 预期未来价格时，下卖单（限价单）
            if expected_future_price < market.get_market_price():
                orders.append(
                    Order(
                        agent_id=self.agent_id,
                        market_id=market.market_id,
                        is_buy=False,
                        kind=LIMIT_ORDER,
                        volume=order_volume,
                        price=order_price,
                        ttl=self.time_window_size,
                    )
                )
        # ----如果使用的是MARGIN_NORMAL（生成的订单的价格为一个随机值）</>----
        return orders
        # ----根据不同的self.margin_type计算订单价格，并将新建的订单添加到订单列表中，并返回</>----

    # 代理类的字符串表示形式
    def __repr__(self) -> str:
        """string representation of FCN agent class.

        Returns:
            str: string representation of this class.
        """
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} | id={self.agent_id}, rnd={self.prng}, "
            f"chart_weight={self.chart_weight}, fundamental_weight={self.fundamental_weight}, "
            f"noise_weight={self.noise_weight}, is_chart_following:{self.is_chart_following}, "
            f"margin_type={self.margin_type}, mean_reversion_time:{self.mean_reversion_time}, "
            f"noise_scale={self.noise_scale}, time_window_size={self.time_window_size}, "
            f"order_margin={'MARGIN_FIXED' if self.order_margin == MARGIN_FIXED else 'MARGIN_NORMAL'}"
        )
