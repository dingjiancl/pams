import random
import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

from .logs import Logger
from .market import Market


# 指数市场
class IndexMarket(Market):
    """Index of market.

    This class inherits from the :class:`pams.market.Market` class.
    """

    def __init__(
        self,
        market_id: int,
        prng: random.Random,
        simulator: "Simulator",  # type: ignore  # NOQA
        name: str,
        logger: Optional[Logger] = None,
    ):
        super().__init__(
            market_id=market_id,
            prng=prng,
            simulator=simulator,
            name=name,
            logger=logger,
        )
        self._components: List[Market] = []  # Market类的列表，用于保存IndexMarket的所有成分市场

    # 初始化IndexMarket对象时设置属性和配置
    def setup(self, settings: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore
        """setup market configuration from setting format.

        Args:
            settings (Dict[str, Any]): market configuration. Usually, automatically set from json config of simulator.
                                       This must include the parameter "markets".
                                       This should include the parameter "requires".

        Returns:
            None
        """
        """
        settings:包含所有必要参数的字典
        *args, **kwargs：处理任意数量的额外参数，将被自动封装成元组和字典
        *args：非关键字参数；**kwargs：关键字参数
        *args, **kwargs：使得函数在调用时，除了settings参数以外，还可以接受任意数量的其他参数
        """
        super(IndexMarket, self).setup(settings, *args, **kwargs)
        # 一些关于参数的异常与警告
        if "markets" not in settings:
            raise ValueError("markets is required for index markets as components")
        if "requires" in settings:
            warnings.warn("requires in index market settings is no longer required")
        # 对于settings["markets"]中的每个市场名称，从字典中获取对应的市场对象，添加到_components列表中
        for market_name in settings["markets"]:
            market: Market = self.simulator.name2market[market_name]
            self._add_market(market=market)

    # 将给定的市场添加到_components列表中
    def _add_market(self, market: Market) -> None:
        """add market. (Internal method)

        Args:
            market (:class:`pams.market.Market`): market.

        Returns:
            None
        """
        if market in self._components:  # 市场已经在列表中
            raise ValueError("market is already registered as components")
        if market.outstanding_shares is None:  # 市场的流通股数为None
            raise AssertionError(
                "outstandingShares is required in component market setting"
            )
        self._components.append(market)

    # 将给定列表中的所有市场添加到_components列表中
    def _add_markets(self, markets: List[Market]) -> None:
        """add markets. (Internal method)

        Args:
            markets (List[:class:`pams.market.Market`]): list of market.

        Returns:
            None
        """
        for market in markets:
            self._add_market(market=market)

    # 计算指数市场的基本价格
    def compute_fundamental_index(self, time: Optional[int] = None) -> float:
        """compute fundamental index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: fundamental index.

        .. note::
            In an index market, there are two types of fundamental values:
             1. fundamental value set from outside such as runner (:func:`pams.index_market.IndexMarket.compute_fundamental_index`)
             2. fundamental value calculated from components' fundamental value
                (:func:`pams.index_market.IndexMarket.get_fundamental_index`,
                 :func:`pams.index_market.IndexMarket.get_fundamental_price`)
            In usual usage, those become the same. But, in some special usage, it could be differ.
            This method return 1st one.
        """
        if time is None:
            time = self.get_time()
        total_value: float = 0
        total_shares: int = 0
        # 对_components中的所有市场进行迭代
        for market in self._components:
            # 对每个市场：基本价格×流通股数，结果累加到总值中
            outstanding_shares = cast(int, market.outstanding_shares)
            total_value += market.get_fundamental_price(time=time) * outstanding_shares
            total_shares += cast(int, outstanding_shares)
        # 指数市场的基本价格：总值/总流通股数
        return total_value / total_shares

    # 计算指数市场的市场价格
    def compute_market_index(self, time: Optional[int] = None) -> float:
        """compute market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
        if time is None:
            time = self.get_time()
        total_value: float = 0
        total_shares: int = 0
        for market in self._components:
            # 对每个市场：市场价格×流通股数，结果累加到总值中
            outstanding_shares = cast(int, market.outstanding_shares)
            total_value += market.get_market_price(time=time) * outstanding_shares
            total_shares += outstanding_shares
        # 指数市场的市场价格：总值/总流通股数
        return total_value / total_shares

    # 返回_components列表中的所有市场
    def get_components(self) -> List[Market]:
        """get components.

        Returns:
            List[:class:`pams.market.Market`]: list of components.
        """
        return self._components

    # 检查所有成分市场是否都处于运行状态
    def is_all_markets_running(self) -> bool:
        """get whether all markets is running or not.

        Returns:
            bool: whether all markets is running or not.
        """
        return sum(map(lambda x: not x.is_running, self._components)) == 0

    # 返回当前指数市场的基本价格
    def get_fundamental_index(self, time: Optional[int] = None) -> float:
        """get fundamental index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: fundamental index.

        .. note::
            In an index market, there are two types of fundamental values:
             1. fundamental value set from outside such as runner (:func:`pams.index_market.IndexMarket.compute_fundamental_index`)
             2. fundamental value calculated from components' fundamental value
                (:func:`pams.index_market.IndexMarket.get_fundamental_index`,
                 :func:`pams.index_market.IndexMarket.get_fundamental_price`)
            In usual usage, those become the same. But, in some special usage, it could be differ.
            This method return 2nd one.
        """
        return self.get_fundamental_price(time=time)

    # 获取当前指数市场的市场价格
    def get_market_index(self, time: Optional[int] = None) -> float:
        """get computed market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
        return self.compute_market_index(time=time)

    # 获取当前指数市场的市场价格（额外定义：增加灵活性）
    def get_index(self, time: Optional[int] = None) -> float:
        """get market index.

        Args:
            time (int, Optional): time step.

        Returns:
            float: market index.
        """
        return self.get_market_index(time=time)
