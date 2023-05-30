import random
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from scipy.linalg import cholesky

'''
用于生成和管理多个市场的基本面（fundamental）数据
指定每个市场的初始价格、漂移（drift）、波动率（volatility）以及它们之间的相关性
生成未来价格的模拟数据
'''


class Fundamentals:
    """Fundamental generator for simulator."""

    def __init__(self, prng: random.Random) -> None:
        """initialize.

        Args:
            prng (random.Random): pseudo random number generator for cholesky.

        Returns:
            None
        """
        self._prng = prng  # 用来生成随机数的Python内置random.Random对象
        #  创建的随机数生成器（种子从self._prng获得）
        self._np_prng: np.random.Generator = np.random.default_rng(
            self._prng.randint(0, 2 ** 31)
        )
        self.correlation: Dict[Tuple[int, int], float] = {}  # 市场之间的相关性
        self.drifts: Dict[int, float] = {}  # 每个市场的漂移
        self.volatilities: Dict[int, float] = {}  # 每个市场的波动率
        self.prices: Dict[int, List[float]] = {}  # 每个市场的历史价格数据
        self.market_ids: List[int] = []  # 已经注册的市场的ID
        self.initials: Dict[int, float] = {}  # 每个市场的初始价格
        self.start_at: Dict[int, int] = {}  # 每个市场的历史价格数据开始的时间
        self._generated_until: int = 0  # 目前已经生成的时间点，用于记录生成价格序列的进度
        self._generate_chunk_size = 100  # 一次生成价格序列的长度

    # 向Fundamentals中添加一个市场
    def add_market(
            self,
            market_id: int,
            initial: float,
            drift: float,
            volatility: float,
            start_at: int = 0,
    ) -> None:
        """add a market whose fundamental prices are generated in this class.

        Args:
            market_id (int): market ID to add.
            initial (float): initial value.
            drift (float): drifts.
            volatility (float): volatility.
            start_at (int): time step to start simulation (default 0).

        Returns:
            None
        """
        if market_id in self.market_ids:
            raise ValueError(f"market {market_id} is already registered")
        if volatility < 0.0:
            raise ValueError("volatility must be non-negative")
        if initial <= 0.0:
            raise ValueError("initial value must be positive")
        self.market_ids.append(market_id)
        self.drifts[market_id] = drift
        self.volatilities[market_id] = volatility
        self.initials[market_id] = initial
        self.start_at[market_id] = start_at
        self.prices[market_id] = [initial for _ in range(start_at + 1)]
        self._generated_until = min(start_at, self._generated_until)

    # 从Fundamentals中移除一个市场
    def remove_market(self, market_id: int) -> None:
        """remove a market from the list of markets whose fundamental prices are generated in this class.

        Args:
            market_id (int): market ID to remove.

        Returns:
            None
        """
        self.market_ids.remove(market_id)
        self.drifts.pop(market_id)
        self.volatilities.pop(market_id)
        self.initials.pop(market_id)
        self.start_at.pop(market_id)
        self.prices.pop(market_id)

    # 更改指定市场的波动率
    def change_volatility(
            self, market_id: int, volatility: float, time: int = 0
    ) -> None:
        """change volatility.

        Args:
            market_id (int): market ID.
            volatility (float): volatility.
            time (int): time step to apply the change(default 0).

        Returns:
            None
        """
        if volatility < 0.0:
            raise ValueError("volatility must be non-negative")
        self.volatilities[market_id] = volatility
        self._generated_until = time

    # 更改指定市场的漂移
    def change_drift(self, market_id: int, drift: float, time: int = 0) -> None:
        """change drift.

        Args:
            market_id (int): market ID.
            drift (float): drift.
            time (int): time step to apply the change (default 0).

        Returns:
            None
        """
        self.drifts[market_id] = drift
        self._generated_until = time

    # 设置两个市场之间的相关性
    def set_correlation(
            self, market_id1: int, market_id2: int, corr: float, time: int = 0
    ) -> None:
        """set correlation between fundamental prices of markets.

        Args:
            market_id1 (int): one of the market IDs to set correlation.
            market_id2 (int): the other of the market IDs to set correlation.
            corr (float): correlation.
            time (int): time step to apply the correlation (default 0).

        Returns:
            None
        """
        if not (-1.0 < corr < 1.0):
            raise ValueError("corr must be between 0.0 and 1.0")
        if market_id1 == market_id2:
            raise ValueError("market_id1 and market_id2 must be different")
        if (market_id2, market_id1) in self.correlation:
            self.correlation[(market_id2, market_id1)] = corr
        else:
            self.correlation[(market_id1, market_id2)] = corr
        self._generated_until = time

    # 移除两个市场之间的相关性
    def remove_correlation(
            self, market_id1: int, market_id2: int, time: int = 0
    ) -> None:
        """remove correlation.

        Args:
            market_id1 (int): one of the market IDs to remove correlation.
            market_id2 (int): the other of the market IDs to remove correlation.
            time (int): time step to apply the correlation (default 0).

        Returns:
            None
        """
        if market_id1 == market_id2:
            raise ValueError("market_id1 and market_id2 must be different")
        if (market_id2, market_id1) in self.correlation:
            self.correlation.pop((market_id2, market_id1))
        else:
            self.correlation.pop((market_id1, market_id2))
        self._generated_until = time

    # 为给定的市场id列表生成长度为length的随机对数收益率(log return)矩阵
    def _generate_log_return(
            self, generate_target_ids: List[int], length: int
            # generate_target_ids：需要生成随机对数收益率的市场id列表
            # length：需要生成的随机对数收益率矩阵的长度
    ) -> np.ndarray:
        """get log returns. (Internal method)

        Args:
            generate_target_ids (List[int]): target market ID list.
            length (int): return length.

        Returns:
            np.ndarray: log returns.
        """
        generate_target_ids_cholesky = list(  # 波动率不为0的市场
            filter(lambda x: self.volatilities[x] != 0.0, generate_target_ids)
        )
        generate_target_ids_other = list(  # 波动率为0的市场
            filter(lambda x: self.volatilities[x] == 0.0, generate_target_ids)
        )
        # ----对波动率不为0的市场----
        corr_matrix = np.eye(len(generate_target_ids_cholesky))  # 相关性矩阵
        for (id1, id2), corr in self.correlation.items():
            if id1 not in generate_target_ids_cholesky:
                continue
            if id2 not in generate_target_ids_cholesky:
                continue
            if id1 == id2:
                raise AssertionError
            corr_matrix[
                generate_target_ids_cholesky.index(id1),
                generate_target_ids_cholesky.index(id2),
            ] = corr
            corr_matrix[
                generate_target_ids_cholesky.index(id2),
                generate_target_ids_cholesky.index(id1),
            ] = corr
        vol = np.asarray([self.volatilities[x] for x in generate_target_ids_cholesky])
        cov_matrix = vol * corr_matrix * vol.reshape(-1, 1)  # 协方差矩阵
        try:
            cholesky_matrix = cholesky(cov_matrix, lower=True)  # Cholesky矩阵，利用Cholesky分解求出
        except Exception as e:
            print(
                "Error happened when calculating cholesky matrix for fundamental calculation."
                "This possibly means that fundamental correlations have a invalid circle correlation."
                "Please consider delete a circle correlation."
            )
            raise e

        dw_cholesky = self._np_prng.standard_normal(  # 符合Cholesky分布的随机矩阵（从标准正态分布中生成）
            size=(len(generate_target_ids_cholesky), length)
        )
        drifts_cholesky = np.asarray(
            [self.drifts[x] for x in generate_target_ids_cholesky]
        )
        result_cholesky = np.dot(  # 随机对数收益率矩阵
            cholesky_matrix, dw_cholesky
        ) + drifts_cholesky.T.reshape(-1, 1)
        # ----对波动率不为0的市场</>----

        # ----对波动率为0的市场----
        drifts_others = np.asarray(
            [[self.drifts[x] for _ in range(length)] for x in generate_target_ids_other]
        )  # 用每个市场的漂移来生成对应长度的随机对数收益率矩阵
        # ----对波动率为0的市场</>----

        return np.stack(  # 拼接对应的随机对数收益率矩阵，并返回
            [
                result_cholesky[generate_target_ids_cholesky.index(x)]
                if x in generate_target_ids_cholesky
                else drifts_others[generate_target_ids_other.index(x)]
                for x in generate_target_ids
            ]
        )

    # 生成接下来一段时间的市场价格（生成下一批价格序列）
    def _generate_next(self) -> None:
        """execute to next step. (Internal method)
        This method is called by :func:`pams.Fundamentals.get_fundamental_price` or :func:`pams.Fundamentals.get_fundamental_prices`.
        """
        setting_change_points: List[int] = [  # 参数改变点的时间列表（所有时间点均大于当前_generated_until的值）
            x for x in self.start_at.values() if x > self._generated_until
        ]

        # 计算length：当前价格序列需要生成的价格个数
        if len(setting_change_points) == 0:
            length = self._generate_chunk_size
        else:
            length = min(setting_change_points) - self._generated_until

        next_until = self._generated_until + length  # 下一个价格序列的结束时间
        target_market_ids: List[int] = [  # 需要生成价格序列的市场列表（开始时间早于next_until的市场）
            key for key, value in self.start_at.items() if value < next_until
        ]
        log_return = self._generate_log_return(  # 生成对应市场的价格对数收益率序列
            generate_target_ids=target_market_ids, length=length
        )
        current_prices = np.asarray(  # 当前价格序列
            [self.prices[x][self._generated_until] for x in target_market_ids]
        )

        # 计算新的价格序列并更新属性（根据当前价格序列和对数收益率序列计算）
        prices = current_prices.T.reshape(-1, 1) * np.exp(
            np.cumsum(log_return, axis=-1)
        )
        for market_id, price_seq in zip(target_market_ids, prices):
            self.prices[market_id] = (  # 更新对应市场的价格序列
                    self.prices[market_id][: self._generated_until + 1] + price_seq.tolist()
            )

        self._generated_until += length  # 将_generated_until更新为下一个价格序列的结束时间

    # 获取指定市场在给定时间的基本面价格
    def get_fundamental_price(self, market_id: int, time: int) -> float:
        """get a fundamental price.

        Args:
            market_id (int): market ID.
            time (int): time step to get the price.

        Returns:
            float: fundamental price at the specified time step.
        """
        while time >= self._generated_until:
            self._generate_next()
        return self.prices[market_id][time]

    # 获取指定市场在给定多个时间的基本面价格
    def get_fundamental_prices(
            self, market_id: int, times: Iterable[int]
    ) -> List[float]:
        """get some fundamental prices.

        Args:
            market_id (int): market ID.
            times (Iterable[int]): time steps to get the price.

        Returns:
            List[float]: fundamental prices in specified range of time steps.
        """
        while max([x for x in times]) >= self._generated_until:
            self._generate_next()
        return [self.prices[market_id][x] for x in times]
