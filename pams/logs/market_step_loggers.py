from typing import Dict
from typing import List

from .base import Logger
from .base import MarketStepEndLog


class MarketStepPrintLogger(Logger):
    """Logger of the market step class."""

    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """print the market log.

        Args:
            log (:class:`pams.logs.MarketStepEndLog`): matket log.

        Returns:
            None
        """
        """打印市场的日志信息
        包括session_id、市场时间、市场ID、市场名称、市场价格和基本价格"""
        print(
            f"{log.session.session_id} {log.market.get_time()} {log.market.market_id} {log.market.name} {log.market.get_market_price()} {log.market.get_fundamental_price()}"
        )


class MarketStepSaver(Logger):
    """Saver of the market step class."""

    def __init__(self) -> None:
        super().__init__()
        self.market_step_logs: List[Dict] = []

    # 将市场日志信息保存到market_step_logs列表中
    def process_market_step_end_log(self, log: MarketStepEndLog) -> None:
        """stack the market log.

        Args:
            log (:class:`pams.logs.MarketStepEndLog`): market log.

        Returns:
            None
        """
        self.market_step_logs.append(
            {
                "session_id": log.session.session_id,
                "market_time": log.market.get_time(),
                "market_id": log.market.market_id,
                "market_name": log.market.name,
                "market_price": log.market.get_market_price(),
                "fundamental_price": log.market.get_fundamental_price(),
            }
        )
