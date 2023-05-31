import argparse
import random
from typing import Optional

from cfd_simple_agents import CFDSimpleAgent, CFDSimpleMarketMakerAgent
from cfd_fcn_agent import CFDMarketFCNAgent
from cfd_market_maker_agent import CFDMarketMakerAgent
from cfd_market import CFDMarket, CFDSimulator

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


def main() -> None:

    # 创建一个SequentialRunner对象
    runner = SequentialRunner(
        settings="/Users/chenglong/VS code project/pams_cfd/pams/samples/cfd/test_config.json",
        prng=random.Random(1),
        # MarketStepPrintLogger()：在控制台上输出市场状态的变化信息
        logger=MarketStepPrintLogger(),
    )
    runner.class_register(CFDSimpleAgent)
    runner.class_register(CFDMarketFCNAgent)
    runner.class_register(CFDSimpleMarketMakerAgent)
    runner.class_register(CFDMarketMakerAgent)
    runner.class_register(CFDMarket)
    # 调用runner对象的main()方法启动人工市场的模拟
    runner.main()


if __name__ == "__main__":  # 当该脚本被直接运行时，执行main()函数（如果该脚本被其他脚本导入时，__name__ 的值就不是 "__main__"，该条件不成立，main() 函数也不会被执行）
    main()
