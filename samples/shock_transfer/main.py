import argparse
import random
from typing import Optional

from pams.logs.market_step_loggers import MarketStepPrintLogger
from pams.runners.sequential import SequentialRunner


def main() -> None:
    parser = argparse.ArgumentParser()  # 使用标准库中的argparse模块
    # --config或-c，指定了一个必需的参数type=str，用于指定一个config.json配置文件
    # 给解析器添加命令行参数选项
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="config.json file"
    )
    # --seed或-s，指定了一个可选的参数type=int，用于指定一个随机种子，默认值为None。
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="simulation random seed"
    )
    # 解析并获取config和seed
    args = parser.parse_args()
    config: str = args.config
    seed: Optional[int] = args.seed

    # 创建一个SequentialRunner对象
    runner = SequentialRunner(
        settings=config,
        prng=random.Random(seed) if seed is not None else None,
        # MarketStepPrintLogger()：在控制台上输出市场状态的变化信息
        logger=MarketStepPrintLogger(),
    )
    # 调用runner对象的main()方法启动人工市场的模拟
    runner.main()


if __name__ == "__main__":  # 当该脚本被直接运行时，执行main()函数（如果该脚本被其他脚本导入时，__name__ 的值就不是 "__main__"，该条件不成立，main() 函数也不会被执行）
    main()
