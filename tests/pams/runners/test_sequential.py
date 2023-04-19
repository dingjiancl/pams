import os.path
import random
import time
from typing import Dict
from typing import Type
from typing import cast

import pytest

from pams.runners import Runner
from pams.runners import SequentialRunner
from tests.pams.runners.test_base import TestRunner


class TestSequentialRunner(TestRunner):
    runner_class: Type[Runner] = SequentialRunner
    default_setting: Dict = {
        "simulation": {
            "markets": ["Market"],
            "agents": ["FCNAgents"],
            "sessions": [
                {
                    "sessionName": 0,
                    "iterationSteps": 10,
                    "withOrderPlacement": True,
                    "withOrderExecution": True,
                    "withPrint": True,
                }
            ],
        },
        "Market": {"class": "Market", "tickSize": 0.00001, "marketPrice": 300.0},
        "FCNAgents": {
            "class": "FCNAgent",
            "numAgents": 10,
            "markets": ["Market"],
            "assetVolume": 50,
            "cashAmount": 10000,
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "meanReversionTime": {"uniform": [50, 100]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
        },
    }

    def test__(self) -> None:
        config = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ),
            "samples",
            "CI2002",
            "config.json",
        )
        runner = SequentialRunner(settings=config, prng=random.Random(42))
        runner._setup()
        runner.simulator._update_times_on_markets(markets=runner.simulator.markets)
        start_time = time.time()
        for _ in range(10000):
            _ = runner._collect_orders_from_normal_agents(
                session=runner.simulator.sessions[0]
            )
        end_time = time.time()
        time_per_step = (end_time - start_time) / 1000
        print("time/step", time_per_step)
        assert time_per_step < 0.003

    def test_generate_markets(self) -> None:
        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["market"])
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 1
        market = runner.simulator.markets[0]
        assert market.name == "Market"
        assert len(runner._pending_setups) == 1
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }
        assert runner.simulator.fundamentals.prices == {0: [300.0]}
        assert runner.simulator.fundamentals.drifts == {0: 0.0}
        assert runner.simulator.fundamentals.volatilities == {0: 0.0}

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "from": 0,
                "to": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Market": {"extends": "MarketBase"},
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 1
        market = runner.simulator.markets[0]
        assert market.name == "Market"
        assert len(runner._pending_setups) == 1
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "numMarkets": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
            "Market": {"extends": "MarketBase"},
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 10
        market = runner.simulator.markets[0]
        assert market.name == "Market-0"
        assert list(map(lambda x: x.name, runner.simulator.markets)) == [
            f"Market-{i}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 10
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "MarketBase": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
                "prefix": "Test",
            },
            "Market": {"extends": "MarketBase", "from": 10, "to": 19},
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        runner._generate_markets(market_type_names=["Market"])
        assert len(runner.simulator.markets) == 10
        market = runner.simulator.markets[0]
        assert market.name == "Test10"
        assert list(map(lambda x: x.name, runner.simulator.markets)) == [
            f"Test{i+10}" for i in range(10)
        ]
        assert len(runner._pending_setups) == 10
        assert runner._pending_setups[0][0] == market.setup
        assert runner._pending_setups[0][1] == {
            "settings": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 400.0,
                "outstandingShares": 2000,
            }
        }

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "numMarkets": 10,
                "from": 0,
                "to": 10,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "from": 0,
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Agent",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "fundamentalPrice": 300.0,
                "outstandingShares": 2000,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {
                "class": "Market",
                "tickSize": 0.01,
                "marketPrice": 300.0,
                "outstandingShares": 2000,
                "fundamentalDrift": 0.1,
                "fundamentalVolatility": 0.2,
            },
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        runner._generate_markets(market_type_names=["Market"])
        assert runner.simulator.fundamentals.prices == {0: [300.0]}
        assert runner.simulator.fundamentals.drifts == {0: 0.1}
        assert runner.simulator.fundamentals.volatilities == {0: 0.2}

        setting = {
            "simulation": {"markets": ["Market"]},
            "Market": {"class": "Market", "tickSize": 0.01, "outstandingShares": 2000},
        }
        runner = cast(
            SequentialRunner,
            self.test__init__(
                setting_mode="dict", logger=None, simulator_class=None, setting=setting
            ),
        )
        with pytest.raises(ValueError):
            runner._generate_markets(market_type_names=["Market"])
