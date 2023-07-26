import os
import random
from io import TextIOWrapper
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from ..agents.base import Agent
from ..events import EventABC
from ..events import EventHook
from ..index_market import IndexMarket
from ..logs.base import CancelLog
from ..logs.base import ExecutionLog
from ..logs.base import Log
from ..logs.base import Logger
from ..logs.base import MarketStepBeginLog
from ..logs.base import MarketStepEndLog
from ..logs.base import OrderLog
from ..logs.base import SessionBeginLog
from ..logs.base import SessionEndLog
from ..logs.base import SimulationBeginLog
from ..logs.base import SimulationEndLog
from ..market import Market
from ..order import Cancel
from ..order import Order
from ..session import Session
from ..simulator import Simulator
from ..utils.class_finder import find_class
from ..utils.json_extends import json_extends
from .base import Runner


class SequentialRunner(Runner):
    """Sequential runner class."""

    def __init__(
        self,
        settings: Union[Dict, TextIOWrapper, os.PathLike, str],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator,
    ):
        """initialize.

        Args:
            settings (Union[Dict, TextIOWrapper, os.PathLike, str]): runner configuration.
            prng (random.Random, Optional): pseudo random number generator for this runner.
            logger (Logger, Optional): logger instance.
            simulator_class (Type[Simulator]): type of simulator.

        Returns:
            None
        """
        super().__init__(settings, prng, logger, simulator_class)
        self._pending_setups: List[Tuple[Callable, Dict]] = []  # 创建空列表：元素为 元组[可调用对象，字典]

    # 1、按照给定的市场类型名称生成市场实例
    # 2、将市场实例添加到模拟器中
    def _generate_markets(self, market_type_names: List[str]) -> None:
        """generate markets. (Internal method)

        Args:
            market_type_names (List[str]): name list of market type.

        Returns:
            None
        """
        i_market = 0
        # 获取每个市场类型对应的配置信息（从JSON文件）
        for name in market_type_names:
            if name not in self.settings:
                raise ValueError(f"{name} setting is missing in config")
            # ----如果配置信息中存在"extends"字段，则递归地合并JSON文件中的指定字段----
            # （对market_settings进行扩展）（并排除指定字段）
            market_settings: Dict = self.settings[name]
            market_settings = json_extends(
                whole_json=self.settings,
                parent_name=name,
                target_json=market_settings,
                excludes_fields=["from", "to"],
            )
            # ----如果配置信息中存在"extends"字段，则递归地合并JSON文件中的指定字段</>----
            # TODO: warn "from" and "to" is included in parent setting and not set to this setting.
            n_markets = 1
            id_from = 0
            id_to = 0

            # ----关于市场数量（numMarkets or from, to）----
            if "numMarkets" in market_settings:
                n_markets = int(market_settings["numMarkets"])
                id_to = n_markets - 1
            if "from" in market_settings or "to" in market_settings:
                if "from" not in market_settings or "to" not in market_settings:
                    raise ValueError(
                        f"both {name}.from and {name}.to are required in json file if you use"
                    )
                if "numMarkets" in market_settings:
                    raise ValueError(
                        f"{name}.numMarkets and ({name}.from or {name}.to) cannot be used at the same time"
                    )
                n_markets = int(market_settings["to"]) - int(market_settings["from"])
                id_from = int(market_settings["from"])
                id_to = int(market_settings["to"])
            if "numMarkets" in market_settings:
                del market_settings["numMarkets"]
            if "from" in market_settings:
                del market_settings["from"]
            if "to" in market_settings:
                del market_settings["to"]
            # ----关于市场数量（numMarkets or from, to）</>----
            # ----根据给出的prefix及市场数量，为每个市场实例定义一个唯一的市场名称----
            prefix: str
            if "prefix" in market_settings:
                prefix = market_settings["prefix"]
                del market_settings["prefix"]
            else:
                prefix = name + ("-" if n_markets > 1 else "")
            # ----根据给出的prefix及市场数量，为每个市场实例定义一个唯一的市场名称</>----

            # ----获取市场类（"class"字段）并实例化该类，生成市场实例----
            # 需要为当前市场定义"class"
            if "class" not in market_settings:
                raise ValueError(f"class is not defined for {name}")

            # 返回与给定"class"匹配的类，作为市场的类型
            market_class: Type[Market] = find_class(
                name=market_settings["class"],
                optional_class_list=self.registered_classes,
            )
            if not issubclass(market_class, Market):  # 若不是Market的子类则抛出异常
                raise ValueError(
                    f"market class for {name} does not inherit Market class"
                )
            # ----获取市场类（"class"字段）并实例化该类，生成市场实例</>----

            # ----按JSON文件中给出的市场设置赋值----
            # 基础价格、漂移、波动率
            if "fundamentalPrice" in market_settings:
                fundamental_price = float(market_settings["fundamentalPrice"])
            elif "marketPrice" in market_settings:
                fundamental_price = float(market_settings["marketPrice"])
            else:
                raise ValueError(
                    f"fundamentalPrice or marketPrice is required for {name}"
                )
            fundamental_drift: float = 0.0
            if "fundamentalDrift" in market_settings:
                fundamental_drift = float(market_settings["fundamentalDrift"])
            fundamental_volatility: float = 0.0
            if "fundamentalVolatility" in market_settings:
                fundamental_volatility = float(market_settings["fundamentalVolatility"])
            # ----按JSON文件中给出的市场设置赋值</>----

            for i in range(id_from, id_to + 1):
                # 生成市场
                market = market_class(
                    market_id=i_market,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if n_markets != 1 else ""),
                )
                i_market += 1
                # 将市场实例添加到Simulator中
                self.simulator._add_market(market=market, group_name=name)
                # 将市场实例添加到fundamentals中
                if not isinstance(market, IndexMarket):
                    self.simulator.fundamentals.add_market(
                        market_id=market.market_id,
                        initial=fundamental_price,
                        drift=fundamental_drift,
                        volatility=fundamental_volatility,
                    )
                # 将市场实例的setup方法添加到待处理设置列表中，等待后续调用
                self._pending_setups.append(
                    (market.setup, {"settings": market_settings})
                )

    # 根据配置文件生成agents并添加到simulator中
    def _generate_agents(self, agent_type_names: List[str]) -> None:
        """generate agents. (Internal method)

        Args:
            agent_type_names (List[str]): name list of agent type.

        Returns:
            None
        """
        i_agent = 0  # 追踪生成的agent的数量
        # 获取每个agent类型对应的配置信息（从JSON文件）
        for name in agent_type_names:
            if name not in self.settings:
                raise ValueError(f"{name} setting is missing in config")
            agent_settings: Dict = self.settings[name]
            # 如果配置信息中存在"extends"字段，则递归地合并JSON文件中的指定字段
            agent_settings = json_extends(
                whole_json=self.settings,
                parent_name=name,
                target_json=agent_settings,
                excludes_fields=["from", "to"],
            )
            # TODO: warn "from" and "to" is included in parent setting and not set to this setting.
            n_agents = 1
            id_from = 0
            id_to = 0

            # ----读取agent的数量----
            if "numAgents" in agent_settings:
                n_agents = int(agent_settings["numAgents"])
                id_to = n_agents - 1
            if "from" in agent_settings or "to" in agent_settings:
                if "from" not in agent_settings or "to" not in agent_settings:
                    raise ValueError(
                        f"both {name}.from and {name}.to are required in json file if you use"
                    )
                if "numAgents" in agent_settings:
                    raise ValueError(
                        f"{name}.numMarkets and ({name}.from or {name}.to) cannot be used at the same time"
                    )
                n_agents = int(agent_settings["to"]) - int(agent_settings["from"])
                id_from = int(agent_settings["from"])
                id_to = int(agent_settings["to"])
            if "numAgents" in agent_settings:
                del agent_settings["numAgents"]
            if "from" in agent_settings:
                del agent_settings["from"]
            if "to" in agent_settings:
                del agent_settings["to"]
            # ----读取agent的数量</>----

            # ----根据给出的prefix及agent数量，为每个agent实例定义一个唯一的名称----
            prefix: str
            if "prefix" in agent_settings:
                prefix = agent_settings["prefix"]
                del agent_settings["prefix"]
            else:
                prefix = name + ("-" if n_agents > 1 else "")
            # ----根据给出的prefix及agent数量，为每个agent实例定义一个唯一的名称</>----

            # 根据给定的class名称查找匹配的类，作为agent的类型，并将其赋值给agent_class
            if "class" not in agent_settings:
                raise ValueError(f"class is not defined for {name}")
            agent_class: Type[Agent] = find_class(
                name=agent_settings["class"],
                optional_class_list=self.registered_classes,
            )
            if not issubclass(agent_class, Agent):
                raise ValueError(f"agent class for {name} does not inherit Agent class")

            # 如果agent_settings中没有指定markets，则抛出ValueError
            # 否则，将其赋值给accessible_market_names
            if "markets" not in agent_settings:
                raise ValueError(f"markets is required in {name}")
            accessible_market_names: List[str] = agent_settings["markets"]

            # 生成可访问市场的ID列表
            # 根据可访问的市场名称查找相应的市场对象，将市场ID添加到accessible_market_ids列表中
            accessible_market_ids: List[int] = sum(
                [
                    list(
                        map(
                            lambda m: m.market_id,
                            self.simulator.markets_group_name2market[x],
                        )
                    )
                    for x in accessible_market_names
                ],
                [],
            )

            # 生成agent
            for i in range(id_from, id_to + 1):
                agent = agent_class(
                    agent_id=i_agent,
                    prng=random.Random(self._prng.randint(0, 2**31)),
                    simulator=self.simulator,
                    logger=self.logger,
                    name=prefix + (str(i) if n_agents != 1 else ""),
                )
                i_agent += 1

                # 将agent添加到Simulator中
                self.simulator._add_agent(agent=agent, group_name=name)

                # 将代理人的设置和可访问市场的ID列表传递给待处理的代理人列表_pending_setups，以在后续调用中设置代理人
                self._pending_setups.append(
                    (
                        agent.setup,
                        {
                            "settings": agent_settings,
                            "accessible_markets_ids": accessible_market_ids,
                        },
                    )
                )

    # 设置市场基本面之间的相关性
    def _set_fundamental_correlation(self) -> None:
        """set fundamental correlation. (Internal method)"""
        if "fundamentalCorrelations" in self.settings["simulation"]:
            corr_settings: Dict = self.settings["simulation"]["fundamentalCorrelations"]
            for key, value in corr_settings.items():
                if key == "pairwise":
                    if (
                        not isinstance(value, list)
                        or sum([len(x) != 3 for x in value]) > 0
                    ):
                        raise ValueError(
                            "simulation.fundamentalCorrelations.pairwise has invalid format data"
                        )
                    for (market1_name, market2_name, corr) in value:
                        market1 = self.simulator.name2market[market1_name]
                        market2 = self.simulator.name2market[market2_name]
                        for m in [market1, market2]:
                            if (
                                self.simulator.fundamentals.volatilities[m.market_id]
                                == 0.0
                            ):
                                raise ValueError(
                                    f"For applying fundamental correlation fo {m.name}, "
                                    f"fundamentalVolatility for {m.name} is required"
                                )
                        # 设置两个市场之间的相关性
                        self.simulator.fundamentals.set_correlation(
                            market_id1=market1.market_id,
                            market_id2=market2.market_id,
                            corr=float(corr),
                        )
                else:
                    raise NotImplementedError(
                        f"{key} for simulation.fundamentalCorrelations is not supported"
                    )

    # 生成会话
    def _generate_sessions(self) -> None:
        """generate sessions. (Internal method)"""
        if "sessions" not in self.settings["simulation"]:
            raise ValueError("sessions is missing under 'simulation' config")
        # 从配置文件中获取会话（sessions）的配置信息（session_settings）
        session_settings: Dict = self.settings["simulation"]["sessions"]
        if not isinstance(session_settings, list):
            raise ValueError("simulation.sessions must be list[dict]")
        i_session = 0
        i_event = 0
        session_start_time: int = 0

        # 通过for循环逐个生成会话
        for session_setting in session_settings:
            if "sessionName" not in session_setting:
                raise ValueError(
                    "for each element in simulation.sessions must have sessionName"
                )
            session = Session(
                session_id=i_session,
                prng=random.Random(self._prng.randint(0, 2**31)),
                session_start_time=session_start_time,
                simulator=self.simulator,
                name=str(session_setting["sessionName"]),
                logger=self.logger,
            )
            i_session += 1

            # 增加会话的迭代步骤数（iterationSteps），并将会话添加到模拟器对象中
            if "iterationSteps" not in session_setting:
                raise ValueError(
                    "iterationSteps is required in each element of simulation.sessions"
                )
            session_start_time += session_setting["iterationSteps"]
            self.simulator._add_session(session=session)

            # 将会话的设置和对应事件的设置分别添加到_pending_setups属性中，以便稍后调用
            self._pending_setups.append((session.setup, {"settings": session_setting}))

            # 如果会话的设置中包含了事件（即 "events" 键），则迭代处理每个事件
            if "events" in session_setting:
                event_names: List[str] = session_setting["events"]
                for event_name in event_names:

                    # 获取该事件的设置 -> event_setting
                    event_setting = self.settings[event_name]
                    event_setting = json_extends(
                        whole_json=self.settings,
                        parent_name=event_name,
                        target_json=event_setting,
                        excludes_fields=["numMarkets", "from", "to", "prefix"],
                    )

                    # 获取事件类 -> event_class
                    if "class" not in event_setting:
                        raise ValueError(f"class is required in {event_name}")
                    event_class_name = event_setting["class"]
                    event_class: Type[EventABC] = find_class(
                        name=event_class_name,
                        optional_class_list=self.registered_classes,
                    )

                    # 初始化一个事件实例event（根据event_class）
                    event = event_class(
                        event_id=i_event,
                        prng=random.Random(self._prng.randint(0, 2**31)),
                        session=session,
                        simulator=self.simulator,
                        name=event_name,
                    )
                    i_event += 1

                    # 将设置添加到_pending_setups属性中，以便稍后调用
                    self._pending_setups.append(
                        (event.setup, {"settings": event_setting})
                    )

                    # 创建并注册事件钩子（EventHook）至Simulator
                    def event_hook_setup(_event: EventABC):
                        event_hooks: List[EventHook] = _event.hook_registration()
                        for event_hook in event_hooks:
                            self.simulator._add_event(event_hook)

                    self._pending_setups.append((event_hook_setup, {"_event": event}))

    # 根据配置生成市场、代理和会话实例
    def _setup(self) -> None:
        """runner setup. (Internal method)"""
        # 检查配置文件中是否包含必要字段
        if "simulation" not in self.settings:
            raise ValueError("simulation is required in json file")

        if "markets" not in self.settings["simulation"]:
            raise ValueError("simulation.markets is required in json file")

        # 生成市场
        # （获取配置文件中的市场类型列表，并根据类型名称生成相应的市场实例）
        market_type_names: List[str] = self.settings["simulation"]["markets"]
        if (
            not isinstance(market_type_names, list)
            or sum([not isinstance(m, str) for m in market_type_names]) > 0
        ):
            raise ValueError("simulation.markets in json file have to be list[str]")
        self._generate_markets(market_type_names=market_type_names)

        # 设置市场间基本面的相关性
        self._set_fundamental_correlation()

        # 生成代理
        # 获取配置文件中的代理类型列表，并根据类型名称生成相应的代理实例
        if "agents" not in self.settings["simulation"]:
            raise ValueError("agents.markets is required in json file")
        agent_type_names: List[str] = self.settings["simulation"]["agents"]
        if (
            not isinstance(agent_type_names, list)
            or sum([not isinstance(m, str) for m in agent_type_names]) > 0
        ):
            raise ValueError("simulation.agents in json file have to be list[str]")
        self._generate_agents(agent_type_names=agent_type_names)

        # 获取配置文件中的会话配置列表，生成相应的会话实例
        if "sessions" not in self.settings["simulation"]:
            raise ValueError("simulation.sessions is required in json file")
        session_settings: List[Dict[str, Any]] = self.settings["simulation"]["sessions"]
        if (
            not isinstance(session_settings, list)
            or sum([not isinstance(m, dict) for m in session_settings]) > 0
        ):
            raise ValueError("simulation.sessions in json file have to be List[Dict]")
        self._generate_sessions()

        # 执行_pending_setups列表中保存的函数（不需要结果，只是想运行，故用“_”忽略返回值）
        _ = [func(**kwargs) for func, kwargs in self._pending_setups]

    # 收集所有normal agent的订单，返回所有agent的订单列表
    def _collect_orders_from_normal_agents(
        self, session: Session  # 接受参数：session，表示当前交易会话
    ) -> List[List[Union[Order, Cancel]]]:
        """collect orders from normal_agents. (Internal method)
        orders are corrected until the total number of orders reaches max_normal_orders

        Args:
            session (Session): session.

        Returns:
            List[List[Union[Order, Cancel]]]: orders lists.
        """
        # 获取所有agent并打乱顺序
        agents = self.simulator.normal_frequency_agents
        agents = self._prng.sample(agents, len(agents))

        n_orders = 0  # 当前已经收集的订单数量
        all_orders: List[List[Union[Order, Cancel]]] = []  # 收集到的所有订单的列表
        # 遍历所有agent，获取其订单，并加入到all_orders（返回值）
        for agent in agents:
            # 若已经收集的订单数量大于当前交易会话的最大订单数，则退出循环
            if n_orders >= session.max_normal_orders:
                break
            # 当前agent在所有市场中提交订单
            orders = agent.submit_orders(markets=self.simulator.markets)
            if len(orders) > 0:
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                if sum([order.agent_id != agent.agent_id for order in orders]) > 0:
                    raise ValueError(
                        "spoofing order is not allowed. please check agent_id in order"
                    )
                all_orders.append(orders)
                # TODO: currently the original impl is used
                # n_orders += len(orders)
                n_orders += 1
        # 返回所有agent的订单的列表
        return all_orders

    # 处理订单
    def _handle_orders(
        self, session: Session, local_orders: List[List[Union[Order, Cancel]]]
    ) -> List[List[Union[Order, Cancel]]]:
        """handle orders. (Internal method)
        processing local orders and correct and process the orders from high frequency agents.

        Args:
            session (Session): session.
            local_orders (List[List[Union[Order, Cancel]]]): local orders.

        Returns:
            List[List[Union[Order, Cancel]]]: order lists.
        """
        # 将本地订单按随机顺序排序
        sequential_orders = self._prng.sample(local_orders, len(local_orders))
        all_orders: List[List[Union[Order, Cancel]]] = [*sequential_orders]

        # 循环处理所有订单
        for orders in sequential_orders:
            for order in orders:
                # 检查当前会话是否接受该订单
                if not session.with_order_placement:
                    raise AssertionError("currently order is not accepted")
                # 当前订单对应的市场
                market: Market = self.simulator.id2market[order.market_id]
                # ----对本地订单进行处理----
                # 订单为Order类时
                if isinstance(order, Order):
                    self.simulator._trigger_event_before_order(order=order)  # 在提交订单前触发事件（触发"order_before"事件）
                    log: OrderLog = market._add_order(order=order)  # 将订单添加到市场中，并记录在OrderLog中
                    agent: Agent = self.simulator.id2agent[order.agent_id]  # 当前订单对应的agent
                    # 代理向市场提交订单成功时回调（将提交的订单的信息记录在agent中，从而可以再agent中查看其提交订单的记录）
                    agent.submitted_order(log=log)
                    self.simulator._trigger_event_after_order(order_log=log)  # 在订单执行完毕后触发事件（触发"order_after"事件）
                # 订单为Cancel类时
                elif isinstance(order, Cancel):
                    self.simulator._trigger_event_before_cancel(cancel=order)  # 触发取消订单前的事件
                    log_: CancelLog = market._cancel_order(cancel=order)  # 从市场中取消订单，并记录在CancelLog中
                    agent = self.simulator.id2agent[order.order.agent_id]
                    agent.canceled_order(log=log_)
                    self.simulator._trigger_event_after_cancel(cancel_log=log_) # 触发取消订单后的事件
                else:
                    raise NotImplementedError

                # 若当前会话允许订单执行
                if session.with_order_execution:
                    # 执行市场中的订单
                    logs: List[ExecutionLog] = market._execution()
                    # 更新代理的资产和现金金额
                    self.simulator._update_agents_for_execution(execution_logs=logs)
                    for execution_log in logs:
                        # 找到卖方和卖方agent，并将执行的订单的信息记录在agent中
                        agent = self.simulator.id2agent[execution_log.buy_agent_id]
                        agent.executed_order(log=execution_log)
                        agent = self.simulator.id2agent[execution_log.sell_agent_id]
                        agent.executed_order(log=execution_log)
                        # 触发执行订单后的事件
                        self.simulator._trigger_event_after_execution(
                            execution_log=execution_log
                        )
                # ----对本地订单进行处理</>----

            # 控制高频交易员下单的频率（通过随机数决定后续代码是否执行）
            if session.high_frequency_submission_rate < self._prng.random():
                continue

            # ----对高频交易员产生的订单进行处理----
            n_high_freq_orders = 0
            # 将高频交易代理随机排序
            agents = self.simulator.high_frequency_agents
            agents = self._prng.sample(agents, len(agents))
            # 遍历所有的高频代理
            for agent in agents:
                if n_high_freq_orders >= session.max_high_frequency_orders:
                    break

                # 高频代理提交的订单
                high_freq_orders: List[Union[Order, Cancel]] = agent.submit_orders(
                    markets=self.simulator.markets
                )
                if len(high_freq_orders) > 0:
                    # 检查当前会话是否接受该订单
                    if not session.with_order_placement:
                        raise AssertionError("currently order is not accepted")

                    # 验证high_freq_orders中的订单是由高频代理所提交的
                    if (
                        sum(
                            [
                                order.agent_id != agent.agent_id
                                for order in high_freq_orders
                            ]
                        )
                        > 0
                    ):
                        raise ValueError(
                            "spoofing order is not allowed. please check agent_id in order"
                        )

                    all_orders.append(high_freq_orders)
                    # TODO: currently the original impl is used
                    n_high_freq_orders += 1
                    # n_high_freq_orders += len(high_freq_orders)
                    for order in high_freq_orders:
                        market = self.simulator.id2market[order.market_id]
                        # 将订单添加到市场，触发相关事件并将日志记录到代理
                        if isinstance(order, Order):
                            self.simulator._trigger_event_before_order(order=order)
                            log = market._add_order(order=order)
                            agent = self.simulator.id2agent[order.agent_id]
                            agent.submitted_order(log=log)
                            self.simulator._trigger_event_after_order(order_log=log)
                        elif isinstance(order, Cancel):
                            self.simulator._trigger_event_before_cancel(cancel=order)
                            log_ = market._cancel_order(cancel=order)
                            agent = self.simulator.id2agent[order.order.agent_id]
                            agent.canceled_order(log=log_)
                            self.simulator._trigger_event_after_cancel(cancel_log=log_)
                        else:
                            raise NotImplementedError
                        # 若启用了订单执行，则执行订单并触发相应事件
                        if session.with_order_execution:
                            logs = market._execution()
                            self.simulator._update_agents_for_execution(
                                execution_logs=logs
                            )
                            for execution_log in logs:
                                agent = self.simulator.id2agent[
                                    execution_log.buy_agent_id
                                ]
                                agent.executed_order(log=execution_log)
                                agent = self.simulator.id2agent[
                                    execution_log.sell_agent_id
                                ]
                                agent.executed_order(log=execution_log)
                                self.simulator._trigger_event_after_execution(
                                    execution_log=execution_log
                                )
            # ----对高频交易员产生的订单进行处理</>----
        return all_orders

    # 更新市场状态：收集agent的订单并执行
    def _update_markets(self, session: Session) -> None:
        """update markets. (Internal method)

        Args:
            session (Session): session.

        Returns:
            None
        """
        # 收集所有normal agent的订单，返回所有agent的订单列表
        local_orders: List[
            List[Union[Order, Cancel]]
        ] = self._collect_orders_from_normal_agents(session=session)
        # 处理这些订单
        self._handle_orders(session=session, local_orders=local_orders)

    # 迭代更新市场（对所有市场进行多次迭代；更新市场和代理状态，并触发相关事件）
    def _iterate_market_updates(self, session: Session) -> None:
        """iterate market updates. (Internal method)

        Args:
            session (Session): session.

        Returns:
            None
        """
        # 获取所有市场
        markets: List[Market] = self.simulator.markets
        # 根据当前session属性设置市场状态
        for market in markets:
            market._is_running = session.with_order_execution

        # 进行session.iteration_steps次迭代
        for _ in range(session.iteration_steps):
            for market in markets:
                # 触发模拟器的market_before事件
                self.simulator._trigger_event_before_step_for_market(market=market)

                # 创建“市场交易开始”记录，并写入到logger中
                if self.logger is not None:
                    log: Log = MarketStepBeginLog(
                        session=session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
            # 更新市场状态：收集agent的订单并执行（若session.with_order_placement为True）
            if session.with_order_placement:
                self._update_markets(session=session)
            # 创建“市场交易结束”记录，并写入到logger中
            for market in markets:
                if self.logger is not None:
                    log = MarketStepEndLog(
                        session=session, market=market, simulator=self.simulator
                    )
                    log.read_and_write_with_direct_process(logger=self.logger)
                # 触发模拟器的market_after事件
                self.simulator._trigger_event_after_step_for_market(market=market)
            # 更新市场的时间（更新所有市场的时间和基本面价值）
            self.simulator._update_times_on_markets(self.simulator.markets)  # t++

    # 对多个session进行循环迭代（人工市场运行的核心）
    def _run(self) -> None:
        """main process. (Internal method)"""
        if self.logger is not None:
            # 创建SimulationBeginLog对象存储模拟开始的信息
            log: Log = SimulationBeginLog(simulator=self.simulator)  # must be blocking
            # 将对象写入logger
            log.read_and_write(logger=self.logger)
            # 确保信息已被写入logger
            self.logger._process()
        # 更新所有市场的时间（将当前t值从-1改为0）
        self.simulator._update_times_on_markets(self.simulator.markets)  # t: -1 -> 0

        # 遍历所有session，并逐个进行处理
        for session in self.simulator.sessions:
            # 当前会话对象<-session
            self.simulator.current_session = session
            # 触发模拟器的session_before事件
            self.simulator._trigger_event_before_session(session=session)
            # 创建SessionBeginLog对象，记录当前会话开始的日志信息，并写入到logger中
            if self.logger is not None:
                log = SessionBeginLog(
                    session=session, simulator=self.simulator
                )  # must be blocking
                log.read_and_write(logger=self.logger)
                self.logger._process()
            # 迭代更新市场（对所有市场进行多次迭代；更新市场和代理状态，并触发相关事件）
            self._iterate_market_updates(session=session)
            # 触发模拟器的session_after事件
            self.simulator._trigger_event_after_session(session=session)
            # 创建SessionEndLog对象，记录当前会话结束的日志信息，并写入到logger中
            if self.logger is not None:
                log = SessionEndLog(
                    session=session, simulator=self.simulator
                )  # must be blocking
                log.read_and_write(logger=self.logger)
                self.logger._process()
        # 创建SimulationEndLog对象，记录模拟结束的日志信息，并写入到logger中
        if self.logger is not None:
            log = SimulationEndLog(simulator=self.simulator)  # must be blocking
            log.read_and_write(logger=self.logger)
            self.logger._process()
