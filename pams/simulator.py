import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import cast

from .agents.base import Agent
from .agents.high_frequency_agent import HighFrequencyAgent
from .events.base import EventABC
from .events.base import EventHook
from .fundamentals import Fundamentals
from .index_market import IndexMarket
from .logs.base import Logger
from .market import Market
from .session import Session


class Simulator:
    """Simulator class."""

    def __init__(
        self,
        prng: random.Random,
        logger: Optional[Logger] = None,
        fundamental_class: Type[Fundamentals] = Fundamentals,
    ) -> None:
        """initialization.

        Args:
            prng (random.Random): pseudo random number generator for this simulator.
            logger (Logger, Optional): logger for correcting various outputs in one simulation.
                                       logger is usually shared to all classes.
                                       Please note that logger is usually not thread-safe and non-blocking.
            fundamental_class (Type[Fundamentals]): the class that provide fundamental functions for simulator
                                                    (default :class:`pams.fundamentals.Fundamentals`).

        Note:
             `prng` should not be shared with other classes and be used only in this class.
             It is because sometimes agent process runs one of parallelized threads.
        """
        self._prng = prng  # 随机数生成器对象
        self.logger: Optional[Logger] = logger  # 日志记录器对象
        if self.logger is not None:
            self.logger._set_simulator(simulator=self)

        self.n_events: int = 0  # 事件的数量
        self.events: List[EventABC] = []  # 事件列表
        self.id2event: Dict[int, EventABC] = {}  # 事件ID到事件对象的映射字典
        self.event_hooks: List[EventHook] = []  # 事件钩子列表
        self.events_dict: Dict[str, Dict[Optional[int], List[EventHook]]] = {
            "order_before": {},
            "order_after": {},
            "cancel_before": {},
            "cancel_after": {},
            "execution_after": {},
            "session_before": {},
            "session_after": {},
            "market_before": {},
            "market_after": {},
        }  # 按类型和时间组织的事件钩子的嵌套字典
        self.name2event: Dict[str, EventABC] = {}  # 事件名称到事件对象的映射字典

        self.n_agents: int = 0  # 代理的数量
        self.agents: List[Agent] = []  # 代理列表
        self.high_frequency_agents: List[Agent] = []  # 高频代理列表
        self.normal_frequency_agents: List[Agent] = []  # 普通代理列表
        self.id2agent: Dict[int, Agent] = {}  # 代理ID到代理对象的映射字典
        self.name2agent: Dict[str, Agent] = {}  # 代理名称到代理对象的映射字典
        self.agents_group_name2agent: Dict[str, List[Agent]] = {}  # 代理组名到代理列表的映射字典

        self.n_markets: int = 0  # 市场的数量
        self.markets: List[Market] = []  # 市场列表
        self.id2market: Dict[int, Market] = {}  # 市场ID到市场对象的映射字典
        self.name2market: Dict[str, Market] = {}  # 市场名称到市场对象的映射字典
        self.markets_group_name2market: Dict[str, List[Market]] = {}  # 市场组名到市场列表的映射字典

        self.fundamentals = fundamental_class(
            prng=random.Random(self._prng.randint(0, 2**31))
        )  # 基本面类对象，用于计算资产的基本价值

        self.n_sessions: int = 0  # 会话的数量
        self.sessions: List[Session] = []  # 会话列表
        self.id2session: Dict[int, Session] = {}  # 会话ID到会话对象的映射字典
        self.name2session: Dict[str, Session] = {}  # 会话名称到会话对象的映射字典
        self.current_session: Optional[Session] = None  # 当前会话对象。如果未设置，则为None

    # 向事件钩子列表中添加事件钩子（将事件对象添加到模拟器中）
    def _add_event(self, event_hook: EventHook) -> None:
        """add event to the simulator. (Usually, this is called from runner.)

        Args:
            event_hook (:class:`pams.events.base.EventHook`): event hook.

        Returns:
            None
        """
        if event_hook in self.event_hooks:  # 检查该事件钩子是否已经注册
            raise ValueError("event_hook is already registered")
        event = event_hook.event
        if event_hook.event not in self.events:
            self.events.append(event)
        self.n_events += 1
        # 将事件id和事件名称添加到相应的字典中，方便以后的查找
        if event_hook.event.event_id not in self.id2event:
            self.id2event[event.event_id] = event
        if event_hook.event.name not in self.name2event:
            self.name2event[event.name] = event
        # 将事件钩子添加到事件钩子列表中
        self.event_hooks.append(event_hook)
        # 根据事件钩子的类型和时机（before或after）
        # 确定要将事件钩子添加到哪个字典(按类型和时间组织的事件钩子的嵌套字典)中
        # 并将其添加到该字典中
        register_name: str = event_hook.hook_type + (
            "_before" if event_hook.is_before else "_after"
        )
        times: List[Optional[int]] = (
            cast(List[Optional[int]], event_hook.time)
            if event_hook.time is not None
            else cast(List[Optional[int]], [None])
        )
        for time_ in times:
            if time_ not in self.events_dict[register_name]:
                self.events_dict[register_name][time_] = []
            self.events_dict[register_name][time_].append(event_hook)

    # 向市场列表中添加新的市场（将市场对象添加到模拟器中）
    def _add_market(self, market: Market, group_name: Optional[str] = None) -> None:
        """add market to the simulator. (Usually, this is called from runner.)

        Args:
            market (:class:`pamd.market.Market`): market.
            group_name (str, Optional): group name for market (default None).

        Returns:
            None
        """
        # market：要添加的市场
        # group_name：要将市场添加到的市场组名称（如果不提供该参数，则将市场添加到默认的市场组中）
        if market in self.markets:
            raise ValueError("market is already registered")
        # 检查新市场的ID和名称是否已经在列表中出现
        if market.market_id in self.id2market:
            raise ValueError(f"market_id {market.market_id} is duplicated")
        if market.name in self.name2market:
            raise ValueError(f"market name {market.name} is duplicate")
        self.markets.append(market)  # 新市场添加到列表中
        self.n_markets += 1  # 计数器+1
        # 建立id、名称与对象的映射关系
        self.id2market[market.market_id] = market
        self.name2market[market.name] = market
        # 指定了市场组时，将新市场添加到指定组中
        if group_name is not None:
            if group_name not in self.markets_group_name2market:
                self.markets_group_name2market[group_name] = []
            self.markets_group_name2market[group_name].append(market)

    # 向代理列表中添加代理对象（将代理对象添加到模拟器中）
    def _add_agent(self, agent: Agent, group_name: Optional[str] = None) -> None:
        """add agent to the simulator. (Usually, this is called from runner.)

        Args:
            agent (:class:`pams.agents.base.Agent`): agent.
            group_name (str, Optional): group name for agent (default None).

        Returns:
            None
        """
        # 检查代理是否已被注册
        if agent in self.agents:
            raise ValueError("agent is already registered")
        if agent.agent_id in self.id2agent:
            raise ValueError(f"agent_id {agent.agent_id} is duplicated")
        if agent.name in self.name2agent:
            raise ValueError(f"agent name {agent.name} is duplicate")
        self.agents.append(agent)  # 添加到模拟器中
        self.n_agents += 1  # 计数器+1
        # 建立id、名称与代理对象的映射关系
        self.id2agent[agent.agent_id] = agent
        self.name2agent[agent.name] = agent
        # 判断是普通代理还是高频代理，并添加到相应的列表
        if isinstance(agent, HighFrequencyAgent):
            self.high_frequency_agents.append(agent)
        else:
            self.normal_frequency_agents.append(agent)
        # 指定了分组时，添加到指定组中
        if group_name is not None:
            if group_name not in self.agents_group_name2agent:
                self.agents_group_name2agent[group_name] = []
            self.agents_group_name2agent[group_name].append(agent)

    # 向会话列表中添加会话对象（将会话对象添加到模拟器中）
    def _add_session(self, session: Session) -> None:
        """add session to the simulator. (Usually, this is called from runner.)

        Args:
            session (:class:`pams.session.Session`): session.

        Returns:
            None
        """
        # 检查会话是否已被注册
        if session in self.sessions:
            raise ValueError("session is already registered")
        if session.session_id in self.id2session:
            raise ValueError(f"session_id {session.session_id} is duplicated")
        if session.name in self.name2session:
            raise ValueError(f"session name {session.name} is duplicate")
        # 若未被注册则添加到列表中，并建立映射关系
        self.sessions.append(session)
        self.n_sessions += 1
        self.id2session[session.session_id] = session
        self.name2session[session.name] = session

    # 更新市场的时间和基本面价值（普通市场+指数市场）
    # （对于指数市场，在更新时间前需要保证组成指数的各个市场的时间都已经被更新）
    def _update_time_on_market(self, market: Market) -> None:
        """update time on the market. (Usually, this is called from runner.)

        Args:
            market (:class:`pams.market.Market`): market.

        Returns:
            None

        Notes:
            be careful index matket have to be update after component markets.
            Technically, the fundamental values for components markets can be calculated beforehand, but not allowed to avoid future data leakage.
        """
        if not isinstance(market, IndexMarket):
            market._update_time(  # 更新当前市场时间与当前市场价格
                next_fundamental_price=self.fundamentals.get_fundamental_price(
                    market_id=market.market_id, time=market.get_time() + 1
                )
            )
        else:
            market._update_time(  # 更新当前市场时间与当前市场价格
                next_fundamental_price=market.compute_fundamental_index(  # 计算基本指数
                    time=market.get_time() + 1
                )
            )

    # 更新所给列表中所有市场的时间和基本面价值
    def _update_times_on_markets(self, markets: List[Market]) -> None:
        """update times on markets. (Usually, this is called from runner.)

        Args:
            markets (List[:class:`pams.market.Market`]): list of markets.

        Returns:
            None
        """
        for market in filter(lambda x: not isinstance(x, IndexMarket), markets):
            self._update_time_on_market(market=market)
        for market in filter(lambda x: isinstance(x, IndexMarket), markets):
            self._update_time_on_market(market=market)

    # 更新代理的资产和现金金额
    def _update_agents_for_execution(
        self, execution_logs: List["ExecutionLog"]  # type: ignore  # NOQA
    ) -> None:
        """update agents for execution. (Usually, this is called from runner.)

        Args:
            execution_logs (List["ExecutionLog"]): execution logs.

        Returns:
            None
        """
        # 遍历执行日志列表
        for log in execution_logs:
            buy_agent: Agent = self.id2agent[log.buy_agent_id]  # 买方代理
            sell_agent: Agent = self.id2agent[log.sell_agent_id]  # 卖方代理
            price: float = log.price  # 交易价格
            volume: int = log.volume  # 交易数量
            market_id: int = log.market_id  # 交易市场ID
            # 更新代理的信息（现金金额+资产数量）
            buy_agent.cash_amount -= price * volume
            sell_agent.cash_amount += price * volume
            buy_agent.asset_volumes[market_id] += volume
            sell_agent.asset_volumes[market_id] -= volume

    # 检查给定的对象是否满足指定的类和实例要求
    def _check_event_class_and_instance(
        self,
        check_object: object,  # 要检查的对象
        class_requirement: Optional[Type] = None,  # 要求对象所属的类
        instance_requirement: Optional[object] = None,  # 要求对象是某个具体实例
    ) -> bool:
        """check event class and instance. (Usually, this is called from runner.)

        Args:
            check_object (object): object for check.
            class_requirement (Type, Optional): class requirement.
            instance_requirement (object, Optional): instance requirement.

        Returns:
            bool: whether the event class or instance meet the requirements.
        """
        if class_requirement is not None:
            if not isinstance(check_object, class_requirement):
                return False
        if instance_requirement is not None:
            if instance_requirement != check_object:
                return False
        return True

    # 在提交订单前触发事件
    def _trigger_event_before_order(self, order: "Order") -> None:  # type: ignore  # NOQA
        """trigger event before order. (Usually, this is called from runner.)

        Args:
            order (Order): the order before the event.

        Returns:
            None
        """
        time: int = order.placed_at
        event_hooks = self.events_dict["order_before"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_before_order(simulator=self, order=order)

    # 在订单执行完毕后触发事件
    def _trigger_event_after_order(self, order_log: "OrderLog") -> None:  # type: ignore  # NOQA
        """trigger event after order. (Usually, this is called from runner.)

        Args:
            order_log (OrderLog): the order log after the event.

        Returns:
            None
        """
        time: int = order_log.time
        event_hooks = self.events_dict["order_after"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_after_order(simulator=self, order_log=order_log)

    # 在撤销订单前触发事件
    def _trigger_event_before_cancel(self, cancel: "Cancel") -> None:  # type: ignore  # NOQA
        """trigger event before cancel. (Usually, this is called from runner.)

        Args:
            cancel (Cancel): the cancel order before the event.

        Returns:
            None
        """
        time: int = cancel.placed_at
        event_hooks = self.events_dict["cancel_before"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_before_cancel(simulator=self, cancel=cancel)

    # 在撤销订单后触发事件
    def _trigger_event_after_cancel(self, cancel_log: "CancelLog") -> None:  # type: ignore  # NOQA
        """trigger event after cancel. (Usually, this is called from runner.)

        Args:
            cancel_log (CancelLog): the cancel order log after the event.

        Returns:
            None
        """
        time: int = cancel_log.cancel_time
        event_hooks = self.events_dict["cancel_after"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_after_cancel(simulator=self, cancel_log=cancel_log)

    # 在订单成交后触发事件
    def _trigger_event_after_execution(self, execution_log: "ExecutionLog") -> None:  # type: ignore  # NOQA
        """trigger event after execution. (Usually, this is called from runner.)

        Args:
            execution_log (ExecutionLog): the execution log after the event.

        Returns:
            None
        """
        time: int = execution_log.time
        event_hooks = self.events_dict["execution_after"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_after_execution(
                simulator=self, execution_log=execution_log
            )

    # 在会话前触发事件
    def _trigger_event_before_session(self, session: "Session") -> None:  # type: ignore
        """trigger event before session. (Usually, this is called from runner.)

        Args:
            session (Session): the session before the event.

        Returns:
            None
        """
        time: int = session.session_start_time
        event_hooks = self.events_dict["session_before"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_before_session(simulator=self, session=session)

    # 在会话后触发事件
    def _trigger_event_after_session(self, session: "Session") -> None:  # type: ignore
        """trigger event after session. (Usually, this is called from runner.)

        Args:
            session (Session): the session after the event.

        Returns:
            None
        """
        time: int = session.session_start_time + session.iteration_steps - 1
        event_hooks = self.events_dict["session_after"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            event_hook.event.hooked_after_session(simulator=self, session=session)

    # 在市场步骤前触发事件
    def _trigger_event_before_step_for_market(self, market: "Market") -> None:  # type: ignore
        """trigger event before step for market. (Usually, this is called from runner.)

        Args:
            market (Market): the market before the event.

        Returns:
            None
        """
        time: int = market.get_time()
        event_hooks = self.events_dict["market_before"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            if self._check_event_class_and_instance(
                check_object=market,
                class_requirement=event_hook.specific_class,
                instance_requirement=event_hook.specific_instance,
            ):
                event_hook.event.hooked_before_step_for_market(
                    simulator=self, market=market
                )

    # 在市场步骤后触发事件
    def _trigger_event_after_step_for_market(self, market: "Market") -> None:  # type: ignore
        """trigger event after step for market. (Usually, this is called from runner.)

        Args:
            market (Market): the market after the event.

        Returns:
            None
        """
        time: int = market.get_time()
        event_hooks = self.events_dict["market_after"]
        target_event_hooks: List[EventHook] = []
        if None in event_hooks:
            target_event_hooks.extend(event_hooks[None])
        if time in event_hooks:
            target_event_hooks.extend(event_hooks[time])
        for event_hook in target_event_hooks:
            if self._check_event_class_and_instance(
                check_object=market,
                class_requirement=event_hook.specific_class,
                instance_requirement=event_hook.specific_instance,
            ):
                event_hook.event.hooked_after_step_for_market(
                    simulator=self, market=market
                )

    # ToDo get_xxx_by_name
