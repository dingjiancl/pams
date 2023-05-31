from abc import ABC, abstractmethod
import os
from io import TextIOWrapper
from cfd_market import CFDSimulator
import gym
from gym import spaces
import numpy as np
from pams.runners import Runner, SequentialRunner
from pams.order import Order, Cancel
from pams.logs.base import Logger
from pams.market import Market
from pams.session import Session
from pams.simulator import Simulator
import random
from typing import Any, Dic, Dict, List, Optional, Tuple, Type, Union

class SequentialRunnerForMMEnv(SequentialRunner):
    """Sequential Runner for PamsMMEnv class.

    This runner is implemented for the good of PamsMMEnv.
    """
    def __init__(
        self,
        target_mmagent_name: str,
        settings: Union[str, Dic, TextIOWrapper, os.PathLike],
        prng: Optional[random.Random] = None,
        logger: Optional[Logger] = None,
        simulator_class: Type[Simulator] = Simulator
    ) -> None:
        """initialization.

        Args:
            target_mmagent_name (str): the name of RL mm agent.
        """
        super().__init__(settings, prng, logger, simulator_class)
        self.target_mmagent_name = target_mmagent_name

    def _handle_orders(
        self,
        session: Session,
        local_orders: List[List[Union[Order, Cancel]]]
    ):
        """handle orders.

        Usual ._handle_orders method
        1. processes orders by normal agents.
        2. collects orders by high frequency agents.
        3. processes orders by high frequency agents.
        but this runner, used by PamsMMEnv, doesn't collect orders by
        specified target_mmagent_name agent (= partially skip above procedure 2.)
        in order to collect orders from him or her only through PAMSMMEnv.reset/step method.

        Args:
            session (Session):
            local_orders (List): list of orders by normal agents collected in procedure 1.,
                        and by target_mmagent_name agent collected through PAMSMMEnv.reset/step method.
        """
        for agent in self.simulator.high_frequency_agents:
            if agent.name == self.target_mmagent_name:
                target_agent = agent
                self.simulator.high_frequency_agents.remove(target_agent)
                break
        all_orders = super()._handle_orders(session, local_orders)
        self.simulator.high_frequency_agents.append(target_agent)
        return all_orders

class PamsMMEnv(gym.Env, ABC):
    """PamsMMEnv class.

    RL Environment for high frequency mm agent in pams.
    This class inherits from the gym.Env class.
    """
    def __init__(
        self,
        seed: int,
        config_dic: Dic[Any],
        variable_ranges_dic: Dic[Any],
        simulator_class: Type[Simulator],
        target_mmagent_name: str
    ) -> None:
        """initialization.

        Args:
            seed (int): random seed.
            config_dic (Dic): runner configuration. (=settings)
            variable_ranges_dic (Dic): dic to specify the ranges of values for variables in config.
                Ex: {"market1": {"fundamentalPrice": [100,200], "findamentalDrift": [-1,1], ...}, ...}
                The values of variables are uniformelly sampled by .modify_config() method at each episode.
            simulator_class: type of simulator.
        """
        self.config_dic = config_dic
        self.variable_ranges_dic = variable_ranges_dic
        self.simulator_class = simulator_class
        self.seed = seed
        self.action_space, self.observation_space = self.set_action_obs_space()
        self.target_mmagent_name = target_mmagent_name

    def reset(self) -> np.ndarray:
        """reset environmet.

        reset and set up runner and send initial state to mm agent.
        """
        initial_config_dic: Dic[Any] = self.config_dic.copy()
        episode_config_dic: Dic[Any] = self.modify_config(
            initial_config_dic, self.variable_ranges_dic
        )
        self.runner: Runner = SequentialRunnerForMMEnv(
            target_mmagent_name=self.target_mmagent_name,
            settings=episode_config_dic,
            prng=random.Random(self.seed),
            logger=None,
            simulator_class=self.simulator_class
        )
        self.simulator: Simulator = self.runner.simulator
        self.sessions: List[Session] = self.simulator.sessions
        self.markets: List[Market] = self.simulator.markets
        self.current_session_idx: int = 0
        self.current_session_time: int = 0  # current number of time steps within the session
        self.runner._setup()
        self.simulator._update_times_on_markets(self.simulator.markets)
        self.simulator.current_session = self.sessions[self.current_session_idx]
        self.current_session: Session = self.simulator.current_session
        self.simulator._trigger_event_before_session(session=self.current_session)
        for market in self.markets:
            market._is_running = self.current_session.with_order_execution
            self.simulator._trigger_event_before_step_for_market(market=market)
        state: np.ndarray = self.generate_state()
        return state

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dic[Any]]:
        """step environment.

        receive action by mm agent, concert it to orders, and step the environmet.

        Args:
            action (np.ndarray): action by mm agent.
        Returns:
            next_state (np.ndarray): next state.
            reward (float): immediate reward.
            done (bool): wheter the environment ended.
            info (Dic): empty dic.
        """
        mm_orders: List[Order] = self.convert_action2orders(action)
        local_orders: List[
            List[Union[Order, Cancel]]
        ] = self._collect_orders_from_normal_agents(session=self.current_session)
        local_orders.append(mm_orders)
        self.runner._handle_orders(
            session=self.current_session,
            local_orders=local_orders
        )
        for market in self.markets:
            self.simulator._trigger_event_after_step_for_market(market=market)
        reward: float = self.generate_reward()
        self.current_session_time += 1
        if self.current_session_time == self.current_session.iteration_steps:
            self.simulator._trigger_event_after_session(session=self.current_session)
            if self.current_session_idx + 1 == len(self.sessions):
                done: bool = True
                next_state: np.ndarray = self.reset()
            else:
                self.current_session_idx += 1
                self.simulator._update_times_on_markets(self.simulator.markets)
                self.simulator.current_session = self.sessions[self.current_session_idx]
                self.current_session_time = 0
                self.current_session = self.simulator.current_session
                self.simulator._trigger_event_before_session(session=self.current_session)
                for market in self.markets:
                    market._is_running = self.current_session.with_order_execution
                    self.simulator._trigger_event_before_step_for_market(market=market)
                done: bool = False
                next_state: np.ndarray = self.generate_state()
        else:
            self.simulator._update_times_on_markets(self.simulator.markets)
            self.current_session_time += 1
            for market in self.markets:
                self.simulator._trigger_event_before_step_for_market(market=market)
            done: bool = False
            next_state: np.ndarray = self.generate_state()
        info: Dic[Any] = None
        return next_state, reward, done, info

    @abstractmethod
    def set_action_obs_space(self) -> Tuple(spaces):
        pass

    @abstractmethod
    def modify_config(
        self,
        config_dic: Dic[Any],
        variable_ranges_dic: Dic[Any]
        # Ex: {"market1": {"fundamentalPrice": [100,200], "findamentalDrift": [-1,1], ...}, ...}
    ):
        pass

    @abstractmethod
    def generate_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def generate_reward(self) -> float:
        pass

    @abstractmethod
    def convert_action2orders(self, action: np.ndarray) -> List[Order]:
        pass

class CFDMMEnv(PamsMMEnv):

    def generate_state():


    def generate_reward():


    def convert_action2order():
        

