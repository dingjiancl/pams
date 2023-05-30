import heapq
from typing import Dict
from typing import List
from typing import Optional

from .order import Cancel
from .order import Order


class OrderBook:
    """Order book class."""

    def __init__(self, is_buy: bool) -> None:
        """initialization.

        Args:
            is_buy (bool): whether it is a buy order or not.

        Returns:
            None
        """
        self.priority_queue: List[Order] = []  # 一个优先队列，用于存储订单，其中Order是订单类
        heapq.heapify(self.priority_queue)
        self.time: int = 0  # 当前时间
        self.is_buy = is_buy  # 当前订单簿是买单簿还是卖单簿
        self.expire_time_list: Dict[int, List[Order]] = {}  # 字典，键：过期时间，值：过期订单列表

    def __repr__(self) -> str:
        return f"<{self.__class__.__module__}.{self.__class__.__name__} | is_buy={self.is_buy}>"

    # 添加一个订单到订单簿中
    def add(self, order: Order) -> None:
        """add the book of order.

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            None
        """
        if order.is_buy != self.is_buy:
            raise ValueError("buy/sell is incorrect")
        order.placed_at = self.time  # 订单被提交到交易所的时间=OrderBook对象的当前时间
        heapq.heappush(self.priority_queue, order)
        # 如果订单中包含ttl(生存时间)，则将订单添加到expire_time_list中以便之后的过期处理
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            if expiration_time not in self.expire_time_list:
                self.expire_time_list[expiration_time] = []
            self.expire_time_list[expiration_time].append(order)

    def _remove(self, order: Order) -> None:
        """remove the book of order. (Internal method. Usually, it is not called from the outside of this class.)

        Args:
            order (:class:`pams.order.Order`): order.

        Returns:
            None
        """
        if order == self.priority_queue[0]:
            x = heapq.heappop(self.priority_queue)
            assert x == order
        else:
            # 从订单簿中删除某个订单
            self.priority_queue.remove(order)  # 删除订单簿中的订单
            heapq.heapify(self.priority_queue)
        if order.placed_at is None:
            raise AssertionError("the order is not yet placed")
        if order.ttl is not None:
            expiration_time = order.placed_at + order.ttl
            self.expire_time_list[expiration_time].remove(order)  # 在过期时间列表中删除订单


    def cancel(self, cancel: Cancel) -> None:
        """cancel the book of order.

        Args:
            cancel (:class:`pams.order.Cancel`): cancel order.

        Returns:
            None
        """
        """
        接受参数：Cancel对象
        从订单簿中删除对应订单(Order)，同时将订单的is_canceled属性设置为True
        """
        cancel.order.is_canceled = True
        cancel.placed_at = self.time
        if cancel.order in self.priority_queue:
            # in case that order is executed before canceling.
            self._remove(cancel.order)

    def get_best_order(self) -> Optional[Order]:
        """get the order with the highest priority.

        Returns:
            :class:`pams.order.Order`, Optional: the order with the highest priority.
        """
        # 获取订单簿中最优的订单(Order)
        if len(self.priority_queue) > 0:
            return self.priority_queue[0]  # 返回优先队列(priority_queue)的队首元素(即最优订单)
        else:
            return None

    # 获取订单簿中最优订单的价格
    def get_best_price(self) -> Optional[float]:
        """get the order price with the highest priority.

        Returns:
            float, Optional: the order price with the highest priority.
        """
        if len(self.priority_queue) > 0:
            return self.priority_queue[0].price  # 返回优先队列(priority_queue)的队首元素(即最优订单)的价格
        else:
            return None

    # 更改一个订单(Order)的成交量
    def change_order_volume(self, order: Order, delta: int) -> None:
        """change order volume.

        Args:
            order (:class:`pams.order.Order`): order.
            delta (int): amount of volume change.

        Returns:
            None
        """
        order.volume += delta  # 将订单的成交量(volume)加上delta
        # ToDo: check if volume is non-negative
        if order.volume == 0:
            self._remove(order=order)  # 如果订单的成交量(volume)为0，则将该订单从订单簿中删除
        if order.volume < 0:
            raise AssertionError

    # 检查订单是否过期并将过期订单从订单簿中删除
    def _check_expired_orders(self) -> None:
        """check and delete expired orders. (Internal Method)"""
        # 将expire_time_list中所有过期时间小于当前时间(time)的订单从priority_queue队列中删除
        delete_orders: List[Order] = sum(
            [value for key, value in self.expire_time_list.items() if key < self.time],
            [],
        )
        delete_keys: List[int] = [
            key for key, value in self.expire_time_list.items() if key < self.time
        ]
        if len(delete_orders) == 0:
            return
        # TODO: think better sorting in the following 3 lines
        for delete_order in delete_orders:
            self.priority_queue.remove(delete_order)
        heapq.heapify(self.priority_queue)
        for key in delete_keys:
            self.expire_time_list.pop(key)  # 从expire_time_list中删除对应的过期时间

    # 设置当前时间(time)（+检查订单是否过期）
    def _set_time(self, time: int) -> None:
        """set time step. (Usually, it is called from market.)

        Args:
            time (int): time step.

        Returns:
            None
        """
        self.time = time
        self._check_expired_orders()

    # 更新当前时间(time)（+检查订单是否过期）
    def _update_time(self) -> None:
        """update time. (Usually, it is called from market.)
        Advance the time step and check expired orders.
        """
        self.time += 1
        self._check_expired_orders()

    # 获取订单簿中订单(Order)的数量
    def __len__(self) -> int:
        """get length of the order queue.

        Returns:
            int: length of the order queue.
        """
        return len(self.priority_queue)

    # 获取订单簿中不同价格(price)的订单成交量(volume)
    '''
    返回：字典
    键：订单价格(price)；值：订单成交量(volume)
    如果订单簿中存在市价订单，则键为None
    '''
    def get_price_volume(self) -> Dict[Optional[float], int]:
        """get price and volume (order book).

        Returns:
            Dict[Optional[float], int]: order book dict. Dict key is order price and the value is volumes.
        """
        """
        1、使用map函数和lambda表达式从优先级队列中获取所有订单的价格
        2、用set函数去重得到一个集合
        3、（集合中可能包含None值，它表示市价订单，其价格不固定）
        """
        keys: List[Optional[float]] = list(
            set(map(lambda x: x.price, self.priority_queue))
        )
        has_market_order: bool = None in keys
        if has_market_order:
            keys.remove(None)  # 移除集合中的None值
        keys.sort(reverse=self.is_buy)  # 根据is_buy属性（表示是否为买单）将价格升序或降序排序
        if has_market_order:
            keys.insert(0, None)  # 如果集合中包含None值，将它插入到列表的开头，以便在返回结果时它能排在第一位
        result: Dict[Optional[float], int] = dict(
            [
                (
                    key,
                    sum(
                        [
                            order.volume
                            for order in self.priority_queue
                            if order.price == key
                        ]
                    ),
                )
                for key in keys
            ]
        )
        return result
