import asyncio
from collections import deque
import time
from typing import Deque


class WindowRateLimiter:
    """
    滑动窗口限流器，确保在指定时间窗口内的请求不超过最大限制
    """
    def __init__(self, max_requests: int, window_seconds: int = 60, sleep_time: float = 0.1):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps: Deque[float] = deque()
        self.lock = asyncio.Lock()
        self.sleep_time = sleep_time
    
    async def acquire(self):
        """
        尝试获取一个请求许可，如果超过限制则等待
        """
        while True:
            async with self.lock:
                # 清除窗口外的时间戳
                current_time = time.time()
                while self.request_timestamps and current_time - self.request_timestamps[0] > self.window_seconds:
                    self.request_timestamps.popleft()
                
                # 检查是否可以发送新请求
                if len(self.request_timestamps) < self.max_requests:
                    self.request_timestamps.append(current_time)
                    return
            
            await asyncio.sleep(self.sleep_time)