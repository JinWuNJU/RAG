import asyncio
from typing import Awaitable, Callable, Coroutine, List
from pydantic import BaseModel

# worker函数类型，接收任务队列和结果列表作为参数
WorkerType = Callable[[asyncio.Queue, List[BaseModel | None]], Coroutine[None, None, None]]

def worker_wrapper():
    """
    装饰器工厂函数，用于包装异步任务函数。
    
    返回的装饰器将普通异步函数转换为符合WorkerType类型的异步任务函数，
    被装饰函数接收任务index，执行任务，返回pydantic.BaseModel对象或None。
    
    示例：
    ```python
        def process_data(data: List[str]):
            @worker_wrapper()
            async def my_worker(index: int) -> BaseModel | None:
                nonlocal data
                value = data[index]
                # 处理任务并获得结果
                # ...
                value1 = ...
                value2 = ...
                return SomeModel(field1=value1, field2=value2)
    ```
    """
    def decorator(func: Callable[[int], Awaitable[BaseModel | None]]) -> WorkerType:
        async def wrapper(queue: asyncio.Queue, result: List[BaseModel | None]) -> None:
            """
            包装后的异步任务函数，从队列中获取任务并执行。
            
            参数：
                queue (asyncio.Queue): 任务队列，存储任务索引。
                result (list[BaseModel]): 结果列表，用于存储任务结果。
            """
            while True:
                try:
                    # 从队列中获取任务索引
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    # 如果队列为空，退出循环
                    break
                try:
                    # 执行任务函数并获取结果
                    eval_result = await func(i)
                    if eval_result:
                        result[i] = eval_result
                    else:
                        result[i] = None
                finally:
                    # 标记任务完成
                    queue.task_done()
        return wrapper
    return decorator

async def worker_dispatch(worker: WorkerType, max_tasks: int, worker_num: int) -> List[BaseModel | None]:
    """
    异步任务管理，用于发起多个任务。
    
    参数：
        worker (WorkerType): 符合WorkerType类型的异步任务函数。
        max_tasks (int): 最大任务数。
        worker_num (int): 并发任务的最大同时任务数。
    
    返回：
        list: 包含所有任务结果的列表，异常的任务结果为None。
    
    示例：
    ```python
        @worker_wrapper()
        async def my_worker(index: int) -> MyModel | None:
            ...
        
        results = await worker_dispatch(my_worker, max_tasks=10, batch_size=3)
    ```
    """
    result = []
    total_tasks = 0
    queue = asyncio.Queue()
    
    for i in range(max_tasks):
        if i >= len(result):
            result.append(None)
        if result[i] is None:
            total_tasks += 1
            await queue.put(i)
    if total_tasks == 0:
        return result

    workers = [
        asyncio.create_task(worker(queue, result))
        for _ in range(min(worker_num, total_tasks))
    ]
    await queue.join()
    
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    return result