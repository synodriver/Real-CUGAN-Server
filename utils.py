import asyncio
import contextvars
from functools import partial, wraps
from typing import Any, Awaitable, Callable


def run_sync(func: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """
    一个用于包装 sync function 为 async function 的装饰器
    :param func:
    :return:
    """

    @wraps(func)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        pfunc = partial(func, *args, **kwargs)
        context = contextvars.copy_context()
        context_run = context.run
        result = await loop.run_in_executor(None, context_run, pfunc)
        return result

    return _wrapper
