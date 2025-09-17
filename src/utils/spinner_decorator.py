import threading
import itertools
import sys
import time
from functools import wraps

def spinner(stop_event, message):
    for c in itertools.cycle('|/-\\'):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{message} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(message) + 4) + '\r')

def spinner_decorator(message, async_mode=False):
    def decorator(func):
        if async_mode:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                stop_event = threading.Event()
                t = threading.Thread(target=spinner, args=(stop_event, message))
                t.start()
                try:
                    result = await func(*args, **kwargs)
                finally:
                    stop_event.set()
                    t.join()
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                stop_event = threading.Event()
                t = threading.Thread(target=spinner, args=(stop_event, message))
                t.start()
                try:
                    result = func(*args, **kwargs)
                finally:
                    stop_event.set()
                    t.join()
                return result
            return sync_wrapper
    return decorator

# Usage:
# @spinner_decorator("Loading schema...", async_mode=False)
# def infer_schema_with_spinner(): ...

# @spinner_decorator("Running...", async_mode=True)
# async def run_agent_with_spinner(...): ...