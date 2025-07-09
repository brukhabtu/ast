"""
Common decorator patterns in Python code.
"""

from functools import wraps, lru_cache
from typing import Callable, TypeVar, ParamSpec
import time

P = ParamSpec('P')
T = TypeVar('T')

# Simple decorator
def timer(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Decorator factory
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            raise RuntimeError("Should not reach here")
        return wrapper
    return decorator

# Class with decorated methods
class DataProcessor:
    def __init__(self):
        self._cache = {}
    
    @property
    def cache_size(self) -> int:
        return len(self._cache)
    
    @staticmethod
    def validate_input(data: str) -> bool:
        return bool(data and data.strip())
    
    @classmethod
    def from_config(cls, config: dict) -> 'DataProcessor':
        return cls()
    
    @lru_cache(maxsize=128)
    def expensive_operation(self, n: int) -> int:
        return sum(i ** 2 for i in range(n))
    
    @timer
    @retry(max_attempts=3)
    def process_data(self, data: str) -> str:
        if not self.validate_input(data):
            raise ValueError("Invalid input")
        return data.upper()

# Stacked decorators
@timer
@retry(max_attempts=5, delay=0.5)
@lru_cache(maxsize=256)
def complex_calculation(x: int, y: int) -> int:
    """Function with multiple decorators."""
    return x ** y

# Property decorators
class Temperature:
    def __init__(self, celsius: float = 0.0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5/9