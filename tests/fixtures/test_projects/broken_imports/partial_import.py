"""Module with partial imports."""

from .good_module import GoodClass, good_function
from .missing_import import BrokenClass  # This will work for AST parsing
from .syntax_error import valid_function  # This import is problematic


class PartiallyWorking:
    """Class that partially works."""
    
    def __init__(self):
        self.good = GoodClass()
        # self.broken = BrokenClass()  # Would fail at runtime
    
    def working_method(self) -> str:
        """Method that works."""
        return self.good.good_method()
    
    def failing_method(self) -> str:
        """Method that would fail."""
        broken = BrokenClass()
        return broken.broken_method()