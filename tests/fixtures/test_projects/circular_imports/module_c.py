"""Module C that imports both A and B."""

from .module_a import ClassA, function_a
from .module_b import ClassB, function_b


class ClassC:
    """Class C using both A and B."""
    
    def __init__(self):
        self.a = ClassA()
        self.b = ClassB()
    
    def orchestrate(self) -> str:
        """Orchestrate A and B."""
        result_a = self.a.process_with_b(self.b)
        result_b = self.b.process_with_a(self.a)
        return f"{result_a} | {result_b}"


def function_c() -> str:
    """Function using both A and B."""
    return f"C uses: {function_a()} and {function_b()}"