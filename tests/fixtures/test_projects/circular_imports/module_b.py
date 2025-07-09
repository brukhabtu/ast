"""Module B with circular import."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module_a import ClassA


class ClassB:
    """Class B that references A."""
    
    def __init__(self):
        self.name = "B"
    
    def get_a_instance(self) -> "ClassA":
        """Get instance of ClassA."""
        from .module_a import ClassA
        return ClassA()
    
    def process_with_a(self, a: "ClassA") -> str:
        """Process with A instance."""
        return f"B processing with {a.name}"


def function_b() -> str:
    """Function in module B."""
    from .module_a import function_a
    return f"B calls {function_a()}"