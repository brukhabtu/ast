"""Module A with circular import."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module_b import ClassB


class ClassA:
    """Class A that references B."""
    
    def __init__(self):
        self.name = "A"
    
    def get_b_instance(self) -> "ClassB":
        """Get instance of ClassB."""
        from .module_b import ClassB
        return ClassB()
    
    def process_with_b(self, b: "ClassB") -> str:
        """Process with B instance."""
        return f"A processing with {b.name}"


def function_a() -> str:
    """Function in module A."""
    from .module_b import function_b
    return f"A calls {function_b()}"