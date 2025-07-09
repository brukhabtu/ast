"""Module with missing imports."""

from .non_existent_module import MissingClass  # This import will fail
from .good_module import GoodClass


class BrokenClass(MissingClass):  # Will fail due to missing base
    """Class with missing base."""
    
    def broken_method(self) -> str:
        """Method that would work if base existed."""
        return "broken"


def uses_missing() -> None:
    """Function using missing import."""
    obj = MissingClass()  # Would fail at runtime
    return obj.missing_method()