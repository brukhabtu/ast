"""Module with syntax errors."""

# Valid part
def valid_function():
    return "valid"

# Syntax error - missing colon
def broken_function()
    return "broken"

# More valid code that won't be parsed
class AfterError:
    """This class comes after the syntax error."""
    pass