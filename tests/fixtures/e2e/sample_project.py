"""Sample Python project for e2e testing"""

class Calculator:
    """A simple calculator class for testing AST analysis"""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a"""
        return a - b
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: int, b: int) -> float:
        """Divide a by b"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


def main():
    """Main function to demonstrate calculator usage"""
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")


if __name__ == "__main__":
    main()