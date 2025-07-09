"""
Test project generator for performance and integration testing.
"""

import random
import string
from pathlib import Path
from typing import List, Dict, Tuple


class TestProjectGenerator:
    """Generate test projects with various characteristics."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True, parents=True)
    
    def generate_large_project(self, name: str = "large_project", 
                             num_packages: int = 10,
                             modules_per_package: int = 10,
                             symbols_per_module: int = 20) -> Path:
        """Generate a large project with many files and symbols."""
        project_path = self.base_path / name
        project_path.mkdir(exist_ok=True)
        
        # Track imports for cross-references
        all_modules = []
        
        for pkg_idx in range(num_packages):
            package_name = f"package_{pkg_idx}"
            package_path = project_path / package_name
            package_path.mkdir(exist_ok=True)
            
            # Create __init__.py
            (package_path / "__init__.py").write_text(
                f'"""Package {package_name}."""\n\n'
                f'__all__ = [{", ".join(f'"module_{i}"' for i in range(modules_per_package))}]\n'
            )
            
            for mod_idx in range(modules_per_package):
                module_name = f"module_{mod_idx}"
                module_path = package_path / f"{module_name}.py"
                all_modules.append((package_name, module_name))
                
                # Generate module content
                content = self._generate_module_content(
                    package_name, module_name, pkg_idx, mod_idx,
                    symbols_per_module, all_modules
                )
                module_path.write_text(content)
        
        return project_path
    
    def _generate_module_content(self, package_name: str, module_name: str,
                                pkg_idx: int, mod_idx: int,
                                num_symbols: int, all_modules: List[Tuple[str, str]]) -> str:
        """Generate content for a single module."""
        lines = [
            f'"""Module {module_name} in {package_name}."""',
            "",
            "import os",
            "import sys",
            "from typing import List, Dict, Optional, Union, Tuple",
            "from dataclasses import dataclass",
            "from abc import ABC, abstractmethod",
            ""
        ]
        
        # Add cross-package imports
        if len(all_modules) > 1 and random.random() > 0.3:
            # Import from other packages
            num_imports = random.randint(1, min(3, len(all_modules)))
            for _ in range(num_imports):
                other_pkg, other_mod = random.choice(all_modules)
                if other_pkg != package_name:
                    lines.append(f"from ..{other_pkg}.{other_mod} import helper_function_{other_pkg}")
        
        lines.extend(["", ""])
        
        # Generate constants
        for i in range(min(5, num_symbols // 4)):
            lines.append(f"CONSTANT_{pkg_idx}_{mod_idx}_{i} = {random.randint(1, 100)}")
            lines.append(f'STRING_CONST_{i} = "value_{pkg_idx}_{mod_idx}_{i}"')
        
        lines.extend(["", ""])
        
        # Generate functions
        num_functions = num_symbols // 3
        for i in range(num_functions):
            lines.extend(self._generate_function(pkg_idx, mod_idx, i))
            lines.append("")
        
        # Generate classes
        num_classes = num_symbols // 3
        for i in range(num_classes):
            lines.extend(self._generate_class(pkg_idx, mod_idx, i))
            lines.append("")
        
        # Add helper function that others might import
        lines.extend([
            f"def helper_function_{package_name}(value: int) -> int:",
            f'    """Helper function for {package_name}."""',
            f"    return value * {pkg_idx + 1}",
            ""
        ])
        
        return "\n".join(lines)
    
    def _generate_function(self, pkg_idx: int, mod_idx: int, func_idx: int) -> List[str]:
        """Generate a function definition."""
        func_name = f"function_{pkg_idx}_{mod_idx}_{func_idx}"
        is_async = random.random() > 0.8
        
        lines = []
        
        # Add decorator randomly
        if random.random() > 0.7:
            decorator = random.choice(["@staticmethod", "@classmethod", "@property", 
                                     f"@custom_decorator_{func_idx}"])
            lines.append(decorator)
        
        # Function signature
        params = []
        num_params = random.randint(0, 4)
        for i in range(num_params):
            param_type = random.choice(["int", "str", "List[str]", "Dict[str, int]", "Optional[int]"])
            params.append(f"param_{i}: {param_type}")
        
        if is_async:
            lines.append(f"async def {func_name}({', '.join(params)}) -> Dict[str, int]:")
        else:
            return_type = random.choice(["int", "str", "bool", "List[int]", "None"])
            lines.append(f"def {func_name}({', '.join(params)}) -> {return_type}:")
        
        lines.append(f'    """Function {func_name}."""')
        
        # Function body
        if is_async:
            lines.append(f'    return {{"result": {func_idx}}}')
        else:
            lines.append(f"    # Complex logic here")
            lines.append(f"    result = {func_idx}")
            lines.append(f"    return result")
        
        return lines
    
    def _generate_class(self, pkg_idx: int, mod_idx: int, class_idx: int) -> List[str]:
        """Generate a class definition."""
        class_name = f"Class_{pkg_idx}_{mod_idx}_{class_idx}"
        base_class = random.choice(["", "ABC", f"BaseClass_{class_idx}", "object"])
        
        lines = []
        
        if base_class:
            lines.append(f"class {class_name}({base_class}):")
        else:
            lines.append(f"class {class_name}:")
        
        lines.append(f'    """Class {class_name}."""')
        lines.append("")
        
        # Class variables
        lines.append(f"    class_var = {class_idx}")
        lines.append(f'    class_string = "class_{pkg_idx}_{mod_idx}_{class_idx}"')
        lines.append("")
        
        # __init__ method
        lines.extend([
            "    def __init__(self, value: int = 0):",
            '        """Initialize instance."""',
            "        self.value = value",
            f"        self.name = '{class_name}'",
            "        self._private = value * 2",
            ""
        ])
        
        # Regular methods
        num_methods = random.randint(2, 5)
        for i in range(num_methods):
            method_name = f"method_{i}"
            is_async = random.random() > 0.9
            
            if is_async:
                lines.append(f"    async def {method_name}(self) -> int:")
            else:
                lines.append(f"    def {method_name}(self) -> int:")
            
            lines.append(f'        """Method {method_name}."""')
            lines.append(f"        return self.value + {i}")
            lines.append("")
        
        # Property
        lines.extend([
            "    @property",
            "    def computed_value(self) -> int:",
            '        """Computed property."""',
            "        return self.value ** 2",
            ""
        ])
        
        # Static method
        lines.extend([
            "    @staticmethod",
            f"    def static_method() -> str:",
            '        """Static method."""',
            f'        return "{class_name}"',
            ""
        ])
        
        return lines
    
    def generate_circular_import_project(self, name: str = "circular_complex") -> Path:
        """Generate a project with complex circular imports."""
        project_path = self.base_path / name
        project_path.mkdir(exist_ok=True)
        
        # Create a web of interconnected modules
        modules = ["auth", "models", "views", "utils", "services", "handlers"]
        
        for i, module in enumerate(modules):
            content = f'"""Module {module} with circular dependencies."""\n\n'
            content += "from typing import TYPE_CHECKING, Optional\n\n"
            
            # TYPE_CHECKING imports
            content += "if TYPE_CHECKING:\n"
            for j, other in enumerate(modules):
                if i != j:
                    content += f"    from .{other} import {other.capitalize()}Class\n"
            content += "\n\n"
            
            # Main class
            content += f"class {module.capitalize()}Class:\n"
            content += f'    """Class for {module}."""\n\n'
            content += "    def __init__(self):\n"
            content += f'        self.name = "{module}"\n\n'
            
            # Methods that reference other modules
            for j, other in enumerate(modules):
                if i != j:
                    content += f"    def get_{other}(self) -> '{other.capitalize()}Class':\n"
                    content += f'        """Get {other} instance."""\n'
                    content += f"        from .{other} import {other.capitalize()}Class\n"
                    content += f"        return {other.capitalize()}Class()\n\n"
            
            (project_path / f"{module}.py").write_text(content)
        
        # Create __init__.py
        init_content = '"""Circular import test package."""\n\n'
        init_content += "__all__ = [\n"
        for module in modules:
            init_content += f'    "{module.capitalize()}Class",\n'
        init_content += "]\n\n"
        for module in modules:
            init_content += f"from .{module} import {module.capitalize()}Class\n"
        
        (project_path / "__init__.py").write_text(init_content)
        
        return project_path
    
    def generate_deeply_nested_project(self, name: str = "deeply_nested",
                                     depth: int = 10) -> Path:
        """Generate a project with deep nesting."""
        project_path = self.base_path / name
        project_path.mkdir(exist_ok=True)
        
        current_path = project_path
        for level in range(depth):
            package_name = f"level_{level}"
            current_path = current_path / package_name
            current_path.mkdir(exist_ok=True)
            
            # Create __init__.py
            (current_path / "__init__.py").write_text(
                f'"""Package at level {level}."""\n'
            )
            
            # Create module at this level
            module_content = f'"""Module at level {level}."""\n\n'
            
            # Import from parent levels
            if level > 0:
                for i in range(level):
                    parent_import = ".." * (level - i)
                    module_content += f"from {parent_import}level_{i}.module_{i} import Class_{i}\n"
                module_content += "\n"
            
            # Define class
            module_content += f"class Class_{level}:\n"
            module_content += f'    """Class at level {level}."""\n'
            module_content += f"    level = {level}\n\n"
            
            # Methods referencing parent classes
            if level > 0:
                module_content += "    def get_parents(self):\n"
                module_content += '        """Get parent class instances."""\n'
                module_content += "        return [\n"
                for i in range(level):
                    module_content += f"            Class_{i}(),\n"
                module_content += "        ]\n"
            
            (current_path / f"module_{level}.py").write_text(module_content)
        
        return project_path