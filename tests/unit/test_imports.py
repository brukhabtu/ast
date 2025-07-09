#!/usr/bin/env python3
"""Unit tests for import analysis functionality."""

import ast
import pytest
from pathlib import Path
from textwrap import dedent

from astlib.imports import (
    ImportType,
    ImportInfo,
    ModuleImports,
    ImportVisitor,
    extract_imports,
    analyze_imports,
    _infer_module_name
)


class TestImportInfo:
    """Test ImportInfo dataclass."""
    
    def test_absolute_import(self):
        """Test absolute import info."""
        imp = ImportInfo(
            module="os.path",
            names=[("join", "path_join")],
            import_type=ImportType.ABSOLUTE,
            level=0,
            line=1,
            column=0
        )
        
        assert imp.imported_names == ["path_join"]
        assert imp.resolve_relative("mypackage.module") == "os.path"
        
    def test_relative_import_single_dot(self):
        """Test relative import with single dot."""
        imp = ImportInfo(
            module="utils",
            names=[("helper", None)],
            import_type=ImportType.RELATIVE,
            level=1,
            line=1,
            column=0
        )
        
        assert imp.imported_names == ["helper"]
        assert imp.resolve_relative("mypackage.submodule") == "mypackage.utils"
        assert imp.resolve_relative("mypackage") == "utils"
        
    def test_relative_import_double_dot(self):
        """Test relative import with double dots."""
        imp = ImportInfo(
            module="sibling",
            names=[("func", None)],
            import_type=ImportType.RELATIVE,
            level=2,
            line=1,
            column=0
        )
        
        assert imp.resolve_relative("pkg.sub.module") == "pkg.sibling"
        assert imp.resolve_relative("pkg.module") == "sibling"
        assert imp.resolve_relative("module") is None  # Too many dots
        
    def test_relative_import_no_module(self):
        """Test relative import without module name."""
        imp = ImportInfo(
            module=None,
            names=[("sibling", None)],
            import_type=ImportType.RELATIVE,
            level=1,
            line=1,
            column=0
        )
        
        assert imp.resolve_relative("pkg.module") == "pkg"
        
    def test_star_import(self):
        """Test star import."""
        imp = ImportInfo(
            module="os",
            names=[("*", None)],
            import_type=ImportType.STAR,
            level=0,
            line=1,
            column=0,
            is_star=True
        )
        
        assert imp.is_star
        assert imp.imported_names == ["*"]


class TestImportVisitor:
    """Test ImportVisitor AST visitor."""
    
    def test_simple_import(self):
        """Test parsing simple import statement."""
        code = "import os"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module is None
        assert imp.names == [("os", None)]
        assert imp.import_type == ImportType.ABSOLUTE
        assert imp.level == 0
        
    def test_import_with_alias(self):
        """Test import with alias."""
        code = "import numpy as np"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.names == [("numpy", "np")]
        
    def test_multiple_imports(self):
        """Test multiple imports in one statement."""
        code = "import os, sys, json"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Each import should be separate
        assert len(visitor.imports) == 3
        modules = [imp.names[0][0] for imp in visitor.imports]
        assert modules == ["os", "sys", "json"]
        
    def test_from_import(self):
        """Test from-import statement."""
        code = "from os.path import join, dirname"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module == "os.path"
        assert imp.names == [("join", None), ("dirname", None)]
        assert imp.import_type == ImportType.ABSOLUTE
        
    def test_from_import_with_alias(self):
        """Test from-import with aliases."""
        code = "from collections import defaultdict as dd, Counter as C"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module == "collections"
        assert imp.names == [("defaultdict", "dd"), ("Counter", "C")]
        
    def test_relative_import(self):
        """Test relative imports."""
        code = "from . import utils"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module is None
        assert imp.names == [("utils", None)]
        assert imp.import_type == ImportType.RELATIVE
        assert imp.level == 1
        
    def test_relative_import_with_module(self):
        """Test relative import with module."""
        code = "from ..package import module"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module == "package"
        assert imp.level == 2
        assert imp.import_type == ImportType.RELATIVE
        
    def test_star_import(self):
        """Test star import."""
        code = "from os import *"
        tree = ast.parse(code)
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        assert len(visitor.imports) == 1
        imp = visitor.imports[0]
        assert imp.module == "os"
        assert imp.is_star
        assert imp.import_type == ImportType.STAR
        
    def test_mixed_imports(self):
        """Test file with various import types."""
        code = dedent("""
            import os
            import sys as system
            from pathlib import Path
            from collections import defaultdict as dd
            from . import utils
            from ..parent import helper
            from os.path import *
        """)
        
        tree = ast.parse(code)
        imports = extract_imports(tree)
        
        assert len(imports) == 7
        
        # Check each import type
        assert imports[0].module is None  # import os
        assert imports[1].names == [("sys", "system")]  # import sys as system
        assert imports[2].module == "pathlib"  # from pathlib import Path
        assert imports[3].names == [("defaultdict", "dd")]  # alias
        assert imports[4].import_type == ImportType.RELATIVE  # from . import
        assert imports[5].level == 2  # from .. import
        assert imports[6].is_star  # from x import *


class TestModuleImports:
    """Test ModuleImports class."""
    
    def test_imported_modules_absolute(self):
        """Test getting imported modules for absolute imports."""
        imports = [
            ImportInfo(
                module=None,
                names=[("os", None)],
                import_type=ImportType.ABSOLUTE,
                level=0,
                line=1,
                column=0
            ),
            ImportInfo(
                module="collections",
                names=[("defaultdict", None)],
                import_type=ImportType.ABSOLUTE,
                level=0,
                line=2,
                column=0
            )
        ]
        
        mod_imports = ModuleImports(
            module_name="mymodule",
            file_path=Path("mymodule.py"),
            imports=imports
        )
        
        assert mod_imports.imported_modules == {"os", "collections"}
        
    def test_imported_modules_relative(self):
        """Test getting imported modules for relative imports."""
        imports = [
            ImportInfo(
                module="utils",
                names=[("helper", None)],
                import_type=ImportType.RELATIVE,
                level=1,
                line=1,
                column=0
            ),
            ImportInfo(
                module="common",
                names=[("base", None)],
                import_type=ImportType.RELATIVE,
                level=2,
                line=2,
                column=0
            )
        ]
        
        mod_imports = ModuleImports(
            module_name="pkg.sub.module",
            file_path=Path("pkg/sub/module.py"),
            imports=imports
        )
        
        # Should resolve relative imports
        assert mod_imports.imported_modules == {"pkg.sub.utils", "pkg.common"}
        
    def test_has_circular_potential(self):
        """Test circular import detection potential."""
        imports = [
            ImportInfo(
                module="pkg.other",
                names=[("func", None)],
                import_type=ImportType.ABSOLUTE,
                level=0,
                line=1,
                column=0
            ),
            ImportInfo(
                module="external",
                names=[("tool", None)],
                import_type=ImportType.ABSOLUTE,
                level=0,
                line=2,
                column=0
            )
        ]
        
        mod_imports = ModuleImports(
            module_name="pkg.module",
            file_path=Path("pkg/module.py"),
            imports=imports
        )
        
        # Imports from same package - potential circular
        assert mod_imports.has_circular_potential
        
        # Module without package shouldn't have circular potential
        mod_imports2 = ModuleImports(
            module_name="standalone",
            file_path=Path("standalone.py"),
            imports=imports
        )
        assert not mod_imports2.has_circular_potential


class TestAnalyzeImports:
    """Test analyze_imports function."""
    
    def test_analyze_simple_file(self, tmp_path):
        """Test analyzing a simple Python file."""
        code = dedent("""
            import os
            from pathlib import Path
            
            def main():
                print(Path.cwd())
        """)
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = analyze_imports(file_path, "test")
        
        assert result.module_name == "test"
        assert len(result.imports) == 2
        assert result.imported_modules == {"os", "pathlib"}
        
    def test_analyze_with_syntax_error(self, tmp_path):
        """Test analyzing file with syntax error."""
        code = "import os\nthis is not valid python"
        
        file_path = tmp_path / "bad.py"
        file_path.write_text(code)
        
        with pytest.raises(SyntaxError):
            analyze_imports(file_path)
            
    def test_analyze_nonexistent_file(self):
        """Test analyzing non-existent file."""
        with pytest.raises(FileNotFoundError):
            analyze_imports("/does/not/exist.py")


class TestInferModuleName:
    """Test module name inference."""
    
    def test_simple_module(self, tmp_path):
        """Test inferring name for simple module."""
        file_path = tmp_path / "module.py"
        file_path.touch()
        
        assert _infer_module_name(file_path) == "module"
        
    def test_package_module(self, tmp_path):
        """Test inferring name for module in package."""
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()
        
        module_path = pkg_dir / "module.py"
        module_path.touch()
        
        assert _infer_module_name(module_path) == "mypackage.module"
        
    def test_nested_package(self, tmp_path):
        """Test inferring name for nested package."""
        pkg_dir = tmp_path / "pkg"
        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir(parents=True)
        
        (pkg_dir / "__init__.py").touch()
        (sub_dir / "__init__.py").touch()
        
        module_path = sub_dir / "module.py"
        module_path.touch()
        
        assert _infer_module_name(module_path) == "pkg.sub.module"
        
    def test_package_init(self, tmp_path):
        """Test inferring name for __init__.py."""
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        
        init_path = pkg_dir / "__init__.py"
        init_path.touch()
        
        assert _infer_module_name(init_path) == "mypackage"