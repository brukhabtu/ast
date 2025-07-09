#!/usr/bin/env python3
"""Integration tests for import resolution and dependency analysis."""

import pytest
from pathlib import Path
from textwrap import dedent

from astlib.imports import analyze_imports, find_all_imports
from astlib.import_graph import ImportGraph, build_import_graph, CircularImport


class TestImportResolution:
    """Test import resolution in real package structures."""
    
    def create_package_structure(self, tmp_path):
        """Create a test package structure."""
        # Create package directories
        pkg = tmp_path / "testpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        utils = pkg / "utils"
        utils.mkdir()
        (utils / "__init__.py").write_text("")
        
        core = pkg / "core"
        core.mkdir()
        (core / "__init__.py").write_text("")
        
        # Create modules with various imports
        
        # testpkg/main.py
        (pkg / "main.py").write_text(dedent("""
            import sys
            from pathlib import Path
            from .utils import helper
            from .core.engine import Engine
            
            def main():
                engine = Engine()
                helper.setup()
        """))
        
        # testpkg/utils/helper.py
        (utils / "helper.py").write_text(dedent("""
            import os
            from ..core import config
            from . import tools
            
            def setup():
                config.load()
                tools.init()
        """))
        
        # testpkg/utils/tools.py
        (utils / "tools.py").write_text(dedent("""
            import json
            from typing import Dict
            
            def init():
                pass
        """))
        
        # testpkg/core/engine.py
        (core / "engine.py").write_text(dedent("""
            from typing import Optional
            from ..utils import helper
            from . import config
            
            class Engine:
                def __init__(self):
                    self.config = config.get_config()
        """))
        
        # testpkg/core/config.py
        (core / "config.py").write_text(dedent("""
            import os
            from pathlib import Path
            
            _config = {}
            
            def load():
                pass
                
            def get_config():
                return _config
        """))
        
        return pkg
        
    def test_find_all_imports(self, tmp_path):
        """Test finding all imports in a package."""
        pkg = self.create_package_structure(tmp_path)
        
        all_imports = find_all_imports(pkg)
        
        # Should find all Python files
        assert len(all_imports) >= 5  # main, helper, tools, engine, config
        
        # Check main.py imports
        main_imports = all_imports.get("testpkg.main")
        assert main_imports is not None
        assert "sys" in main_imports.imported_modules
        assert "pathlib" in main_imports.imported_modules
        # from .utils import helper -> imports from testpkg.utils module
        assert "testpkg.utils" in main_imports.imported_modules
        assert "testpkg.core.engine" in main_imports.imported_modules
        
        # Check relative import resolution
        helper_imports = all_imports.get("testpkg.utils.helper")
        assert helper_imports is not None
        # from ..core import config -> imports from testpkg.core module
        assert "testpkg.core" in helper_imports.imported_modules
        # from . import tools -> resolves to testpkg.utils.tools
        assert "testpkg.utils.tools" in helper_imports.imported_modules
        
    def test_import_graph_construction(self, tmp_path):
        """Test building import dependency graph."""
        pkg = self.create_package_structure(tmp_path)
        
        graph = build_import_graph(pkg)
        
        # Check nodes exist
        assert "testpkg.main" in graph.nodes
        assert "testpkg.utils.helper" in graph.nodes
        assert "testpkg.core.engine" in graph.nodes
        
        # Check dependencies
        main_deps = graph.get_dependencies("testpkg.main")
        assert "testpkg.utils" in main_deps  # from .utils import helper
        assert "testpkg.core.engine" in main_deps
        assert "sys" in main_deps
        
        # Check reverse dependencies
        utils_deps = graph.get_dependents("testpkg.utils")  # utils module is imported
        assert "testpkg.main" in utils_deps
        
        # Core is imported by both helper and engine
        core_deps = graph.get_dependents("testpkg.core")
        assert "testpkg.utils.helper" in core_deps
        
    def test_transitive_dependencies(self, tmp_path):
        """Test getting transitive dependencies."""
        pkg = self.create_package_structure(tmp_path)
        graph = build_import_graph(pkg)
        
        # Get all dependencies of main (including transitive)
        all_deps = graph.get_dependencies("testpkg.main", recursive=True)
        
        # Should include direct and indirect dependencies
        assert "testpkg.utils" in all_deps         # direct (from .utils import helper)
        assert "testpkg.core.engine" in all_deps   # direct
        assert "testpkg.core.config" in all_deps   # indirect via engine
        assert "typing" in all_deps                # indirect via engine
        
        # Note: testpkg.core is NOT a transitive dependency because:
        # - main imports from testpkg.utils (not testpkg.utils.helper)
        # - testpkg.utils has no imports
        # - So there's no path from main to testpkg.utils.helper
        
    def test_import_levels(self, tmp_path):
        """Test organizing modules by import levels."""
        pkg = self.create_package_structure(tmp_path)
        graph = build_import_graph(pkg)
        
        levels = graph.get_import_levels()
        
        # Level 0 should have modules with no internal dependencies
        assert "testpkg.utils.tools" in levels[0]
        assert "testpkg.core.config" in levels[0]
        
        # Higher levels depend on lower levels
        # (exact levels depend on import structure)
        all_modules = []
        for level_modules in levels.values():
            all_modules.extend(level_modules)
            
        # All internal modules should be assigned a level
        internal_modules = [
            name for name, node in graph.nodes.items() 
            if not node.is_external
        ]
        for mod in internal_modules:
            assert mod in all_modules


class TestCircularImports:
    """Test circular import detection."""
    
    def create_circular_package(self, tmp_path):
        """Create a package with circular imports."""
        pkg = tmp_path / "circular"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        # Create circular import: a -> b -> c -> a
        (pkg / "a.py").write_text(dedent("""
            from .b import func_b
            
            def func_a():
                return func_b()
        """))
        
        (pkg / "b.py").write_text(dedent("""
            from .c import func_c
            
            def func_b():
                return func_c()
        """))
        
        (pkg / "c.py").write_text(dedent("""
            from .a import func_a
            
            def func_c():
                return func_a()
        """))
        
        # Create another cycle: x <-> y
        (pkg / "x.py").write_text(dedent("""
            from .y import func_y
            
            def func_x():
                return func_y()
        """))
        
        (pkg / "y.py").write_text(dedent("""
            from .x import func_x
            
            def func_y():
                return func_x()
        """))
        
        return pkg
        
    def test_detect_circular_imports(self, tmp_path):
        """Test detecting circular imports."""
        pkg = self.create_circular_package(tmp_path)
        graph = build_import_graph(pkg)
        
        cycles = graph.find_circular_imports()
        
        # Should find both cycles
        assert len(cycles) >= 2
        
        # Check cycle contents
        cycle_modules = [set(cycle.cycle) for cycle in cycles]
        
        # Should have a->b->c->a cycle
        abc_cycle = {"circular.a", "circular.b", "circular.c"}
        assert any(abc_cycle.issubset(modules) for modules in cycle_modules)
        
        # Should have x<->y cycle
        xy_cycle = {"circular.x", "circular.y"}
        assert any(xy_cycle == modules for modules in cycle_modules)
        
    def test_circular_import_string_representation(self):
        """Test CircularImport string representation."""
        cycle = CircularImport(cycle=["a", "b", "c"])
        assert str(cycle) == "a -> b -> c -> a"
        
    def test_no_circular_imports(self, tmp_path):
        """Test graph with no circular imports."""
        pkg = tmp_path / "clean"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        # Create clean hierarchy: base <- mid <- top
        (pkg / "base.py").write_text("# No imports")
        
        (pkg / "mid.py").write_text(dedent("""
            from . import base
            
            def use_base():
                pass
        """))
        
        (pkg / "top.py").write_text(dedent("""
            from . import mid
            from . import base
            
            def use_all():
                pass
        """))
        
        graph = build_import_graph(pkg)
        cycles = graph.find_circular_imports()
        
        assert len(cycles) == 0


class TestImportGraphStats:
    """Test import graph statistics."""
    
    def test_graph_statistics(self, tmp_path):
        """Test getting graph statistics."""
        # Create simple package
        pkg = tmp_path / "stats"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        (pkg / "a.py").write_text(dedent("""
            import os
            import sys
            from pathlib import Path
        """))
        
        (pkg / "b.py").write_text(dedent("""
            from . import a
            import json
        """))
        
        (pkg / "c.py").write_text(dedent("""
            from . import a
            from . import b
        """))
        
        graph = build_import_graph(pkg)
        stats = graph.get_stats()
        
        assert stats['internal_modules'] >= 3  # a, b, c
        assert stats['external_modules'] >= 4  # os, sys, pathlib, json
        assert stats['total_modules'] >= 7
        assert stats['avg_imports_per_module'] > 0
        assert stats['circular_import_count'] == 0
        
    def test_empty_graph_statistics(self):
        """Test statistics for empty graph."""
        graph = ImportGraph()
        stats = graph.get_stats()
        
        assert stats['total_modules'] == 0
        assert stats['internal_modules'] == 0
        assert stats['external_modules'] == 0
        assert stats['avg_imports_per_module'] == 0.0


class TestEdgeCases:
    """Test edge cases and special import scenarios."""
    
    def test_dynamic_imports(self, tmp_path):
        """Test handling of dynamic imports."""
        pkg = tmp_path / "dynamic"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        # Dynamic imports are not captured by static analysis
        (pkg / "dynamic.py").write_text(dedent("""
            import importlib
            
            def load_module(name):
                return importlib.import_module(name)
                
            # These won't be detected
            mod = __import__('os')
            exec('import sys')
        """))
        
        all_imports = find_all_imports(pkg)
        dynamic_imports = all_imports.get("dynamic.dynamic")
        
        # Should only find static import
        assert dynamic_imports is not None
        assert "importlib" in dynamic_imports.imported_modules
        # Dynamic imports not detected
        assert "os" not in dynamic_imports.imported_modules
        assert "sys" not in dynamic_imports.imported_modules
        
    def test_conditional_imports(self, tmp_path):
        """Test handling of conditional imports."""
        pkg = tmp_path / "conditional"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        (pkg / "conditional.py").write_text(dedent("""
            import os
            
            if os.name == 'nt':
                import winreg
            else:
                import pwd
                
            try:
                import numpy
            except ImportError:
                numpy = None
        """))
        
        all_imports = find_all_imports(pkg)
        cond_imports = all_imports.get("conditional.conditional")
        
        # All imports are detected regardless of conditions
        imported = cond_imports.imported_modules
        assert "os" in imported
        assert "winreg" in imported
        assert "pwd" in imported
        assert "numpy" in imported
        
    def test_future_imports(self, tmp_path):
        """Test handling of __future__ imports."""
        pkg = tmp_path / "future"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        
        (pkg / "future.py").write_text(dedent("""
            from __future__ import annotations
            from __future__ import print_function
            
            import os
            from typing import List
        """))
        
        all_imports = find_all_imports(pkg)
        future_imports = all_imports.get("future.future")
        
        # Should capture all imports including __future__
        imported = future_imports.imported_modules
        assert "__future__" in imported
        assert "os" in imported
        assert "typing" in imported