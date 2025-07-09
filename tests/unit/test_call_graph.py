"""
Unit tests for call graph functionality.
"""

import pytest
from typing import List

from astlib.call_graph import (
    CallNode, CallEdge, CallChain, CallGraph, CallGraphStats,
    build_call_graph, _find_containing_function, _analyze_file_calls
)
from astlib.symbols import Symbol, SymbolType, Position
from astlib.parser import parse_file


class TestCallNode:
    """Test CallNode class."""
    
    def test_call_node_equality(self):
        """Test that CallNodes are equal based on qualified name and file."""
        node1 = CallNode(
            name="func",
            qualified_name="module.func",
            file_path="/path/file.py",
            line=10
        )
        node2 = CallNode(
            name="func",
            qualified_name="module.func",
            file_path="/path/file.py",
            line=20  # Different line
        )
        node3 = CallNode(
            name="func",
            qualified_name="other.func",
            file_path="/path/file.py",
            line=10
        )
        
        assert node1 == node2
        assert node1 != node3
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)
        
    def test_call_node_method_info(self):
        """Test CallNode with method information."""
        node = CallNode(
            name="method",
            qualified_name="MyClass.method",
            file_path="/path/file.py",
            line=15,
            is_method=True,
            class_name="MyClass"
        )
        
        assert node.is_method
        assert node.class_name == "MyClass"


class TestCallEdge:
    """Test CallEdge class."""
    
    def test_recursive_edge_detection(self):
        """Test detection of recursive calls."""
        node = CallNode("func", "func", "/file.py", 10)
        
        edge1 = CallEdge(caller=node, callee=node, line=15)
        assert edge1.is_recursive
        
        other_node = CallNode("other", "other", "/file.py", 20)
        edge2 = CallEdge(caller=node, callee=other_node, line=15)
        assert not edge2.is_recursive
        
    def test_call_type(self):
        """Test different call types."""
        caller = CallNode("func1", "func1", "/file.py", 10)
        callee = CallNode("func2", "func2", "/file.py", 20)
        
        edge = CallEdge(
            caller=caller,
            callee=callee,
            line=15,
            call_type="method"
        )
        
        assert edge.call_type == "method"


class TestCallChain:
    """Test CallChain class."""
    
    def test_chain_string_representation(self):
        """Test string representation of call chain."""
        nodes = [
            CallNode("main", "main", "/file.py", 1),
            CallNode("process", "process", "/file.py", 10),
            CallNode("save", "save", "/file.py", 20)
        ]
        
        chain = CallChain(nodes=nodes)
        assert str(chain) == "main → process → save"
        
    def test_chain_depth(self):
        """Test chain depth calculation."""
        nodes = [
            CallNode("a", "a", "/file.py", 1),
            CallNode("b", "b", "/file.py", 2),
            CallNode("c", "c", "/file.py", 3)
        ]
        
        chain = CallChain(nodes=nodes)
        assert chain.depth == 3
        
    def test_recursion_detection(self):
        """Test detection of recursion in chain."""
        node1 = CallNode("func1", "func1", "/file.py", 1)
        node2 = CallNode("func2", "func2", "/file.py", 2)
        
        # Chain without recursion
        chain1 = CallChain(nodes=[node1, node2])
        assert not chain1.contains_recursion
        
        # Chain with recursion
        chain2 = CallChain(nodes=[node1, node2, node1])
        assert chain2.contains_recursion


class TestCallGraph:
    """Test CallGraph class."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple call graph for testing."""
        graph = CallGraph()
        
        # Add nodes
        main = CallNode("main", "main", "/main.py", 1)
        func1 = CallNode("func1", "func1", "/utils.py", 10)
        func2 = CallNode("func2", "func2", "/utils.py", 20)
        helper = CallNode("helper", "helper", "/helpers.py", 5)
        
        graph.add_node(main)
        graph.add_node(func1)
        graph.add_node(func2)
        graph.add_node(helper)
        
        # Add edges
        graph.add_edge(CallEdge(main, func1, 5))
        graph.add_edge(CallEdge(main, func2, 7))
        graph.add_edge(CallEdge(func1, helper, 15))
        graph.add_edge(CallEdge(func2, helper, 25))
        
        return graph
        
    def test_add_nodes_and_edges(self, simple_graph):
        """Test adding nodes and edges to graph."""
        assert len(simple_graph.nodes) == 4
        assert len(simple_graph.edges) == 4
        
    def test_get_callers(self, simple_graph):
        """Test finding callers of a function."""
        # Helper is called by func1 and func2
        callers = simple_graph.get_callers("helper")
        caller_names = {c.name for c in callers}
        assert caller_names == {"func1", "func2"}
        
        # Main has no callers
        callers = simple_graph.get_callers("main")
        assert len(callers) == 0
        
    def test_get_callees(self, simple_graph):
        """Test finding callees of a function."""
        # Main calls func1 and func2
        callees = simple_graph.get_callees("main")
        callee_names = {c.name for c in callees}
        assert callee_names == {"func1", "func2"}
        
        # Helper calls nothing
        callees = simple_graph.get_callees("helper")
        assert len(callees) == 0
        
    def test_find_call_chains(self, simple_graph):
        """Test finding call chains between functions."""
        chains = simple_graph.find_call_chains("main", "helper")
        
        # Should find two chains: main -> func1 -> helper and main -> func2 -> helper
        assert len(chains) == 2
        
        chain_paths = []
        for chain in chains:
            path = [n.name for n in chain.nodes]
            chain_paths.append(path)
            
        assert ["main", "func1", "helper"] in chain_paths
        assert ["main", "func2", "helper"] in chain_paths
        
    def test_find_recursive_functions(self):
        """Test finding recursive functions."""
        graph = CallGraph()
        
        # Add recursive function
        fib = CallNode("fibonacci", "fibonacci", "/math.py", 10)
        graph.add_node(fib)
        graph.add_edge(CallEdge(fib, fib, 15))  # Direct recursion
        
        # Add indirectly recursive functions
        func_a = CallNode("func_a", "func_a", "/recurse.py", 1)
        func_b = CallNode("func_b", "func_b", "/recurse.py", 10)
        graph.add_node(func_a)
        graph.add_node(func_b)
        graph.add_edge(CallEdge(func_a, func_b, 5))
        graph.add_edge(CallEdge(func_b, func_a, 15))
        
        recursive = graph.find_recursive_functions()
        recursive_names = {f.name for f in recursive}
        
        assert "fibonacci" in recursive_names
        assert "func_a" in recursive_names
        assert "func_b" in recursive_names
        
    def test_get_call_hierarchy(self, simple_graph):
        """Test getting call hierarchy."""
        # Test downward hierarchy (callees)
        hierarchy = simple_graph.get_call_hierarchy("main", direction="down", max_depth=2)
        
        assert "main" in hierarchy
        main_data = hierarchy["main"]
        assert len(main_data["calls"]) == 2
        
        # Check that callees have their own callees
        for callee in main_data["calls"]:
            if callee["name"] in ("func1", "func2"):
                assert len(callee["calls"]) == 1
                assert callee["calls"][0]["name"] == "helper"
                
    def test_get_stats(self, simple_graph):
        """Test getting call graph statistics."""
        # Add a recursive function
        recursive = CallNode("recursive", "recursive", "/test.py", 1)
        simple_graph.add_node(recursive)
        simple_graph.add_edge(CallEdge(recursive, recursive, 5))
        
        # Add an isolated function
        isolated = CallNode("isolated", "isolated", "/test.py", 20)
        simple_graph.add_node(isolated)
        
        stats = simple_graph.get_stats()
        
        assert stats.total_functions == 6
        assert stats.total_calls == 5
        assert "recursive" in stats.recursive_functions
        assert "isolated" in stats.isolated_functions
        
        # Check most called functions
        most_called = dict(stats.most_called_functions)
        assert most_called.get("helper") == 2
        
    def test_export_dot(self, simple_graph):
        """Test exporting to DOT format."""
        dot = simple_graph.export_dot()
        
        assert "digraph CallGraph" in dot
        assert '"main"' in dot
        assert '"func1"' in dot
        assert '"helper"' in dot
        assert '"main" -> "func1"' in dot
        assert '"func1" -> "helper"' in dot


class TestCallGraphBuilder:
    """Test call graph building functions."""
    
    def test_find_containing_function(self, tmp_path):
        """Test finding the function containing a node."""
        code = '''
def outer():
    def inner():
        print("hello")
    inner()
    
class MyClass:
    def method(self):
        other_func()
'''
        test_file = tmp_path / "test.py"
        test_file.write_text(code)
        
        result = parse_file(str(test_file))
        assert result.tree
        
        # Find the print call
        import ast
        for node in ast.walk(result.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    func = _find_containing_function(node, result.tree)
                    assert func is not None
                    assert func.name == "inner"
                elif node.func.id == "other_func":
                    func = _find_containing_function(node, result.tree)
                    assert func is not None
                    assert func.name == "method"
                    
    def test_analyze_file_calls(self, tmp_path):
        """Test analyzing calls in a file."""
        # Create test files
        main_file = tmp_path / "main.py"
        main_file.write_text('''
def main():
    process_data()
    save_results()
    
def process_data():
    helper()
    
def helper():
    pass
    
def save_results():
    helper()
''')
        
        # Create a simple project index mock
        class MockIndex:
            def find_definition(self, name, from_file):
                # Simple mock that returns definitions in same file
                definitions = {
                    "process_data": type('obj', (object,), {
                        'file_path': str(main_file),
                        'line': 6
                    }),
                    "save_results": type('obj', (object,), {
                        'file_path': str(main_file),
                        'line': 12
                    }),
                    "helper": type('obj', (object,), {
                        'file_path': str(main_file),
                        'line': 9
                    })
                }
                return definitions.get(name)
                
        graph = CallGraph()
        
        # Add nodes first
        for name, line in [("main", 2), ("process_data", 6), ("helper", 9), ("save_results", 12)]:
            node = CallNode(name, name, str(main_file), line)
            graph.add_node(node)
            
        # Analyze calls
        _analyze_file_calls(str(main_file), graph, MockIndex())
        
        # Check edges were added
        assert len(graph.edges) > 0
        
        # Check main calls process_data and save_results
        main_callees = graph.get_callees("main")
        main_callee_names = {c.name for c in main_callees}
        assert "process_data" in main_callee_names
        assert "save_results" in main_callee_names


class TestCallGraphStats:
    """Test CallGraphStats class."""
    
    def test_stats_initialization(self):
        """Test stats object initialization."""
        stats = CallGraphStats()
        
        assert stats.total_functions == 0
        assert stats.total_calls == 0
        assert len(stats.recursive_functions) == 0
        assert stats.max_call_depth == 0
        assert len(stats.most_called_functions) == 0
        assert len(stats.isolated_functions) == 0
        
    def test_stats_with_data(self):
        """Test stats with actual data."""
        stats = CallGraphStats(
            total_functions=10,
            total_calls=25,
            recursive_functions={"fib", "factorial"},
            max_call_depth=5,
            most_called_functions=[("helper", 10), ("util", 5)],
            isolated_functions={"unused1", "unused2"}
        )
        
        assert stats.total_functions == 10
        assert stats.total_calls == 25
        assert len(stats.recursive_functions) == 2
        assert stats.max_call_depth == 5
        assert len(stats.most_called_functions) == 2
        assert len(stats.isolated_functions) == 2