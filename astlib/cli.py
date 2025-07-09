#!/usr/bin/env python3
"""Command-line interface for AST tools."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback for when tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable
    tqdm.write = print

from astlib.finder import FunctionFinder
from astlib.walker import DirectoryWalker
from astlib.indexer import ProjectIndex
from astlib.symbols import SymbolType


class CLIFormatter:
    """Handle different output formats for CLI results."""

    @staticmethod
    def format_plain(results: List[Dict[str, Any]]) -> str:
        """Format results as plain text."""
        if not results:
            return "No results found."
        
        output = []
        for result in results:
            output.append(f"{result['file']}:{result['line']}:{result['column']} - {result['name']}")
            if result.get('docstring'):
                # First line of docstring
                first_line = result['docstring'].split('\n')[0][:60]
                if len(result['docstring']) > 60:
                    first_line += "..."
                output.append(f"  {first_line}")
        
        return '\n'.join(output)

    @staticmethod
    def format_json(results: List[Dict[str, Any]]) -> str:
        """Format results as JSON."""
        return json.dumps(results, indent=2)

    @staticmethod
    def format_markdown(results: List[Dict[str, Any]]) -> str:
        """Format results as Markdown."""
        if not results:
            return "No results found."
        
        output = ["# Search Results\n"]
        for result in results:
            output.append(f"## `{result['name']}` in {result['file']}")
            output.append(f"- **Location**: Line {result['line']}, Column {result['column']}")
            if result.get('docstring'):
                output.append(f"- **Docstring**: {result['docstring'].split(chr(10))[0]}")
            output.append("")
        
        return '\n'.join(output)


class ASTCli:
    """Main CLI application for AST tools."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog='ast',
            description='AST tools for navigating Python codebases',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ast find-function main
  ast find-function "test_" --path ./tests --format json
  ast find-function parse --ignore "build/*" --ignore "dist/*"
            """
        )

        parser.add_argument(
            '--version',
            action='version',
            version='%(prog)s 0.1.0'
        )

        # Create subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            required=True
        )

        # find-function command
        find_func = subparsers.add_parser(
            'find-function',
            help='Find functions by name pattern',
            description='Search for function definitions in Python files'
        )
        find_func.add_argument(
            'name',
            help='Function name or pattern to search for'
        )
        find_func.add_argument(
            '-p', '--path',
            default='.',
            help='Directory to search in (default: current directory)'
        )
        find_func.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'markdown'],
            default='plain',
            help='Output format (default: plain)'
        )
        find_func.add_argument(
            '--ignore',
            action='append',
            default=[],
            help='Patterns to ignore (can be specified multiple times)'
        )
        find_func.add_argument(
            '--no-progress',
            action='store_true',
            help='Disable progress bar'
        )
        find_func.add_argument(
            '--case-sensitive',
            action='store_true',
            help='Use case-sensitive matching'
        )

        # Index command
        index_cmd = subparsers.add_parser(
            'index',
            help='Build project index for fast navigation',
            description='Index all Python files in a project'
        )
        index_cmd.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        index_cmd.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of parallel workers (default: 4)'
        )
        index_cmd.add_argument(
            '--exclude',
            action='append',
            default=[],
            help='Additional patterns to exclude'
        )
        index_cmd.add_argument(
            '--stats',
            action='store_true',
            help='Show detailed statistics after indexing'
        )

        # Find symbol command
        find_symbol = subparsers.add_parser(
            'find-symbol',
            help='Find symbol definition using index',
            description='Find where a symbol is defined in the project'
        )
        find_symbol.add_argument(
            'name',
            help='Symbol name to find'
        )
        find_symbol.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        find_symbol.add_argument(
            '--from-file',
            help='Context file for import resolution'
        )
        find_symbol.add_argument(
            '-t', '--type',
            choices=['function', 'class', 'method', 'variable', 'all'],
            default='all',
            help='Symbol type to search for'
        )
        find_symbol.add_argument(
            '-f', '--format',
            choices=['plain', 'json'],
            default='plain',
            help='Output format'
        )

        # Find references command
        find_refs = subparsers.add_parser(
            'find-references',
            help='Find all references to a symbol',
            description='Find where a symbol is used throughout the project'
        )
        find_refs.add_argument(
            'name',
            help='Symbol name to find references for'
        )
        find_refs.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        find_refs.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'markdown'],
            default='plain',
            help='Output format (default: plain)'
        )
        find_refs.add_argument(
            '--group-by-file',
            action='store_true',
            help='Group results by file'
        )
        find_refs.add_argument(
            '--no-progress',
            action='store_true',
            help='Disable progress bar'
        )

        # List symbols command
        list_symbols = subparsers.add_parser(
            'list-symbols',
            help='List all symbols in a file or project',
            description='List symbols with their types and locations'
        )
        list_symbols.add_argument(
            '-p', '--path',
            default='.',
            help='File or directory path (default: current directory)'
        )
        list_symbols.add_argument(
            '-t', '--type',
            choices=['function', 'class', 'method', 'variable', 'all'],
            default='all',
            help='Filter by symbol type'
        )
        list_symbols.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'tree'],
            default='plain',
            help='Output format'
        )

        # Show imports command
        show_imports = subparsers.add_parser(
            'show-imports',
            help='Show import dependencies',
            description='Display import graph and dependencies'
        )
        show_imports.add_argument(
            'module',
            nargs='?',
            help='Module to show imports for (optional)'
        )
        show_imports.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        show_imports.add_argument(
            '--circular',
            action='store_true',
            help='Show only circular imports'
        )
        show_imports.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'dot'],
            default='plain',
            help='Output format'
        )

        # Show callers command
        show_callers = subparsers.add_parser(
            'show-callers',
            help='Show functions that call a given function',
            description='Display all functions that call the specified function'
        )
        show_callers.add_argument(
            'function',
            help='Function name to find callers for'
        )
        show_callers.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        show_callers.add_argument(
            '--file',
            help='Specific file to search within'
        )
        show_callers.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'tree'],
            default='plain',
            help='Output format'
        )

        # Show callees command
        show_callees = subparsers.add_parser(
            'show-callees',
            help='Show functions called by a given function',
            description='Display all functions called by the specified function'
        )
        show_callees.add_argument(
            'function',
            help='Function name to find callees for'
        )
        show_callees.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        show_callees.add_argument(
            '--file',
            help='Specific file to search within'
        )
        show_callees.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'tree'],
            default='plain',
            help='Output format'
        )

        # Call chain command
        call_chain = subparsers.add_parser(
            'call-chain',
            help='Find call chains between two functions',
            description='Find all possible call paths from one function to another'
        )
        call_chain.add_argument(
            'start',
            help='Starting function name'
        )
        call_chain.add_argument(
            'end',
            help='Target function name'
        )
        call_chain.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        call_chain.add_argument(
            '--max-depth',
            type=int,
            default=10,
            help='Maximum search depth (default: 10)'
        )
        call_chain.add_argument(
            '-f', '--format',
            choices=['plain', 'json'],
            default='plain',
            help='Output format'
        )

        # Call hierarchy command
        call_hierarchy = subparsers.add_parser(
            'call-hierarchy',
            help='Show call hierarchy for a function',
            description='Display the complete call hierarchy (callers and callees) for a function'
        )
        call_hierarchy.add_argument(
            'function',
            help='Function name to analyze'
        )
        call_hierarchy.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        call_hierarchy.add_argument(
            '--direction',
            choices=['up', 'down', 'both'],
            default='both',
            help='Direction to traverse: up (callers), down (callees), or both'
        )
        call_hierarchy.add_argument(
            '--max-depth',
            type=int,
            default=5,
            help='Maximum depth to traverse (default: 5)'
        )
        call_hierarchy.add_argument(
            '-f', '--format',
            choices=['plain', 'json', 'tree'],
            default='tree',
            help='Output format'
        )

        # Call graph stats command
        call_stats = subparsers.add_parser(
            'call-stats',
            help='Show call graph statistics',
            description='Display statistics about the project\'s call graph'
        )
        call_stats.add_argument(
            '-p', '--path',
            default='.',
            help='Project root directory (default: current directory)'
        )
        call_stats.add_argument(
            '--export-dot',
            help='Export call graph to DOT file'
        )

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parsed_args = self.parser.parse_args(args)
        
        if parsed_args.command == 'find-function':
            return self._find_function(parsed_args)
        elif parsed_args.command == 'index':
            return self._index_project(parsed_args)
        elif parsed_args.command == 'find-symbol':
            return self._find_symbol(parsed_args)
        elif parsed_args.command == 'find-references':
            return self._find_references(parsed_args)
        elif parsed_args.command == 'list-symbols':
            return self._list_symbols(parsed_args)
        elif parsed_args.command == 'show-imports':
            return self._show_imports(parsed_args)
        elif parsed_args.command == 'show-callers':
            return self._show_callers(parsed_args)
        elif parsed_args.command == 'show-callees':
            return self._show_callees(parsed_args)
        elif parsed_args.command == 'call-chain':
            return self._call_chain(parsed_args)
        elif parsed_args.command == 'call-hierarchy':
            return self._call_hierarchy(parsed_args)
        elif parsed_args.command == 'call-stats':
            return self._call_stats(parsed_args)
        
        return 1

    def _find_function(self, args: argparse.Namespace) -> int:
        """Execute the find-function command."""
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
        
        if not path.is_dir():
            print(f"Error: Path '{path}' is not a directory", file=sys.stderr)
            return 1

        # Initialize components
        walker = DirectoryWalker(ignore_patterns=args.ignore)
        finder = FunctionFinder(case_sensitive=args.case_sensitive)
        formatter = CLIFormatter()

        # Collect Python files
        if not args.no_progress:
            print(f"Scanning directory: {path}", file=sys.stderr)
        
        python_files = list(walker.walk(path))
        
        if not python_files:
            print("No Python files found.", file=sys.stderr)
            return 0

        # Search for functions
        results = []
        
        # Create progress bar if enabled
        file_iterator = python_files
        if not args.no_progress:
            file_iterator = tqdm(
                python_files,
                desc="Searching files",
                unit="files",
                disable=args.no_progress
            )

        for file_path in file_iterator:
            try:
                file_results = finder.find_functions(file_path, args.name)
                results.extend(file_results)
            except Exception as e:
                if not args.no_progress:
                    tqdm.write(f"Warning: Error processing {file_path}: {e}", file=sys.stderr)

        # Format and output results
        if args.format == 'json':
            output = formatter.format_json(results)
        elif args.format == 'markdown':
            output = formatter.format_markdown(results)
        else:
            output = formatter.format_plain(results)
        
        print(output)
        
        return 0

    def _index_project(self, args: argparse.Namespace) -> int:
        """Execute the index command."""
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
        
        # Create index
        index = ProjectIndex(max_workers=args.workers)
        
        try:
            # Build index
            print(f"Indexing project: {path}")
            stats = index.build_index(path, exclude_patterns=args.exclude)
            
            # Show basic results
            print(f"\nIndexing complete:")
            print(f"  Files processed: {stats.indexed_files}/{stats.total_files}")
            print(f"  Total symbols: {stats.total_symbols}")
            print(f"  Time: {stats.indexing_time_ms:.1f}ms")
            
            # Show detailed stats if requested
            if args.stats:
                print("\nDetailed statistics:")
                all_stats = index.get_stats()
                print(json.dumps(all_stats, indent=2))
            
            return 0
            
        except Exception as e:
            print(f"Error: Failed to index project: {e}", file=sys.stderr)
            return 1

    def _find_symbol(self, args: argparse.Namespace) -> int:
        """Execute the find-symbol command."""
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
        
        # Build index
        index = ProjectIndex()
        print(f"Building index...", file=sys.stderr)
        index.build_index(path)
        
        # Find symbol
        from_file = Path(args.from_file).resolve() if args.from_file else None
        result = index.find_definition(args.name, from_file)
        
        if result:
            file_path, line_num = result
            if args.format == 'json':
                output = json.dumps({
                    'symbol': args.name,
                    'file': str(file_path),
                    'line': line_num
                })
            else:
                output = f"{file_path}:{line_num}: {args.name}"
            print(output)
            return 0
        else:
            print(f"Symbol '{args.name}' not found", file=sys.stderr)
            return 1

    def _find_references(self, args: argparse.Namespace) -> int:
        """Execute the find-references command."""
        from .api import ASTLib
        from .references import ReferenceType
        
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        try:
            # Use the unified API
            ast_lib = ASTLib(str(path), lazy=False)
            
            # Find references
            if not args.no_progress:
                print(f"Finding references to '{args.name}'...", file=sys.stderr)
                
            references = ast_lib.find_references(args.name)
            
            if not references:
                print(f"No references to '{args.name}' found", file=sys.stderr)
                return 1
                
            # Format output
            if args.format == 'json':
                output = json.dumps([{
                    'symbol': ref.symbol_name,
                    'file': ref.file_path,
                    'line': ref.line,
                    'column': ref.column,
                    'context': ref.context
                } for ref in references], indent=2)
                print(output)
                
            elif args.format == 'markdown':
                print(f"# References to `{args.name}`\n")
                
                if args.group_by_file:
                    # Group by file
                    by_file = {}
                    for ref in references:
                        if ref.file_path not in by_file:
                            by_file[ref.file_path] = []
                        by_file[ref.file_path].append(ref)
                        
                    for file_path, refs in sorted(by_file.items()):
                        print(f"## {file_path}\n")
                        for ref in sorted(refs, key=lambda r: r.line):
                            print(f"- Line {ref.line}: `{ref.context}`")
                        print()
                else:
                    for ref in references:
                        print(f"- {ref.file_path}:{ref.line} - `{ref.context}`")
                        
            else:  # plain format
                if args.group_by_file:
                    # Group by file
                    by_file = {}
                    for ref in references:
                        if ref.file_path not in by_file:
                            by_file[ref.file_path] = []
                        by_file[ref.file_path].append(ref)
                        
                    for file_path, refs in sorted(by_file.items()):
                        print(f"\n{file_path}:")
                        for ref in sorted(refs, key=lambda r: r.line):
                            print(f"  {ref.line}:{ref.column} - {ref.context}")
                else:
                    for ref in references:
                        print(f"{ref.file_path}:{ref.line}:{ref.column} - {ref.context}")
                        
            print(f"\nFound {len(references)} references", file=sys.stderr)
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    def _list_symbols(self, args: argparse.Namespace) -> int:
        """Execute the list-symbols command."""
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
        
        # Build index
        index = ProjectIndex()
        
        if path.is_file():
            # Index single file's directory
            print(f"Indexing file's directory...", file=sys.stderr)
            index.build_index(path.parent)
            symbols = index.find_symbols_in_file(path)
        else:
            # Index whole directory
            print(f"Building index...", file=sys.stderr)
            index.build_index(path)
            symbols = index.get_all_symbols()
        
        # Filter by type if requested
        if args.type != 'all':
            type_map = {
                'function': SymbolType.FUNCTION,
                'class': SymbolType.CLASS,
                'method': SymbolType.METHOD,
                'variable': SymbolType.VARIABLE
            }
            filter_type = type_map[args.type]
            symbols = [s for s in symbols if s.type == filter_type]
        
        # Format output
        if args.format == 'json':
            output = json.dumps([{
                'name': s.name,
                'type': s.type.value,
                'file': s.file_path,
                'line': s.position.line,
                'qualified_name': s.qualified_name
            } for s in symbols], indent=2)
        elif args.format == 'tree':
            # Group by file
            by_file = {}
            for s in symbols:
                if s.file_path not in by_file:
                    by_file[s.file_path] = []
                by_file[s.file_path].append(s)
            
            output_lines = []
            for file_path, file_symbols in sorted(by_file.items()):
                output_lines.append(f"\n{file_path}:")
                for s in sorted(file_symbols, key=lambda x: x.position.line):
                    indent = "  " * (s.qualified_name.count('.'))
                    output_lines.append(f"{indent}{s.name} [{s.type.value}] (line {s.position.line})")
            output = '\n'.join(output_lines)
        else:
            # Plain format
            output_lines = []
            for s in sorted(symbols, key=lambda x: (x.file_path, x.position.line)):
                output_lines.append(f"{s.file_path}:{s.position.line}: {s.qualified_name} [{s.type.value}]")
            output = '\n'.join(output_lines)
        
        print(output)
        return 0

    def _show_imports(self, args: argparse.Namespace) -> int:
        """Execute the show-imports command."""
        path = Path(args.path).resolve()
        
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
        
        # Build index
        index = ProjectIndex()
        print(f"Building index...", file=sys.stderr)
        index.build_index(path)
        
        graph = index.get_import_graph()
        
        if args.circular:
            # Show only circular imports
            cycles = graph.find_circular_imports()
            if args.format == 'json':
                output = json.dumps([{
                    'cycle': c.cycle,
                    'length': c.cycle_length
                } for c in cycles], indent=2)
            else:
                if cycles:
                    output_lines = ["Circular imports found:"]
                    for c in cycles:
                        output_lines.append(f"  {str(c)}")
                    output = '\n'.join(output_lines)
                else:
                    output = "No circular imports found."
            print(output)
        elif args.module:
            # Show imports for specific module
            if args.module in graph.nodes:
                node = graph.nodes[args.module]
                if args.format == 'json':
                    output = json.dumps({
                        'module': args.module,
                        'imports': list(node.imports),
                        'imported_by': list(node.imported_by)
                    }, indent=2)
                else:
                    output_lines = [f"Module: {args.module}"]
                    if node.imports:
                        output_lines.append(f"\nImports ({len(node.imports)}):")
                        for imp in sorted(node.imports):
                            output_lines.append(f"  - {imp}")
                    if node.imported_by:
                        output_lines.append(f"\nImported by ({len(node.imported_by)}):")
                        for imp in sorted(node.imported_by):
                            output_lines.append(f"  - {imp}")
                    output = '\n'.join(output_lines)
                print(output)
            else:
                print(f"Module '{args.module}' not found", file=sys.stderr)
                return 1
        else:
            # Show all imports summary
            if args.format == 'json':
                nodes_data = {}
                for name, node in graph.nodes.items():
                    nodes_data[name] = {
                        'imports': list(node.imports),
                        'imported_by': list(node.imported_by),
                        'is_external': node.is_external
                    }
                output = json.dumps(nodes_data, indent=2)
            elif args.format == 'dot':
                # Generate Graphviz DOT format
                output_lines = ["digraph imports {"]
                for name, node in graph.nodes.items():
                    if not node.is_external:
                        for imp in node.imports:
                            output_lines.append(f'  "{name}" -> "{imp}";')
                output_lines.append("}")
                output = '\n'.join(output_lines)
            else:
                # Summary
                internal_nodes = [n for n in graph.nodes.values() if not n.is_external]
                external_nodes = [n for n in graph.nodes.values() if n.is_external]
                
                output_lines = [
                    f"Import graph summary:",
                    f"  Total modules: {len(graph.nodes)}",
                    f"  Internal modules: {len(internal_nodes)}",
                    f"  External dependencies: {len(external_nodes)}"
                ]
                
                # Most imported modules
                most_imported = sorted(
                    [(n.module_name, n.imported_by_count) for n in internal_nodes],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                if most_imported:
                    output_lines.append("\nMost imported modules:")
                    for name, count in most_imported:
                        output_lines.append(f"  - {name}: {count} imports")
                
                output = '\n'.join(output_lines)
            
            print(output)
        
        return 0

    def _show_callers(self, args: argparse.Namespace) -> int:
        """Execute the show-callers command."""
        from astlib import ASTLib
        
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        ast_lib = ASTLib(str(path), lazy=False)
        callers = ast_lib.find_callers(args.function, args.file)
        
        if not callers:
            print(f"No callers found for '{args.function}'")
            return 0
            
        if args.format == 'json':
            data = [{
                'name': c.name,
                'qualified_name': c.qualified_name,
                'file': c.file_path,
                'line': c.line,
                'is_method': c.is_method,
                'class_name': c.class_name
            } for c in callers]
            print(json.dumps(data, indent=2))
        elif args.format == 'tree':
            print(f"Functions that call '{args.function}':")
            for caller in callers:
                prefix = "└─ " if caller == callers[-1] else "├─ "
                class_info = f" (method of {caller.class_name})" if caller.is_method else ""
                print(f"{prefix}{caller.name}{class_info}")
                print(f"   {caller.file_path}:{caller.line}")
        else:
            print(f"Callers of '{args.function}':")
            for caller in callers:
                class_info = f" ({caller.class_name}.)" if caller.is_method else ""
                print(f"  {class_info}{caller.name} - {caller.file_path}:{caller.line}")
                
        return 0
        
    def _show_callees(self, args: argparse.Namespace) -> int:
        """Execute the show-callees command."""
        from astlib import ASTLib
        
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        ast_lib = ASTLib(str(path), lazy=False)
        callees = ast_lib.find_callees(args.function, args.file)
        
        if not callees:
            print(f"No callees found for '{args.function}'")
            return 0
            
        if args.format == 'json':
            data = [{
                'name': c.name,
                'qualified_name': c.qualified_name,
                'file': c.file_path,
                'line': c.line,
                'is_method': c.is_method,
                'class_name': c.class_name
            } for c in callees]
            print(json.dumps(data, indent=2))
        elif args.format == 'tree':
            print(f"Functions called by '{args.function}':")
            for callee in callees:
                prefix = "└─ " if callee == callees[-1] else "├─ "
                class_info = f" (method of {callee.class_name})" if callee.is_method else ""
                print(f"{prefix}{callee.name}{class_info}")
                print(f"   {callee.file_path}:{callee.line}")
        else:
            print(f"Functions called by '{args.function}':")
            for callee in callees:
                class_info = f" ({callee.class_name}.)" if callee.is_method else ""
                print(f"  {class_info}{callee.name} - {callee.file_path}:{callee.line}")
                
        return 0
        
    def _call_chain(self, args: argparse.Namespace) -> int:
        """Execute the call-chain command."""
        from astlib import ASTLib
        
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        ast_lib = ASTLib(str(path), lazy=False)
        chains = ast_lib.find_call_chains(args.start, args.end, args.max_depth)
        
        if not chains:
            print(f"No call chains found from '{args.start}' to '{args.end}'")
            return 0
            
        if args.format == 'json':
            data = []
            for chain in chains:
                data.append({
                    'path': [n.name for n in chain.nodes],
                    'depth': chain.depth,
                    'contains_recursion': chain.contains_recursion
                })
            print(json.dumps(data, indent=2))
        else:
            print(f"Call chains from '{args.start}' to '{args.end}':")
            for i, chain in enumerate(chains, 1):
                print(f"\nChain {i} (depth: {chain.depth}):")
                print(f"  {chain}")
                if chain.contains_recursion:
                    print("  ⚠️  Contains recursion")
                    
        return 0
        
    def _call_hierarchy(self, args: argparse.Namespace) -> int:
        """Execute the call-hierarchy command."""
        from astlib import ASTLib
        
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        ast_lib = ASTLib(str(path), lazy=False)
        hierarchy = ast_lib.get_call_hierarchy(args.function, args.direction, args.max_depth)
        
        if not hierarchy:
            print(f"Function '{args.function}' not found")
            return 1
            
        if args.format == 'json':
            print(json.dumps(hierarchy, indent=2))
        elif args.format == 'tree':
            for func_name, data in hierarchy.items():
                print(f"\n{func_name}")
                print(f"  File: {data['file']}:{data['line']}")
                
                if 'called_by' in data and data['called_by']:
                    print("\n  Called by:")
                    self._print_hierarchy_tree(data['called_by'], indent=4)
                    
                if 'calls' in data and data['calls']:
                    print("\n  Calls:")
                    self._print_hierarchy_tree(data['calls'], indent=4)
        else:
            for func_name, data in hierarchy.items():
                print(f"\nFunction: {func_name}")
                print(f"Location: {data['file']}:{data['line']}")
                
                if 'called_by' in data and data['called_by']:
                    print("\nCalled by:")
                    for caller in data['called_by']:
                        print(f"  - {caller['name']} ({caller['file']}:{caller['line']})")
                        
                if 'calls' in data and data['calls']:
                    print("\nCalls:")
                    for callee in data['calls']:
                        print(f"  - {callee['name']} ({callee['file']}:{callee['line']})")
                        
        return 0
        
    def _print_hierarchy_tree(self, items: List[dict], indent: int = 0):
        """Helper to print hierarchy as tree."""
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            prefix = "└─ " if is_last else "├─ "
            print(f"{' ' * indent}{prefix}{item['name']}")
            
            # Print nested calls/callers
            if 'calls' in item and item['calls']:
                next_indent = indent + 4
                self._print_hierarchy_tree(item['calls'], next_indent)
            elif 'called_by' in item and item['called_by']:
                next_indent = indent + 4
                self._print_hierarchy_tree(item['called_by'], next_indent)
                
    def _call_stats(self, args: argparse.Namespace) -> int:
        """Execute the call-stats command."""
        from astlib import ASTLib
        
        path = Path(args.path).resolve()
        if not path.exists():
            print(f"Error: Path '{path}' does not exist", file=sys.stderr)
            return 1
            
        ast_lib = ASTLib(str(path), lazy=False)
        stats = ast_lib.get_call_graph_stats()
        
        print("Call Graph Statistics")
        print("=" * 50)
        print(f"Total functions: {stats.total_functions}")
        print(f"Total calls: {stats.total_calls}")
        print(f"Maximum call depth: {stats.max_call_depth}")
        
        if stats.recursive_functions:
            print(f"\nRecursive functions ({len(stats.recursive_functions)}):")
            for func in sorted(stats.recursive_functions):
                print(f"  - {func}")
                
        if stats.isolated_functions:
            print(f"\nIsolated functions ({len(stats.isolated_functions)}):")
            for func in sorted(list(stats.isolated_functions)[:10]):
                print(f"  - {func}")
            if len(stats.isolated_functions) > 10:
                print(f"  ... and {len(stats.isolated_functions) - 10} more")
                
        if stats.most_called_functions:
            print(f"\nMost called functions:")
            for func, count in stats.most_called_functions[:10]:
                print(f"  - {func}: {count} calls")
                
        # Export to DOT if requested
        if args.export_dot:
            dot_content = ast_lib.export_call_graph_dot()
            with open(args.export_dot, 'w') as f:
                f.write(dot_content)
            print(f"\nCall graph exported to: {args.export_dot}")
            
        return 0


def main():
    """Main entry point for the CLI."""
    cli = ASTCli()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()