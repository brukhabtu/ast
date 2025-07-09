#!/usr/bin/env python3
"""
Analyze validation results and update statistics

This script processes the validation results and updates all relevant
documentation with the latest statistics and insights.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

class ValidationAnalyzer:
    """Analyzes validation results and generates insights"""
    
    def __init__(self, validation_dir: Path):
        self.validation_dir = validation_dir
        self.results_file = validation_dir / "validation_results.json"
        self.stats_file = validation_dir / "statistics_report.json"
        self.matrix_file = validation_dir / "compatibility_matrix.json"
        
    def load_results(self) -> Tuple[List[Dict], Dict, Dict]:
        """Load all result files"""
        results = []
        stats = {}
        matrix = {}
        
        if self.results_file.exists():
            with open(self.results_file) as f:
                results = json.load(f)
        
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                stats = json.load(f)
        
        if self.matrix_file.exists():
            with open(self.matrix_file) as f:
                matrix = json.load(f)
        
        return results, stats, matrix
    
    def analyze_error_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze error patterns across all repositories"""
        error_patterns = defaultdict(lambda: {
            "count": 0,
            "repositories": set(),
            "examples": []
        })
        
        for result in results:
            for error in result.get("errors", []):
                error_type = error.get("error_type", "Unknown")
                error_msg = error.get("error_message", "")
                
                # Categorize error
                pattern = self._categorize_error_pattern(error_type, error_msg)
                
                error_patterns[pattern]["count"] += 1
                error_patterns[pattern]["repositories"].add(result["repo_name"])
                
                # Keep up to 5 examples
                if len(error_patterns[pattern]["examples"]) < 5:
                    error_patterns[pattern]["examples"].append({
                        "repo": result["repo_name"],
                        "file": error.get("file", "unknown"),
                        "message": error_msg[:200]  # Truncate long messages
                    })
        
        # Convert sets to lists for JSON serialization
        for pattern in error_patterns:
            error_patterns[pattern]["repositories"] = list(
                error_patterns[pattern]["repositories"]
            )
        
        return dict(error_patterns)
    
    def _categorize_error_pattern(self, error_type: str, error_msg: str) -> str:
        """Categorize error into a pattern"""
        msg_lower = error_msg.lower()
        
        # Python version specific patterns
        if "walrus" in msg_lower or ":=" in msg_lower:
            return "walrus_operator_syntax"
        elif "match" in msg_lower and "case" in msg_lower:
            return "pattern_matching_syntax"
        elif "async" in msg_lower and "await" in msg_lower:
            return "async_await_syntax"
        elif "f-string" in msg_lower or "fstring" in msg_lower:
            return "fstring_syntax"
        
        # Type annotation patterns
        elif "type" in error_type.lower() or "annotation" in msg_lower:
            if "union" in msg_lower:
                return "union_type_annotation"
            elif "generic" in msg_lower:
                return "generic_type_annotation"
            else:
                return "type_annotation"
        
        # Import patterns
        elif "import" in error_type.lower():
            if "circular" in msg_lower:
                return "circular_import"
            elif "relative" in msg_lower:
                return "relative_import"
            else:
                return "import_error"
        
        # Syntax patterns
        elif error_type == "SyntaxError":
            if "encoding" in msg_lower:
                return "encoding_error"
            elif "indentation" in msg_lower:
                return "indentation_error"
            else:
                return "generic_syntax_error"
        
        # Default
        return f"{error_type.lower()}_general"
    
    def calculate_feature_usage(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate language feature usage statistics"""
        # This would require actual AST analysis results
        # For now, return placeholder structure
        return {
            "classes": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "functions": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "async_await": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "decorators": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "type_hints": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "fstrings": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "comprehensions": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "generators": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "lambdas": {"files_using": 0, "percentage": 0, "parse_success": 0},
            "context_managers": {"files_using": 0, "percentage": 0, "parse_success": 0},
        }
    
    def update_compatibility_matrix(self):
        """Update COMPATIBILITY.md with latest results"""
        results, stats, matrix = self.load_results()
        
        compat_file = Path(__file__).parent.parent / "COMPATIBILITY.md"
        if not compat_file.exists():
            print(f"Warning: {compat_file} not found")
            return
        
        content = compat_file.read_text()
        
        # Update repository test results section
        if results:
            repo_table_lines = ["| Repository | Files | Success Rate | Parse Time | Last Tested |",
                               "|------------|-------|--------------|------------|-------------|"]
            
            for result in results:
                success_rate = result.get("metrics", {}).get("success_rate", 0)
                repo_table_lines.append(
                    f"| {result['repo_name']} | "
                    f"{result['total_files']} | "
                    f"{success_rate:.1%} | "
                    f"{result['parse_time_ms']:.0f}ms | "
                    f"{datetime.now().strftime('%Y-%m-%d')} |"
                )
            
            # Replace the table in the content
            pattern = r"(\| Repository \| Files \| Success Rate.*?\n)((?:\|.*\n)*)"
            replacement = "\n".join(repo_table_lines) + "\n"
            content = re.sub(pattern, f"\\1{replacement}", content, flags=re.DOTALL)
        
        # Update Python version support based on results
        if matrix.get("python_versions"):
            for version, data in matrix["python_versions"].items():
                if data["total_files"] > 0:
                    success_rate = data["success_files"] / data["total_files"]
                    coverage = f"{success_rate:.0%}"
                    
                    # Update coverage in table
                    pattern = f"\\| {version} \\| .+? \\| .+? \\|"
                    replacement = f"| {version} | 🟢 Supported | {coverage} |"
                    content = re.sub(pattern, replacement, content)
        
        compat_file.write_text(content)
        print(f"Updated {compat_file}")
    
    def update_stats_report(self):
        """Update stats_report.md with latest statistics"""
        results, stats, matrix = self.load_results()
        
        report_file = self.validation_dir / "stats_report.md"
        if not report_file.exists():
            print(f"Warning: {report_file} not found")
            return
        
        content = report_file.read_text()
        
        # Update overview section
        if stats:
            content = re.sub(
                r"- \*\*Total Repositories Tested\*\*: \d+",
                f"- **Total Repositories Tested**: {stats.get('total_repositories', 0)}",
                content
            )
            content = re.sub(
                r"- \*\*Total Python Files Analyzed\*\*: \d+",
                f"- **Total Python Files Analyzed**: {stats.get('total_files_analyzed', 0):,}",
                content
            )
            content = re.sub(
                r"- \*\*Date of Analysis\*\*: .+",
                f"- **Date of Analysis**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                content
            )
            
            # Update success metrics
            success_rate = stats.get('overall_success_rate', 0)
            content = re.sub(
                r"- \*\*Successfully Parsed\*\*: \d+ files \(\d+%\)",
                f"- **Successfully Parsed**: {stats.get('total_files_parsed', 0):,} files ({success_rate:.1%})",
                content
            )
            content = re.sub(
                r"- \*\*Failed to Parse\*\*: \d+ files \(\d+%\)",
                f"- **Failed to Parse**: {stats.get('total_files_failed', 0):,} files ({(1-success_rate):.1%})",
                content
            )
        
        # Update performance metrics
        if stats.get("performance_metrics"):
            perf = stats["performance_metrics"]
            content = re.sub(
                r"- \*\*Average Parse Time\*\*: \d+ms per file",
                f"- **Average Parse Time**: {perf.get('avg_parse_time_ms', 0):.1f}ms per file",
                content
            )
        
        # Update error distribution
        if stats.get("error_distribution"):
            error_table_lines = ["| Error Type | Count | Percentage | Most Common In |",
                                "|------------|-------|------------|----------------|"]
            
            total_errors = sum(stats["error_distribution"].values())
            
            for error_type, count in sorted(
                stats["error_distribution"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                error_table_lines.append(
                    f"| {error_type} | {count} | {percentage:.1f}% | - |"
                )
            
            # Replace error table
            pattern = r"(\| Error Type \| Count \| Percentage.*?\n)((?:\|.*\n)*)"
            replacement = "\n".join(error_table_lines) + "\n"
            content = re.sub(pattern, f"\\1{replacement}", content, flags=re.DOTALL)
        
        report_file.write_text(content)
        print(f"Updated {report_file}")
    
    def generate_insights_report(self):
        """Generate insights and recommendations"""
        results, stats, matrix = self.load_results()
        
        if not results:
            print("No validation results found. Run validation first.")
            return
        
        insights = {
            "summary": {
                "total_repos": len(results),
                "total_files": sum(r["total_files"] for r in results),
                "overall_success_rate": stats.get("overall_success_rate", 0),
                "most_compatible_repo": None,
                "least_compatible_repo": None,
            },
            "key_findings": [],
            "recommendations": [],
            "error_patterns": self.analyze_error_patterns(results),
            "feature_usage": self.calculate_feature_usage(results)
        }
        
        # Find most/least compatible repositories
        sorted_repos = sorted(
            results,
            key=lambda r: r.get("metrics", {}).get("success_rate", 0),
            reverse=True
        )
        
        if sorted_repos:
            insights["summary"]["most_compatible_repo"] = {
                "name": sorted_repos[0]["repo_name"],
                "success_rate": sorted_repos[0].get("metrics", {}).get("success_rate", 0)
            }
            insights["summary"]["least_compatible_repo"] = {
                "name": sorted_repos[-1]["repo_name"],
                "success_rate": sorted_repos[-1].get("metrics", {}).get("success_rate", 0)
            }
        
        # Generate key findings
        if insights["error_patterns"]:
            most_common_error = max(
                insights["error_patterns"].items(),
                key=lambda x: x[1]["count"]
            )
            insights["key_findings"].append(
                f"Most common error pattern: {most_common_error[0]} "
                f"({most_common_error[1]['count']} occurrences across "
                f"{len(most_common_error[1]['repositories'])} repositories)"
            )
        
        # Generate recommendations
        if stats.get("overall_success_rate", 0) < 0.9:
            insights["recommendations"].append(
                "Focus on improving parser robustness - current success rate is below 90%"
            )
        
        if "walrus_operator_syntax" in insights["error_patterns"]:
            insights["recommendations"].append(
                "Implement proper handling for walrus operator (:=) introduced in Python 3.8"
            )
        
        if "pattern_matching_syntax" in insights["error_patterns"]:
            insights["recommendations"].append(
                "Add support for pattern matching (match/case) introduced in Python 3.10"
            )
        
        # Save insights
        insights_file = self.validation_dir / "validation_insights.json"
        insights_file.write_text(json.dumps(insights, indent=2))
        print(f"Generated insights: {insights_file}")
        
        return insights


def main():
    """Analyze validation results and update documentation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze validation results")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing validation results"
    )
    
    args = parser.parse_args()
    
    analyzer = ValidationAnalyzer(args.validation_dir)
    
    # Check if results exist
    if not analyzer.results_file.exists():
        print("No validation results found.")
        print(f"Run 'python {args.validation_dir}/validate.py' first.")
        return 1
    
    print("Analyzing validation results...")
    
    # Update all reports
    analyzer.update_compatibility_matrix()
    analyzer.update_stats_report()
    insights = analyzer.generate_insights_report()
    
    # Print summary
    if insights:
        print("\n=== Validation Summary ===")
        print(f"Repositories tested: {insights['summary']['total_repos']}")
        print(f"Total files analyzed: {insights['summary']['total_files']:,}")
        print(f"Overall success rate: {insights['summary']['overall_success_rate']:.1%}")
        
        if insights['summary']['most_compatible_repo']:
            print(f"\nMost compatible: {insights['summary']['most_compatible_repo']['name']} "
                  f"({insights['summary']['most_compatible_repo']['success_rate']:.1%})")
        
        if insights['summary']['least_compatible_repo']:
            print(f"Least compatible: {insights['summary']['least_compatible_repo']['name']} "
                  f"({insights['summary']['least_compatible_repo']['success_rate']:.1%})")
        
        if insights['key_findings']:
            print("\n=== Key Findings ===")
            for finding in insights['key_findings']:
                print(f"- {finding}")
        
        if insights['recommendations']:
            print("\n=== Recommendations ===")
            for rec in insights['recommendations']:
                print(f"- {rec}")
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())