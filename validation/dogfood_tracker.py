#!/usr/bin/env python3
"""
Track dogfooding metrics during validation

This script helps track how AST tools would help in analyzing the repositories
during validation, providing metrics for the dogfooding strategy.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class DogfoodingMetric:
    """Track a specific AST tool usage scenario"""
    timestamp: str
    repository: str
    task: str
    tool_needed: str
    time_saved_estimate: float  # in minutes
    manual_files_read: int
    would_have_used: str  # How the AST tool would have been used
    pain_point: Optional[str] = None
    
@dataclass
class DogfoodingSession:
    """Track an entire dogfooding session"""
    start_time: str
    end_time: Optional[str] = None
    repository: str
    total_time_minutes: float = 0.0
    metrics: List[DogfoodingMetric] = None
    insights: List[str] = None
    tool_effectiveness_rating: Dict[str, int] = None  # Tool -> rating (1-10)
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []
        if self.insights is None:
            self.insights = []
        if self.tool_effectiveness_rating is None:
            self.tool_effectiveness_rating = {}


class DogfoodingTracker:
    """Track dogfooding metrics during validation"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_file = output_dir / "dogfooding_metrics.json"
        self.current_session: Optional[DogfoodingSession] = None
        self.all_sessions: List[DogfoodingSession] = []
        self._load_existing_metrics()
    
    def _load_existing_metrics(self):
        """Load existing metrics if available"""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                data = json.load(f)
                self.all_sessions = [
                    DogfoodingSession(**session) for session in data.get("sessions", [])
                ]
    
    def start_session(self, repository: str):
        """Start a new dogfooding session"""
        self.current_session = DogfoodingSession(
            start_time=datetime.now().isoformat(),
            repository=repository
        )
        print(f"Started dogfooding session for {repository}")
    
    def track_tool_need(
        self,
        task: str,
        tool_needed: str,
        time_saved_estimate: float,
        manual_files_read: int,
        would_have_used: str,
        pain_point: Optional[str] = None
    ):
        """Track when an AST tool would have been helpful"""
        if not self.current_session:
            print("Warning: No active session. Starting one with 'unknown' repository")
            self.start_session("unknown")
        
        metric = DogfoodingMetric(
            timestamp=datetime.now().isoformat(),
            repository=self.current_session.repository,
            task=task,
            tool_needed=tool_needed,
            time_saved_estimate=time_saved_estimate,
            manual_files_read=manual_files_read,
            would_have_used=would_have_used,
            pain_point=pain_point
        )
        
        self.current_session.metrics.append(metric)
        print(f"Tracked tool need: {tool_needed} for {task}")
    
    def add_insight(self, insight: str):
        """Add an insight from the dogfooding session"""
        if self.current_session:
            self.current_session.insights.append(insight)
    
    def rate_tool(self, tool_name: str, effectiveness: int):
        """Rate tool effectiveness (1-10)"""
        if self.current_session:
            self.current_session.tool_effectiveness_rating[tool_name] = effectiveness
    
    def end_session(self):
        """End the current dogfooding session"""
        if not self.current_session:
            print("Warning: No active session to end")
            return
        
        self.current_session.end_time = datetime.now().isoformat()
        
        # Calculate total time
        start = datetime.fromisoformat(self.current_session.start_time)
        end = datetime.fromisoformat(self.current_session.end_time)
        self.current_session.total_time_minutes = (end - start).total_seconds() / 60
        
        self.all_sessions.append(self.current_session)
        self._save_metrics()
        
        print(f"Ended dogfooding session for {self.current_session.repository}")
        print(f"Total time: {self.current_session.total_time_minutes:.1f} minutes")
        print(f"Tool needs tracked: {len(self.current_session.metrics)}")
        
        self.current_session = None
    
    def _save_metrics(self):
        """Save all metrics to file"""
        data = {
            "sessions": [asdict(session) for session in self.all_sessions],
            "summary": self._generate_summary()
        }
        
        self.metrics_file.write_text(json.dumps(data, indent=2))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not self.all_sessions:
            return {}
        
        total_metrics = []
        for session in self.all_sessions:
            total_metrics.extend(session.metrics)
        
        # Tool usage frequency
        tool_frequency = {}
        total_time_saved = 0.0
        total_files_read = 0
        
        for metric in total_metrics:
            tool = metric.tool_needed
            tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
            total_time_saved += metric.time_saved_estimate
            total_files_read += metric.manual_files_read
        
        # Average effectiveness ratings
        tool_ratings = {}
        rating_counts = {}
        
        for session in self.all_sessions:
            for tool, rating in session.tool_effectiveness_rating.items():
                if tool not in tool_ratings:
                    tool_ratings[tool] = 0
                    rating_counts[tool] = 0
                tool_ratings[tool] += rating
                rating_counts[tool] += 1
        
        avg_ratings = {
            tool: tool_ratings[tool] / rating_counts[tool]
            for tool in tool_ratings
        }
        
        return {
            "total_sessions": len(self.all_sessions),
            "total_metrics_tracked": len(total_metrics),
            "total_time_saved_estimate": total_time_saved,
            "total_manual_files_read": total_files_read,
            "tool_usage_frequency": tool_frequency,
            "average_tool_ratings": avg_ratings,
            "most_needed_tool": max(tool_frequency.items(), key=lambda x: x[1])[0] if tool_frequency else None,
            "average_metrics_per_session": len(total_metrics) / len(self.all_sessions) if self.all_sessions else 0
        }
    
    def generate_report(self):
        """Generate a comprehensive dogfooding report"""
        summary = self._generate_summary()
        
        report = f"""# AST Library Dogfooding Report

## Summary

- **Total Sessions**: {summary.get('total_sessions', 0)}
- **Total Tool Needs Tracked**: {summary.get('total_metrics_tracked', 0)}
- **Estimated Time That Would Be Saved**: {summary.get('total_time_saved_estimate', 0):.1f} minutes
- **Manual Files Read**: {summary.get('total_manual_files_read', 0)}
- **Most Needed Tool**: {summary.get('most_needed_tool', 'None')}

## Tool Usage Frequency

| Tool | Times Needed | Percentage |
|------|--------------|------------|
"""
        
        total_needs = summary.get('total_metrics_tracked', 1)
        for tool, count in sorted(
            summary.get('tool_usage_frequency', {}).items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = (count / total_needs) * 100
            report += f"| {tool} | {count} | {percentage:.1f}% |\n"
        
        report += f"""
## Tool Effectiveness Ratings

| Tool | Average Rating (1-10) |
|------|----------------------|
"""
        
        for tool, rating in sorted(
            summary.get('average_tool_ratings', {}).items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"| {tool} | {rating:.1f} |\n"
        
        report += "\n## Session Details\n\n"
        
        for i, session in enumerate(self.all_sessions, 1):
            report += f"""### Session {i}: {session.repository}

- **Duration**: {session.total_time_minutes:.1f} minutes
- **Tool Needs**: {len(session.metrics)}
- **Time Saved Estimate**: {sum(m.time_saved_estimate for m in session.metrics):.1f} minutes

#### Tools Needed:
"""
            
            for metric in session.metrics:
                report += f"""
**Task**: {metric.task}
- **Tool**: {metric.tool_needed}
- **Time Saved**: {metric.time_saved_estimate:.1f} minutes
- **Files Read Manually**: {metric.manual_files_read}
- **Would Have Used**: {metric.would_have_used}
"""
                if metric.pain_point:
                    report += f"- **Pain Point**: {metric.pain_point}\n"
            
            if session.insights:
                report += "\n#### Insights:\n"
                for insight in session.insights:
                    report += f"- {insight}\n"
            
            report += "\n---\n\n"
        
        report += """## Key Findings

Based on the dogfooding sessions, the most valuable AST tools are:

1. **Function/Class Finder** - Frequently needed to locate definitions
2. **Import Analyzer** - Critical for understanding dependencies
3. **Reference Finder** - Essential for impact analysis
4. **Symbol Extractor** - Speeds up code understanding

## Recommendations

1. Prioritize implementation of the most frequently needed tools
2. Focus on tools that save the most time per use
3. Ensure tools integrate well with validation workflow
4. Add features based on actual pain points encountered

## Pain Points to Address

"""
        
        # Collect unique pain points
        pain_points = set()
        for session in self.all_sessions:
            for metric in session.metrics:
                if metric.pain_point:
                    pain_points.add(metric.pain_point)
        
        for pain_point in pain_points:
            report += f"- {pain_point}\n"
        
        return report
    
    def save_report(self):
        """Save the dogfooding report"""
        report = self.generate_report()
        report_file = self.output_dir / "dogfooding_report.md"
        report_file.write_text(report)
        print(f"Saved dogfooding report to {report_file}")


# Example usage functions for tracking during validation
def track_finding_error_patterns(tracker: DogfoodingTracker, repo: str):
    """Track tool usage for finding error patterns"""
    tracker.track_tool_need(
        task="Finding all try/except blocks to analyze error handling",
        tool_needed="ast.find_pattern",
        time_saved_estimate=15.0,
        manual_files_read=25,
        would_have_used='ast.find_pattern("try: ... except: ...")',
        pain_point="Had to grep through files and manually check each one"
    )

def track_understanding_imports(tracker: DogfoodingTracker, repo: str):
    """Track tool usage for understanding imports"""
    tracker.track_tool_need(
        task="Understanding import structure of the repository",
        tool_needed="ast.analyze_imports",
        time_saved_estimate=20.0,
        manual_files_read=50,
        would_have_used='imports = ast.analyze_imports(); imports.visualize()',
        pain_point="Difficult to trace transitive dependencies manually"
    )

def track_finding_test_files(tracker: DogfoodingTracker, repo: str):
    """Track tool usage for finding test files"""
    tracker.track_tool_need(
        task="Locating all test files and their patterns",
        tool_needed="ast.find_pattern",
        time_saved_estimate=10.0,
        manual_files_read=15,
        would_have_used='ast.find_files("test_*.py") + ast.find_pattern("class Test*")',
        pain_point="Test files scattered across multiple directories"
    )


def main():
    """Example dogfooding tracking session"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track dogfooding metrics")
    parser.add_argument("--repo", required=True, help="Repository being analyzed")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    
    args = parser.parse_args()
    
    tracker = DogfoodingTracker(args.output_dir)
    tracker.start_session(args.repo)
    
    # Simulate tracking tool needs during validation
    print("\nTracking dogfooding metrics...")
    print("(This would normally happen during actual validation)")
    
    # Example tracking
    track_finding_error_patterns(tracker, args.repo)
    track_understanding_imports(tracker, args.repo)
    track_finding_test_files(tracker, args.repo)
    
    # Add insights
    tracker.add_insight("Symbol search would save significant time in large codebases")
    tracker.add_insight("Import visualization is critical for understanding architecture")
    
    # Rate tools
    tracker.rate_tool("ast.find_pattern", 9)
    tracker.rate_tool("ast.analyze_imports", 10)
    tracker.rate_tool("ast.find_files", 7)
    
    tracker.end_session()
    tracker.save_report()
    
    print("\nDogfooding tracking complete!")


if __name__ == "__main__":
    main()