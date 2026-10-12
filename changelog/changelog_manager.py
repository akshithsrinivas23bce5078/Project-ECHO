"""
Project ECHO — Change Logger
================================
Centralized change logging system that tracks all project
modifications with timestamps, categories, and semantic versioning.

Writes both human-readable CHANGELOG.md and machine-parseable
changelog.json for programmatic access.

Usage:
    from changelog.changelog_manager import ChangeLogger

    logger = ChangeLogger()
    logger.log("feature", "Added confusion matrix evaluation plot",
               files=["evaluation/evaluation_plots.py"])
    logger.save()
"""

import os
import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# Valid change categories
CHANGE_CATEGORIES = [
    "feature",       # New functionality
    "fix",           # Bug fixes
    "refactor",      # Code restructuring
    "metric",        # Evaluation metrics additions
    "visualization", # New or updated plots
    "config",        # Configuration changes
    "test",          # Test additions or modifications
    "docs",          # Documentation changes
    "pipeline",      # Pipeline integration changes
    "performance",   # Performance improvements
]

# Category display labels with emoji
CATEGORY_LABELS = {
    "feature": "[Feature]",
    "fix": "[Fix]",
    "refactor": "[Refactor]",
    "metric": "[Metric]",
    "visualization": "[Visualization]",
    "config": "[Config]",
    "test": "[Test]",
    "docs": "[Docs]",
    "pipeline": "[Pipeline]",
    "performance": "[Performance]",
}

# Default paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHANGELOG_MD_PATH = os.path.join(PROJECT_ROOT, "CHANGELOG.md")
CHANGELOG_JSON_PATH = os.path.join(PROJECT_ROOT, "changelog", "changelog.json")


@dataclass
class ChangeEntry:
    """A single change log entry."""
    category: str
    description: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    files: List[str] = field(default_factory=list)
    version: str = ""
    author: str = "Project ECHO Team"

    def __post_init__(self):
        if self.category not in CHANGE_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of: {CHANGE_CATEGORIES}"
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_markdown_line(self) -> str:
        """Format as a markdown list item."""
        label = CATEGORY_LABELS.get(self.category, self.category)
        files_str = ""
        if self.files:
            files_str = " — " + ", ".join(f"`{f}`" for f in self.files)
        return f"- **{label}**: {self.description}{files_str}"


@dataclass
class VersionRelease:
    """A versioned release containing multiple change entries."""
    version: str
    date: str
    title: str
    entries: List[ChangeEntry] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "date": self.date,
            "title": self.title,
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_markdown(self) -> str:
        """Format as a markdown section."""
        lines = [
            f"## [{self.version}] — {self.date}",
            f"### {self.title}",
            "",
        ]

        # Group entries by category
        grouped: Dict[str, List[ChangeEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.category, []).append(entry)

        for category in CHANGE_CATEGORIES:
            if category in grouped:
                for entry in grouped[category]:
                    lines.append(entry.to_markdown_line())

        lines.append("")
        return "\n".join(lines)


class ChangeLogger:
    """
    Centralized change logging system for Project ECHO.

    Tracks modifications with timestamps, categories, and semantic
    versioning. Outputs both CHANGELOG.md (human-readable) and
    changelog.json (machine-parseable).
    """

    def __init__(
        self,
        md_path: str = CHANGELOG_MD_PATH,
        json_path: str = CHANGELOG_JSON_PATH,
    ):
        self.md_path = md_path
        self.json_path = json_path
        self._releases: List[VersionRelease] = []
        self._current_entries: List[ChangeEntry] = []

        # Load existing changelog if present
        self._load_existing()

    def _load_existing(self):
        """Load existing changelog.json if it exists."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for release_data in data.get("releases", []):
                    entries = [
                        ChangeEntry(**e) for e in release_data.get("entries", [])
                    ]
                    release = VersionRelease(
                        version=release_data["version"],
                        date=release_data["date"],
                        title=release_data["title"],
                        entries=entries,
                    )
                    self._releases.append(release)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Start fresh if corrupt

    def log(
        self,
        category: str,
        description: str,
        files: Optional[List[str]] = None,
        author: str = "Project ECHO Team",
    ) -> ChangeEntry:
        """
        Log a single change.

        Args:
            category: Change category (feature, fix, metric, etc.)
            description: Human-readable description of the change.
            files: List of affected file paths (relative to project root).
            author: Author of the change.

        Returns:
            The created ChangeEntry.
        """
        entry = ChangeEntry(
            category=category,
            description=description,
            files=files or [],
            author=author,
        )
        self._current_entries.append(entry)
        return entry

    def create_release(
        self,
        version: str,
        title: str,
        date: Optional[str] = None,
    ) -> VersionRelease:
        """
        Finalize current entries into a versioned release.

        Args:
            version: Semantic version string (e.g., "2.0.0").
            title: Release title.
            date: Release date (defaults to today).

        Returns:
            The created VersionRelease.
        """
        if date is None:
            date = datetime.date.today().isoformat()

        release = VersionRelease(
            version=version,
            date=date,
            title=title,
            entries=list(self._current_entries),
        )

        # Assign version to all entries
        for entry in release.entries:
            entry.version = version

        self._releases.insert(0, release)  # Most recent first
        self._current_entries.clear()
        return release

    def save(self):
        """Save changelog to both markdown and JSON files."""
        self._save_markdown()
        self._save_json()

    def _save_markdown(self):
        """Write CHANGELOG.md."""
        lines = [
            "# Changelog",
            "",
            "> All notable changes to Project ECHO are documented in this file.",
            "> Format follows [Keep a Changelog](https://keepachangelog.com/).",
            "",
            "---",
            "",
        ]

        # Add pending entries if any
        if self._current_entries:
            lines.append("## [Unreleased]")
            lines.append("")
            for entry in self._current_entries:
                lines.append(entry.to_markdown_line())
            lines.append("")
            lines.append("---")
            lines.append("")

        # Add releases (most recent first)
        for release in self._releases:
            lines.append(release.to_markdown())
            lines.append("---")
            lines.append("")

        os.makedirs(os.path.dirname(self.md_path), exist_ok=True)
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_json(self):
        """Write changelog.json."""
        data = {
            "project": "Project ECHO",
            "generated_at": datetime.datetime.now().isoformat(),
            "pending_entries": [e.to_dict() for e in self._current_entries],
            "releases": [r.to_dict() for r in self._releases],
        }

        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_latest_version(self) -> Optional[str]:
        """Get the latest release version string."""
        if self._releases:
            return self._releases[0].version
        return None

    def get_all_entries(self) -> List[ChangeEntry]:
        """Get all entries across all releases plus pending."""
        entries = list(self._current_entries)
        for release in self._releases:
            entries.extend(release.entries)
        return entries

    def summary(self) -> str:
        """Generate a text summary of the changelog state."""
        lines = [
            f"Changelog Summary:",
            f"  Releases: {len(self._releases)}",
            f"  Pending entries: {len(self._current_entries)}",
        ]
        if self._releases:
            lines.append(f"  Latest version: {self._releases[0].version}")
        total = sum(len(r.entries) for r in self._releases)
        total += len(self._current_entries)
        lines.append(f"  Total entries: {total}")
        return "\n".join(lines)
