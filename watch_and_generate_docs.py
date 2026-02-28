"""
Auto-generates FULL_PROJECT_SOURCE.md whenever any source file changes.
Run this in the background: python watch_and_generate_docs.py
"""
import os
import time
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUT_FILE = PROJECT_ROOT / "FULL_PROJECT_SOURCE.md"

# File extensions to include
INCLUDE_EXTENSIONS = {".py", ".json", ".txt"}

# Directories/files to exclude
EXCLUDE_PATTERNS = {"__pycache__", ".git", "watch_and_generate_docs.py", "FULL_PROJECT_SOURCE.md", ".pyc"}


def should_include(filepath: Path) -> bool:
    """Check if file should be included in documentation."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in str(filepath):
            return False
    return filepath.suffix in INCLUDE_EXTENSIONS


def get_all_source_files() -> list:
    """Get all source files sorted by path."""
    files = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", ".venv", "venv", "node_modules"}]
        for filename in filenames:
            filepath = Path(root) / filename
            if should_include(filepath):
                files.append(filepath)
    return sorted(files, key=lambda f: str(f.relative_to(PROJECT_ROOT)))


def get_lang_id(filepath: Path) -> str:
    """Get language identifier for code blocks."""
    ext_map = {
        ".py": "py",
        ".json": "json",
        ".txt": "txt",
    }
    return ext_map.get(filepath.suffix, "")


def generate_docs():
    """Generate the FULL_PROJECT_SOURCE.md file."""
    files = get_all_source_files()

    lines = []
    lines.append("# MlProject-2 - Full Project Source Code\n")
    lines.append("## Directory Structure\n")
    lines.append("```")
    for f in files:
        lines.append(str(f.relative_to(PROJECT_ROOT)))
    lines.append("```\n")

    for f in files:
        rel_path = f.relative_to(PROJECT_ROOT)
        lang = get_lang_id(f)

        lines.append("---\n")
        lines.append(f"### {rel_path}\n")

        try:
            content = f.read_text(encoding="utf-8")
        except Exception as e:
            content = f"# Error reading file: {e}"

        lines.append(f"```{lang}")
        lines.append(content)
        lines.append("```\n")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Generated {OUTPUT_FILE.name} with {len(files)} files at {time.strftime('%H:%M:%S')}")


def compute_snapshot(files: list) -> str:
    """Compute a hash snapshot of all file contents and modification times."""
    hasher = hashlib.md5()
    for f in files:
        try:
            stat = f.stat()
            hasher.update(str(f).encode())
            hasher.update(str(stat.st_mtime_ns).encode())
            hasher.update(str(stat.st_size).encode())
        except FileNotFoundError:
            pass
    return hasher.hexdigest()


def watch(interval: float = 2.0):
    """Watch for file changes and regenerate docs."""
    print("=" * 60)
    print("👁️  FULL_PROJECT_SOURCE.md Auto-Generator")
    print("=" * 60)
    print(f"Watching: {PROJECT_ROOT}")
    print(f"Output:   {OUTPUT_FILE}")
    print(f"Interval: {interval}s")
    print(f"Press Ctrl+C to stop\n")

    # Generate initial version
    generate_docs()
    last_snapshot = compute_snapshot(get_all_source_files())

    try:
        while True:
            time.sleep(interval)
            current_files = get_all_source_files()
            current_snapshot = compute_snapshot(current_files)

            if current_snapshot != last_snapshot:
                print(f"🔄 Change detected, regenerating...")
                generate_docs()
                last_snapshot = current_snapshot
    except KeyboardInterrupt:
        print("\n👋 Watcher stopped.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # One-time generation
        generate_docs()
    else:
        # Continuous watching
        watch()