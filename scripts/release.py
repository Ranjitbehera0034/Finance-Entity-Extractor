#!/usr/bin/env python3
"""
FinEE Release Script
====================

Automates the release process with proper checks:
1. Ensure on develop branch
2. Run all tests
3. Run benchmark
4. Merge to main
5. Create release tag
6. Build and upload to PyPI
7. Sync to Hugging Face

Usage:
    python scripts/release.py 1.0.4
    python scripts/release.py 1.0.4 --dry-run  # Preview only

Author: Ranjit Behera
"""

import argparse
import subprocess
import sys
import re
from pathlib import Path


def run(cmd: str, check: bool = True, capture: bool = False):
    """Run a shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ Command failed with exit code {result.returncode}")
        if capture:
            print(result.stderr)
        sys.exit(1)
    return result


def get_current_branch() -> str:
    """Get current git branch."""
    result = run("git branch --show-current", capture=True)
    return result.stdout.strip()


def check_clean_working_dir() -> bool:
    """Check if working directory is clean."""
    result = run("git status --porcelain", capture=True)
    return result.stdout.strip() == ""


def run_tests() -> bool:
    """Run all tests."""
    print("\n🧪 Running tests...")
    result = subprocess.run("./venv/bin/pytest tests/ -v", shell=True)
    return result.returncode == 0


def run_benchmark() -> bool:
    """Run benchmark suite."""
    print("\n📊 Running benchmark...")
    result = subprocess.run("./venv/bin/python benchmark.py --all", shell=True)
    return result.returncode == 0


def update_version(new_version: str):
    """Update version in pyproject.toml."""
    pyproject = Path("pyproject.toml")
    content = pyproject.read_text()
    
    # Find and replace version
    pattern = r'version = "[^"]+"'
    new_content = re.sub(pattern, f'version = "{new_version}"', content)
    
    pyproject.write_text(new_content)
    print(f"  ✅ Updated pyproject.toml to version {new_version}")


def update_changelog(new_version: str):
    """Move [Unreleased] to new version section."""
    changelog = Path("CHANGELOG.md")
    content = changelog.read_text()
    
    from datetime import date
    today = date.today().isoformat()
    
    # Replace [Unreleased] with new version
    content = content.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n### Added\n- (Next features go here)\n\n### Changed\n\n### Fixed\n\n---\n\n## [{new_version}] - {today}"
    )
    
    changelog.write_text(content)
    print(f"  ✅ Updated CHANGELOG.md for version {new_version}")


def main():
    parser = argparse.ArgumentParser(description="FinEE Release Script")
    parser.add_argument("version", help="New version (e.g., 1.0.4)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test suite")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark")
    args = parser.parse_args()
    
    version = args.version
    if not re.match(r'^\d+\.\d+\.\d+$', version):
        print(f"❌ Invalid version format: {version}")
        print("   Expected: X.Y.Z (e.g., 1.0.4)")
        sys.exit(1)
    
    print("=" * 60)
    print(f"🚀 FinEE Release v{version}")
    print("=" * 60)
    
    if args.dry_run:
        print("⚠️  DRY RUN - No changes will be made\n")
    
    # Step 1: Check branch
    print("\n📍 Step 1: Checking branch...")
    branch = get_current_branch()
    if branch != "develop":
        print(f"  ❌ Must be on 'develop' branch, currently on '{branch}'")
        print("  Run: git checkout develop")
        sys.exit(1)
    print(f"  ✅ On develop branch")
    
    # Step 2: Check clean working directory
    print("\n📍 Step 2: Checking working directory...")
    if not check_clean_working_dir():
        print("  ❌ Working directory not clean")
        print("  Run: git status")
        sys.exit(1)
    print("  ✅ Working directory clean")
    
    # Step 3: Run tests
    if not args.skip_tests:
        print("\n📍 Step 3: Running tests...")
        if not args.dry_run:
            if not run_tests():
                print("  ❌ Tests failed! Fix before releasing.")
                sys.exit(1)
            print("  ✅ All tests passed")
        else:
            print("  ⏭️  Skipped (dry run)")
    
    # Step 4: Run benchmark
    if not args.skip_benchmark:
        print("\n📍 Step 4: Running benchmark...")
        if not args.dry_run:
            if not run_benchmark():
                print("  ⚠️  Benchmark had failures (continuing...)")
            print("  ✅ Benchmark complete")
        else:
            print("  ⏭️  Skipped (dry run)")
    
    # Step 5: Update version
    print("\n📍 Step 5: Updating version...")
    if not args.dry_run:
        update_version(version)
        update_changelog(version)
        run(f'git add pyproject.toml CHANGELOG.md')
        run(f'git commit -m "chore: bump version to {version}"')
        run('git push origin develop')
    else:
        print(f"  ⏭️  Would update to version {version}")
    
    # Step 6: Merge to main
    print("\n📍 Step 6: Merging to main...")
    if not args.dry_run:
        run('git checkout main')
        run('git pull origin main')
        run('git merge develop --no-edit')
        run('git push origin main')
    else:
        print("  ⏭️  Would merge develop → main")
    
    # Step 7: Create tag
    print("\n📍 Step 7: Creating release tag...")
    if not args.dry_run:
        run(f'git tag -a v{version} -m "Release v{version}"')
        run(f'git push origin v{version}')
    else:
        print(f"  ⏭️  Would create tag v{version}")
    
    # Step 8: Build and upload to PyPI
    print("\n📍 Step 8: Building and uploading to PyPI...")
    if not args.dry_run:
        run('rm -rf dist/')
        run('./venv/bin/python -m build')
        run('./venv/bin/python -m twine upload dist/*')
    else:
        print("  ⏭️  Would build and upload to PyPI")
    
    # Step 9: Return to develop
    print("\n📍 Step 9: Returning to develop branch...")
    if not args.dry_run:
        run('git checkout develop')
        run('git merge main')
        run('git push origin develop')
    else:
        print("  ⏭️  Would checkout develop")
    
    # Done
    print("\n" + "=" * 60)
    if args.dry_run:
        print("✅ DRY RUN COMPLETE - No changes made")
        print(f"   Run without --dry-run to release v{version}")
    else:
        print(f"🎉 RELEASE v{version} COMPLETE!")
        print("\nLinks:")
        print(f"  PyPI: https://pypi.org/project/finee/{version}/")
        print("  GitHub: https://github.com/Ranjitbehera0034/Finance-Entity-Extractor/releases")
    print("=" * 60)


if __name__ == "__main__":
    main()
