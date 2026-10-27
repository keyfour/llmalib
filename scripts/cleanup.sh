#!/usr/bin/env bash
#
# scripts/cleanup.sh — safe development-artifact cleanup for the llmalib repo.
#
# Removes only gitignored build/test caches and the stray untracked probe
# crate (`crates/giltest-probe`, a GIL-release experiment leftover that owns
# its own [workspace], so it is not a member of the main workspace).
#
# Safety guards:
# - resolves the repo root from the script's own location and verifies the
#   marker files (pyproject.toml / Cargo.toml / crates) before touching
#   anything;
# - every target path is written out literally — nothing is built from
#   variables except the fixed cache dirs found under llmalib/tests/docs;
# - refuses to delete any directory that contains files TRACKED by git
#   (a stray must never be adopted into the tree mid-cleanup);
# - verifies the tracked tree state before/after is unchanged, so whatever
#   was already dirty before the run stays exactly as it was.
#
# Usage:
#   scripts/cleanup.sh [--dry-run] [--target] [--rebuild]
#
#   --dry-run   Print what would be removed; delete nothing.
#   --target    Also remove the workspace Rust `target/` dir (large; the
#               next cargo command does a full rebuild lazily).
#   --rebuild   Rebuild the Python extension afterwards and run the test
#               suite (uv sync --extra dev --reinstall-package llmalib).
#
# Exit codes: 0 ok, 1 misuse, 2 refused-for-safety or post-check failure.

set -u

# --- resolve repo root from the script location ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

if [ ! -f pyproject.toml ] || [ ! -f Cargo.toml ] || [ ! -d crates ]; then
    echo "cleanup: not an llmalib checkout (no pyproject.toml/Cargo.toml/crates) — refusing" >&2
    exit 2
fi

# --- flags -------------------------------------------------------------------
DRY_RUN=0
CLEAN_TARGET=0
REBUILD=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --target)  CLEAN_TARGET=1 ;;
        --rebuild) REBUILD=1 ;;
        *)
            echo "cleanup: unknown argument '$arg'" >&2
            echo "usage: cleanup.sh [--dry-run] [--target] [--rebuild]" >&2
            exit 1
            ;;
    esac
done

# --- helpers -----------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    remove() {
        if [ -e "$1" ]; then
            echo "would remove: $1"
        else
            echo "absent (skip): $1"
        fi
    }
else
    remove() {
        if [ -e "$1" ]; then
            if rm -rf -- "$1"; then
                echo "removed: $1"
            else
                echo "FAILED to remove: $1" >&2
            fi
        else
            echo "absent (skip): $1"
        fi
    }
fi

echo "== cleanup in $REPO_ROOT (dry-run=$DRY_RUN target=$CLEAN_TARGET rebuild=$REBUILD)"

# --- snapshot tracked-tree state (to re-verify afterwards) -------------------
# Tracked paths only: ignores untracked files (including this very script and
# any pre-existing dirt), so a dirty worktree does not produce false alarms.
TRACK_PATHS="llmalib crates tests docs pyproject.toml uv.lock Cargo.toml Cargo.lock"
dirty_before="$(git status --porcelain -- $TRACK_PATHS 2>/dev/null)"
if [ -n "$dirty_before" ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "== note: tracked tree has uncommitted changes — keeping their diff for comparison" >&2
    diff_before="$(mktemp)" || diff_before=""
    [ -n "$diff_before" ] && printf '%s\n' "$dirty_before" >"$diff_before"
    trap '[ -n "${diff_before:-}" ] && rm -f "$diff_before" 2>/dev/null' EXIT
fi

# --- stray probe crate -------------------------------------------------------
probe="crates/giltest-probe"
if [ -e "$probe" ]; then
    if [ -n "$(git ls-files "$probe")" ]; then
        echo "cleanup: $probe contains TRACKED files — refusing to delete" >&2
        exit 2
    fi
    remove "$probe"
    if [ "$DRY_RUN" -eq 0 ] && [ -e "$probe" ]; then
        echo "cleanup: $probe still present after removal — remove manually" >&2
        exit 2
    fi
elif [ "$DRY_RUN" -eq 1 ]; then
    echo "absent (skip): $probe"
fi

# --- python test / tool caches (all gitignored) ------------------------------
remove ".coverage"
remove "htmlcov"
remove ".pytest_cache"
remove ".mypy_cache"
remove ".ruff_cache"
for cache in llmalib tests docs; do
    [ -d "$cache" ] || continue
    # -print: prunes nested traversal; __pycache__ dirs never contain each other.
    find "$cache" -type d -name __pycache__ -prune -print | while read -r dir; do
        remove "$dir"
    done
done

# --- rust build artifacts ----------------------------------------------------
if [ "$CLEAN_TARGET" -eq 1 ]; then
    remove "target"
else
    echo "keeping: target (pass --target to remove)"
fi

# --- verify the tracked tree is untouched ------------------------------------
# Ignore untracked listings (`??`): untracked files are not what cleanup
# removes (except the probe leftover, where removal is intentional).
normalize() { sed '/^?? /d'; }
dirty_before_trim="$(printf '%s\n' "$dirty_before" | normalize)"
dirty_after_trim="$(git status --porcelain -- $TRACK_PATHS | normalize)"
if [ -n "${diff_before:-}" ]; then
    if [ "$dirty_after_trim" = "$dirty_before_trim" ]; then
        echo "tracked-tree diff unchanged (pre-existing work preserved)"
    else
        echo "cleanup: tracked tree changed unexpectedly:" >&2
        diff "$diff_before" <(printf '%s\n' "$dirty_after_trim") >&2 || true
        exit 2
    fi
elif [ -n "$dirty_after_trim" ]; then
    echo "cleanup: cleanup removed/changed TRACKED content:" >&2
    echo "$dirty_after_trim" >&2
    exit 2
fi

# --- optional rebuild --------------------------------------------------------
if [ "$REBUILD" -eq 1 ]; then
    echo "== rebuilding the llmalib extension (uv sync --extra dev)"
    uv sync --extra dev --reinstall-package llmalib || {
        echo "cleanup: rebuild failed" >&2
        exit 2
    }
    uv run pytest tests/ -q -o addopts="" || {
        echo "cleanup: test suite failed after rebuild" >&2
        exit 2
    }
else
    echo "skipping rebuild (pass --rebuild to run uv sync + pytest)"
fi

echo "== done"
exit 0
