#!/bin/sh
#
# Build the embedded web UI, pack it as a local archive under files/,
# and invalidate the extracted dist in build directories so the next CMake
# build re-extracts from the new archive.
#
# Package manager: prefers bun, falls back to npm/node if bun is not in PATH.
#
# Usage:
#   ./build-ui.sh                # version derived from git rev-list --count HEAD
#   ./build-ui.sh b12345         # explicit version tag
#
# Prerequisites: (bun | node+npm), git, tar

set -e

log()  { echo "[+] $*" 1>&2; }
warn() { echo "[!] $*" 1>&2; }
die()  { echo "[E] $*" 1>&2; exit 1; }

# Resolve version: explicit arg > git rev-list count
if [ -n "$1" ]; then
    version="$1"
else
    if out=$(git rev-list --count HEAD 2>/dev/null); then
        version="b$(printf '%s' "$out" | tr -d '\n')"
    else
        die "cannot resolve version from git; pass it explicitly: ./build-ui.sh b<version>"
    fi
fi

case "$version" in
    b*) : ;;
    *)  die "version must start with 'b' (got '$version')"
esac

root=$(cd "$(dirname "$0")" && pwd)
ui_src="$root/tools/ui"
archive="$root/files/llama-${version}-ui.tar.gz"

command -v tar >/dev/null 2>&1 || die "tar not found in PATH"
[ -f "$ui_src/package.json" ] || die "UI source not found: $ui_src/package.json"

# Select package manager: prefer bun, fall back to npm/node
if command -v bun >/dev/null 2>&1; then
    pkg_mgr="bun"
elif command -v npm >/dev/null 2>&1 && command -v node >/dev/null 2>&1; then
    pkg_mgr="npm"
else
    die "neither bun nor npm+node found in PATH"
fi

mkdir -p "$root/files"

log "version: $version"
log "pkg mgr: $pkg_mgr"
log "ui src : $ui_src"
log "archive: $archive"

# Install deps if missing or lockfile newer than marker.
# Marker is shared between package managers; lockfile staleness is checked
# against whichever lockfile is present for the chosen manager.
marker="$ui_src/node_modules/.ui-deps-stamp"
need_install=0
if [ ! -f "$marker" ]; then
    need_install=1
else
    case "$pkg_mgr" in
        bun)
            if [ -f "$ui_src/bun.lock" ] && [ "$ui_src/bun.lock" -nt "$marker" ]; then
                need_install=1
            elif [ -f "$ui_src/bun.lockb" ] && [ "$ui_src/bun.lockb" -nt "$marker" ]; then
                need_install=1
            fi
            ;;
        npm)
            if [ -f "$ui_src/package-lock.json" ] && [ "$ui_src/package-lock.json" -nt "$marker" ]; then
                need_install=1
            fi
            ;;
    esac
fi

if [ "$need_install" = "1" ]; then
    log "running $pkg_mgr install"
    (cd "$ui_src" && "$pkg_mgr" install)
    mkdir -p "$ui_src/node_modules"
    : > "$marker"
else
    log "deps up-to-date, skipping $pkg_mgr install"
fi

# Build (LLAMA_UI_VERSION / LLAMA_BUILD_NUMBER drive build.json)
# Clean dist and .svelte-kit/output first so stale chunks from previous
# builds don't accumulate and get precached by the service worker.
log "cleaning dist and .svelte-kit/output"
rm -rf "$ui_src/dist" "$ui_src/.svelte-kit/output"

log "running $pkg_mgr run build"
(
    cd "$ui_src"
    LLAMA_UI_VERSION="$version" \
    LLAMA_BUILD_NUMBER="${version#b}" \
    "$pkg_mgr" run build
)

[ -f "$ui_src/dist/index.html" ] || die "build failed: dist/index.html missing"
[ -f "$ui_src/dist/loading.html" ] || die "build failed: dist/loading.html missing"

# Pack archive (contents of dist/ at archive root; ui-assets.cmake flattens if wrapped)
log "packing archive"
rm -f "$archive"
tar -czf "$archive" -C "$ui_src/dist" .
log "archive size: $(wc -c < "$archive" | tr -d ' ') bytes"

# Invalidate extracted dist in all build directories so next cmake run re-extracts
for build_dir in "$root"/build-*/tools/ui; do
    [ -d "$build_dir" ] || continue
    if [ -d "$build_dir/dist" ] || [ -f "$build_dir/.ui-stamp" ]; then
        log "invalidating $build_dir/dist and stamp"
        rm -rf "$build_dir/dist"
        rm -f "$build_dir/.ui-stamp"
    fi
done

log "done. next 'cmake --build' will extract from $archive"
