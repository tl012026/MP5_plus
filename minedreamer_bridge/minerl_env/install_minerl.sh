#!/bin/bash
# Build Zhoues/minerl (MineRL 1.0 + chat, MC 1.16.5) in place on /scratch, inside minerl-mp5.sif.
# Replaces setup.py's prep_mcp() (which re-clones MCP-Reborn on every run) with resumable stages
# and two fixes for 2026:
#   1. MCP-Reborn's :mcp project has no repositories, so ca.weblite:java-objc-bridge:1.0.0:natives-osx
#      cannot resolve ("no repositories are defined"); libraries.minecraft.net is added for `setup` only.
#   2. gradle downloadAssets races on duplicate hashes (icons/*); assets are prefetched into
#      $GRADLE_USER_HOME/caches/forge_gradle/assets by fetch_mc_assets.py before this runs.
# Use from Python with PYTHONPATH=$SRC (no pip install, so the package dir stays writable for logs).
#
#   apptainer exec minerl-mp5.sif bash /scratch/jl16678/env/minerl/install_minerl.sh
set -euxo pipefail
ROOT=/scratch/jl16678/env/minerl
SRC=$ROOT/minerl-src
MCP=$SRC/minerl/MCP-Reborn
export GRADLE_USER_HOME=$ROOT/gradle-home
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export JAVA_TOOL_OPTIONS="-Djava.net.preferIPv4Stack=true"
# fix 4: no gradle daemon. A daemon outlives `apptainer exec`; once the image is unmounted it throws
# NoClassDefFoundError (java/util/stream/...) on the next build that reuses it.
export GRADLE_OPTS="-Dorg.gradle.daemon=false"
cd "$MCP"

if [ ! -f .stage_setup ]; then
    [ -f build.gradle.vanilla ] || cp build.gradle build.gradle.vanilla
    cp build.gradle.vanilla build.gradle
    python3 - <<'EOF'
from pathlib import Path
p = Path("build.gradle"); s = p.read_text()
old = "project(':mcp') {\n"
new = old + "    repositories {\n        maven { url = 'https://libraries.minecraft.net/' }\n        mavenCentral()\n    }\n"
assert old in s
p.write_text(s.replace(old, new, 1))
EOF
    chmod +x gradlew
    # fix 3: FG3 fails to run the client 'rename' step late in configuration; fix_client_srg.sh runs it.
    ./gradlew setup || { bash "$ROOT/fix_client_srg.sh" && ./gradlew setup; }
    cp build.gradle.vanilla build.gradle          # the MineRL patch expects the vanilla file
    touch .stage_setup
fi

if [ ! -f .stage_patched ]; then
    rm -rf .git run projects build .gradle .gitignore .github README.md
    bash "$SRC/scripts/patch_mcp.sh"
    touch .stage_patched
fi

if [ ! -f .stage_assets ]; then
    for i in 1 2 3; do ./gradlew downloadAssets && break; [ "$i" = 3 ] && exit 1; done
    python3 - <<'EOF'
import json, os, shutil
assets = os.path.join(os.environ["GRADLE_USER_HOME"], "caches", "forge_gradle", "assets")
out = os.path.join(os.getcwd(), "src", "main", "resources")
index = json.load(open(os.path.join(assets, "indexes", "1.16.json")))
for key, val in index["objects"].items():
    h = val["hash"]
    dst = os.path.join(out, key)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(assets, "objects", h[:2], h), dst)
print("unpacked", len(index["objects"]), "assets")
EOF
    touch .stage_assets
fi

if [ ! -f .stage_built ]; then
    ./gradlew clean build shadowJar
    touch .stage_built
fi
ls -la build/libs/
echo "MineRL build done: PYTHONPATH=$SRC"
