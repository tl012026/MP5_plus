#!/bin/bash
# ForgeGradle 3 cannot create the client 'rename' JavaExec late in configuration
# ("ProjectScopeServices has been closed" -> "client SRG jar not found"). Run the same
# SpecialSource call the joined pipeline used and write the matching step hash so FG3 skips it.
set -euo pipefail
B=/scratch/jl16678/env/minerl/gradle-home/caches/forge_gradle/mcp_repo/de/oceanlabs/mcp/mcp_config/1.16.5-20210115.111550
SS=/scratch/jl16678/env/minerl/gradle-home/caches/forge_gradle/maven_downloader/net/md-5/SpecialSource/1.8.3/SpecialSource-1.8.3-shaded.jar
R=$B/client/rename
[ -f "$B/client/strip/output.jar" ] || { echo "client/strip/output.jar missing; run gradlew setup first"; exit 1; }
mkdir -p "$R/config"
[ -f "$R/config/joined.tsrg" ] || cp "$B/joined/rename/config/joined.tsrg" "$R/config/"
cd "$R"
java -cp "$SS" net.md_5.specialsource.SpecialSource --in-jar "$B/client/strip/output.jar" --out-jar "$R/output.jar" --srg-in config/joined.tsrg
sha() { sha1sum "$1" | cut -d' ' -f1; }
printf 'args=%s\njvmargs=%s\ninput=%s\nmappings=%s\njar=%s' \
  "$(printf '%s' '--in-jar {input} --out-jar {output} --srg-in {mappings}' | sha1sum | cut -d' ' -f1)" \
  "$(printf '' | sha1sum | cut -d' ' -f1)" "$(sha "$B/client/strip/output.jar")" "$(sha config/joined.tsrg)" "$(sha "$SS")" > lastinput.sha1
cat lastinput.sha1; echo; ls -la output.jar
