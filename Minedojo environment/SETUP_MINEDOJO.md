# MineDojo environment setup (Singularity)

Site-agnostic notes for running MineDojo from the official Docker image via Singularity/Apptainer. 

On a typical CPU node this stack ran MineDojo at ~18.6 steps/s with the MP5 config; the first `reset` took about one minute.

## Where to run

- Run setup on a **compute node** that has `singularity` (or `apptainer`) and your job scheduler (`sbatch` or equivalent).
- Some browser IDEs run inside a container that does **not** have Singularity; use those only for editing.
- Prefer batch jobs with `--requeue` (or the local equivalent) for anything longer than a few minutes, so idle interactive sessions are not cancelled.

Scripts in this folder:

- `fetch_mc_assets.py` — fill missing Minecraft assets
- `validate_minedojo.py` — create a task and measure step rate
- `validate_minedojo.sbatch` — example Slurm wrapper (empty `--account` / `--partition`; set them when you submit)

## 1. Environment file

Paths must not use `$USER` inside the image: the MineDojo container sets `$USER` to `user`.

```bash
# SCR = your writable scratch (set this yourself)
mkdir -p "$SCR"/{env,logs,.gradle}
cat > "$SCR/mp5_env.sh" <<EOF
export SCR=$SCR
export MP5=\$SCR/MP5_plus
export GRADLE_USER_HOME=\$SCR/.gradle
export PIP_CACHE_DIR=\$SCR/.pip_cache
export SINGULARITY_CACHEDIR=\$SCR/.sing_cache
export PYTHONUSERBASE=\$SCR/env/mp5_pyuser
export PATH=\$PYTHONUSERBASE/bin:\$PATH
export JAVA_TOOL_OPTIONS="-Djava.net.preferIPv4Stack=true"
EOF
source "$SCR/mp5_env.sh"
```

- `GRADLE_USER_HOME` on scratch: the Minecraft build cache persists across jobs.
- `PYTHONUSERBASE`: extra Python packages go to scratch (the image's site-packages is read-only).
- `preferIPv4Stack`: avoids hanging downloads over IPv6 on some networks.

## 2. MineDojo image

```bash
cd $SCR/env
singularity pull minedojo.sif docker://minedojo/minedojo:latest
```

The image contains Python 3.9, JDK 8, Xvfb and MineDojo. No overlay is required for this baseline.

## 3. MP5 Python dependencies

Pin legacy versions (the MP5 code uses `openai.api_base` and pre-1.0 `langchain` imports). Constrain MineDojo's core packages so they are not shadowed.

```bash
singularity exec $SCR/env/minedojo.sif bash
source "$SCR/mp5_env.sh"

PYTHONNOUSERSITE=1 python -m pip list --format=freeze \
  | grep -iE "^(numpy|gym|minedojo|scipy|pillow|opencv-python|torch|lxml|xmltodict|psutil|pyyaml|protobuf)==" \
  > $SCR/env/minedojo_core_constraints.txt

python -m pip install --user -c $SCR/env/minedojo_core_constraints.txt \
  openai==0.28.1 langchain==0.0.300 chromadb==0.4.14 tiktoken==0.5.1

python -m pip check
python -c "import minedojo, openai, langchain, chromadb; from langchain.chat_models import ChatOpenAI; print('ok')"
exit
```

Expected: `numpy 1.23.1`, `openai 0.28.1`, `langchain 0.0.300`, `chromadb 0.4.14`.

## 4. Fix the MixinGradle dependency

MineDojo's Minecraft build needs `com.github.SpongePowered:MixinGradle:dcfaf61`, which is no longer on jitpack ([MineDojo#113](https://github.com/MineDojo/MineDojo/issues/113)). Serve a community copy from a local Maven repo, added through a Gradle init script.

```bash
git clone --depth 1 https://github.com/verityw/MixinGradle-dcfaf61 $SCR/env/MixinGradle-dcfaf61
M2=$SCR/env/m2/com/github/SpongePowered/MixinGradle/dcfaf61
mkdir -p $M2
cp $SCR/env/MixinGradle-dcfaf61/MixinGradle/dcfaf61/MixinGradle-dcfaf61.jar $M2/
cat > $M2/MixinGradle-dcfaf61.pom <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.github.SpongePowered</groupId>
  <artifactId>MixinGradle</artifactId>
  <version>dcfaf61</version>
</project>
EOF

mkdir -p $GRADLE_USER_HOME/init.d
cat > $GRADLE_USER_HOME/init.d/mixingradle-local.gradle <<EOF
allprojects {
    buildscript {
        repositories {
            maven { url "file://$SCR/env/m2" }
        }
    }
}
EOF
```

## 5. First build and Minecraft assets

The first build downloads ~1,200 Minecraft asset files and can hang on a single slow download (`Current status: 269/1196` in the Gradle log). Fix: run the build once, then fill in missing assets with the script and build again.

```bash
HERE="$MP5/Minedojo environment"
sbatch "$HERE/validate_minedojo.sbatch"   # first attempt; may hang at the asset step
# if it hangs: cancel it, then
python3 "$HERE/fetch_mc_assets.py"        # skips existing files, retries, prints failures
python3 "$HERE/fetch_mc_assets.py"        # rerun until "failed = 0"
sbatch "$HERE/validate_minedojo.sbatch"
```

`fetch_mc_assets.py` needs the asset index, which the first build downloads to `$GRADLE_USER_HOME/caches/minecraft/assets/indexes/`.

## 6. Validate

`validate_minedojo.py` creates a task, steps it, and measures speed with the MP5 config. Expected output:

```
[INFO] reset: ~60s
[INFO] rgb shape: (3, 288, 512)
[INFO] Installation Success
[INFO] MP5 config: ~18 steps/s
[INFO] voxels: (11, 11, 11)
```

Progress while it runs:

```bash
tail -f "$SCR/logs/validate_minedojo_<jobid>.out"                       # script output
tail -f $(ls -t $GRADLE_USER_HOME/daemon/*/daemon-*.out.log | head -1)   # Gradle build
```

## Running code in this environment

```bash
singularity exec $SCR/env/minedojo.sif bash -c '
  source "$SCR/mp5_env.sh"
  cd $MP5 && xvfb-run -a python <script>.py'
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| pip installs to `~/.local` | image site-packages is read-only | set `PYTHONUSERBASE` (step 1) |
| `Could not install ... /tmp/build/.../work` | `pip freeze` writes conda `file://` paths | use `pip list --format=freeze` |
| `ResolutionImpossible` on `tqdm` etc. | constraints too strict | constrain only core packages (step 3) |
| scratch path `.../user/...` missing inside container | `$USER` is `user` in the image | hard-code the scratch path in `mp5_env.sh` |
| `No module named minedojo.scripts` | pip build lacks scripts | use `validate_minedojo.py` in this folder |
| `Could not find ...MixinGradle:dcfaf61` | jitpack artifact removed | step 4 |
| build stuck at `Current status: N/1196` | asset download hangs | step 5 |
| job cancelled while idle | scheduler policy | submit a batch job with requeue; avoid long idle sessions |

## Known overhead

Each Minecraft launch runs a Gradle build (~1 min with a warm cache), because MineDojo copies the Malmo directory to `/tmp` and no prebuilt `MalmoMod-0.37.0-fat.jar` exists. A prebuilt jar would remove this (not done yet).
