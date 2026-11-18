# MineRL 1.0 environment (MC 1.16.5) for the MineDreamer track

The prebuilt `base-vgl-env.sif` from Zhoues/minerl-apptainer is no longer downloadable: the Google Drive id returns 404 and the alternate IP refuses connections. These files rebuild it on the cluster without root. Paths are hard-coded to `/scratch/jl16678/env/minerl`; edit `ROOT` and the paths to reuse them.

1. `build.sh` builds `minerl-mp5.sif` from `minerl-mp5.def` (`apptainer build --fakeroot`, about 20 min). `SINGULARITY_BIND` must be unset during the build.
2. Prefetch the 1.16 assets into `gradle-home/caches/forge_gradle/assets` with `Minedojo environment/fetch_mc_assets.py <indexes/1.16.json>`.
3. `apptainer exec minerl-mp5.sif bash install_minerl.sh` builds Zhoues/minerl@8dd7d8a in place (about 15 min, resumable stages). It includes four fixes: repositories for `:mcp`, a retry for the client-jar download timeout, `fix_client_srg.sh`, and no gradle daemon.
4. Use: `apptainer exec minerl-mp5.sif bash -c 'export PYTHONPATH=/scratch/jl16678/env/minerl/minerl-src; xvfb-run -a python ...'`

CPU smoke test on 2026-11-16 (n2c48m24, no GPU): `HumanSurvival` reset took 33 s and stepping ran at 10–14 steps/s with a 360×640 pov. `location_stats` gives xpos/ypos/zpos, biome_id, is_raining, light_level and sun_brightness. `/clear` + `/give` crafting works, but the commands land a few ticks later.
