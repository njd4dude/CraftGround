# Starting Several Minecraft Instances at Once

Why running 6 Minecraft instances side by side sometimes hung at startup, what we
tried, what worked, and what the code does now. Investigated 2026-11-14 and
2026-11-15.

## Summary

- **Problem:** with 6 instances started at the same moment, one instance sometimes
  failed to start. Training, evaluation and the benchmark step all instances
  together, so the other five then waited for it forever and the run hung.
- **Cause:** every instance ran from **one shared CraftGround folder**. While
  starting, instances write files and create a world save in that folder, and two
  instances doing that in the same split second get in each other's way. It is not
  memory, and it is not the world type (arena and natural fail alike). It only
  happens at startup.
- **Fix now in the code:** each instance runs from **its own copy of the folder**
  (`instances.py`), old Loom lock files are cleaned up before every launch, and new
  copies are built one at a time the first time they are used. All instances still
  start together, so startup stays at about 1.5 to 2 minutes for 6 arena instances.
- **Result:** with the shared folder, 5 of 22 rounds of 6 simultaneous launches
  had a failed instance. With separate folders, 24 of 25 rounds started cleanly,
  and the one failure was caused by a leftover lock file, which the code now
  cleans up.

## Background: how an instance starts

Some terms used below:

| Term | What it is |
| --- | --- |
| **CraftGround's game folder** | `site-packages\craftground_runtime_mc121\` inside the conda env, installed with CraftGround. It holds the mod's code and build files, plus `run\`, where the game keeps world saves, logs and settings. |
| **Gradle** | The build tool CraftGround uses. Every instance starts with `gradlew runClient`, which builds the mod (if needed) and then starts Minecraft. |
| **Loom** | The Gradle plugin for Minecraft mods. It converts Minecraft's game code into a form the mod can use and keeps the result in a cache. |
| **Gradle's cache** | `C:\Users\njdon\.gradle\caches\`. One per Windows user, shared by every Gradle build on the machine. Loom's cache and lock files live in `fabric-loom\` inside it. |
| **Port** | The number CraftGround uses to talk to one Minecraft instance. Every running instance needs its own (train.py uses 8023, 8033, ... 8073). |

By default all instances started from the one installed game folder.

CraftGround's docs and the author's own benchmark repo only ever run **one
instance at a time**. Nothing documents running several, so this setup was ours to
work out.

## The symptom

`benchmark.py` with 6 natural-world instances hung twice (18:38 and 23:04 on
2026-11-14). One instance failed with `RuntimeError: Minecraft process failed to
start`, and the other five sat in reset with no output until stopped by hand. The
23:04 one's Gradle error was `Failed to setup Minecraft ... Failed to remap
minecraft`. Arena with 6 instances had worked once, which at first made it look
like a natural-world problem.

## How we investigated

### Ideas that turned out wrong

| Idea | How we tested it | Result |
| --- | --- | --- |
| **Out of memory** (natural instances use ~2.8 GB each) | Watched free RAM during the failed launch | Ruled out: 15+ GB stayed free |
| **Something about the natural world** | `launch_test.py`: 4 rounds each of arena and natural, 6 instances at once | Ruled out: the arena failed 2 of 4 rounds and natural 0 of 4. The launch is identical for both worlds; the world only matters once the game is running. |
| **Loom rebuilds its cache whenever it finds a leftover lock, and that alone causes the crash** | Planted a fake leftover lock ("disowned") before 4 rounds in the shared folder | Wrong on its own: Loom rebuilt the cache each time, but the other 5 builds **waited** for it, and all 6 started in 4 of 4 rounds |
| **Several builds read the leftover lock at the same moment and all rebuild together** | The same planted-lock rounds, with Loom's messages made visible | Wrong: exactly 1 build rebuilt and 5 waited, every time |

### Two mistakes in the investigation

- **Loom's messages were hidden at first.** CraftGround starts Gradle with `-w`
  (warnings only), which hides Loom's lock messages. The first launch test
  reported "no lock messages", but that proved nothing. `launch_test.py` now
  removes `-w` and saves each instance's full Gradle output to `launch_logs/`
  (kept out of git; the important lines are quoted below).
- **The early failure rate was too high.** It came from about 11 rounds,
  including the two natural benchmark failures, at least one of which followed a
  killed run (see collision 3). Over 22 rounds the
  shared folder failed about 1 in 4 to 1 in 7, depending on what you count.

### What the logs actually showed: four collisions

Each failure was a different collision, depending on which one happened to strike
first.

| # | Collision | What happens | Seen in | Fixed by |
| --- | --- | --- | --- | --- |
| 1 | **Same world name** | Minecraft names a new save "New World (N)" using the next free number. It checks for a free name, then creates the folder, as two separate steps. Two instances checked at the same moment and both took "New World (149)". The second could not open it and never started. | First launch test, arena round 1 | Separate folders (each copy has its own `saves\`) |
| 2 | **Same logging-settings file** | Every launch rewrites Gradle's logging config file (`generateLog4jConfig`) in the game folder. Windows locks a file while it is being written, so two builds at once means one fails. | First launch test, arena round 3; alternating test, shared round 6 | Separate folders |
| 3 | **Leftover Loom lock after a killed run** | Loom writes its process ID into a lock file while it sets up. If that build is killed (Ctrl+C, out of memory, a crash, a test stopped by hand), the lock stays behind. The next launch sees a lock from a dead process and rewrites Loom's cache **in Gradle's shared cache folder** while other builds are reading it. A failed build also writes "disowned" into the lock, which can set off the same thing on the next launch. | Alternating test, separate round 1, right after the previous test was force-stopped. The 23:04 natural benchmark failure also came right after a run was killed by hand, and its cache files were rewritten during that launch. | Deleting stale locks before every launch (`clear_stale_loom_locks`) |
| 4 | **Six first-time builds at once** | A brand-new folder copy compiles the mod's Kotlin code on its first launch. Gradle sends Kotlin compiling from every build on the machine to **one shared Kotlin compiler process**, and six at once ran it out of memory. 3 of 6 builds failed. | First check of the final setup, on fresh copies | Building new copies one at a time, once (`instances.py`) |

#### The error messages

These are the lines from each failed instance's Gradle output that identify the
collision (the full logs stayed on the test machine).

**1. Same world name.** Two instances' logs both show `New World (149)`; the
second one then fails:

```
[23:33:35] [Render thread/WARN] (Minecraft) Failed to create access for New World (149)
java.io.IOException: The process cannot access the file because another process has locked a portion of the file
```

**2. Same logging-settings file:**

```
> Task :generateLog4jConfig FAILED
* What went wrong:
Execution failed for task ':generateLog4jConfig'.
> Failed to generate log4j config
```

**3. Leftover Loom lock** (pid 35752 was the build killed when the previous test
was stopped):

```
"Lock for cache='C:\Users\njdon\.gradle\caches\fabric-loom', project=':'" is currently held by pid '35752'.
Locking process does not exist, assuming abrupt termination and deleting lock file.
Found existing cache lock file (ACQUIRED_PREVIOUS_OWNER_MISSING), rebuilding loom cache. This may have been caused by a failed or canceled build.
java.nio.file.FileSystemException: C:\Users\njdon\.gradle\caches\fabric-loom\1.21\net.fabricmc.yarn.1_21.1.21+build.9-v2\mappings.jar: The process cannot access the file because it is being used by another process
> Failed to setup Minecraft, java.io.UncheckedIOException: Failed to setup mappings: net.fabricmc:yarn:1.21+build.9
```

The 23:04 natural benchmark failure ended with
`> Failed to setup Minecraft, java.lang.RuntimeException: Failed to remap minecraft`;
its Gradle output was not saved, so the lines before that are unknown.

**4. Six first-time builds at once:**

```
Details: Internal error in file lowering: java.lang.OutOfMemoryError: GC overhead limit exceeded
Execution failed for task ':compileKotlin'.
```

For comparison, this is what the **other** builds print when Loom's lock works as
intended (one build rebuilds, the rest wait):

```
"Lock for cache='C:\Users\njdon\.gradle\caches\fabric-loom', project=':'" is currently held by pid '27476'.
Waiting for lock to be released...
```

Collisions 1 and 2 are why separate folders were needed. Collisions 3 and 4 show
up because of what stays shared even with separate folders: Gradle's cache and its
Kotlin compiler.

### Why it only happens at startup, and why two different fixes work

All four collisions happen while instances are **starting**, never mid-run. The
difference is what an instance does with the shared folder in each phase:

| Phase | What an instance does in the folder | Can two instances clash? |
| --- | --- | --- |
| **Startup** (first 1 to 2 min) | **Writes** the logging-settings file, **picks a name and creates** its world save, and Loom may **write** its lock and cache | **Yes**, if two do it in the same split second |
| **Running** (hours) | **Reads** the game code, and **writes only into its own world folder** (one instance's "New World (149)", another's "New World (150)") | **No** |

Reading is not the problem: any number of programs can read the same file at
once. Clashes come from two programs **writing** the same file, or **claiming**
the same name, at the same moment. And that only happens during startup.

That is why both fixes work:

- **One at a time** keeps the shared folder, but makes sure the startup steps
  never overlap. Instance 1 writes the settings file, takes "New World (149)" and
  finishes starting before instance 2 begins; instance 2 then sees 149 is taken and
  picks 150. After startup they only read shared files and write their own worlds,
  so running side by side is fine.
- **Separate folders** let the startup steps overlap, because each instance writes
  its **own** settings file and creates worlds in its **own** saves folder. There is
  nothing shared to clash over (apart from Gradle's cache, which collisions 3 and 4
  cover).

An analogy: six people must each sign in on a sheet and then take a desk. The
problem is all six grabbing the one sheet at once, so two write on the same line.
"One at a time" has them sign in one after another; once seated, they work at their
own desks without bothering each other. "Separate folders" gives each person their
own sign-in sheet, so all six can sign in at once.

## Solutions we considered

| Option | How it works | Status |
| --- | --- | --- |
| **Launch one at a time** (`launch_lock.py`) | Each instance waits until the previous one is fully up | **Works, but slow:** 6 natural instances started fine in 10.3 min (`benchmark.py --serial-launch`). That delay comes back on **every** run (~7 min arena, ~10 min natural). Kept as a fallback flag. |
| **Separate folder per instance** (`instances.py`) | Each instance runs from its own copy of the game folder, via CraftGround's `env_path` option | **Implemented.** Fixes collisions 1 and 2, and keeps startup as fast as launching everyone together. |
| **Delete stale Loom locks before launching** | Remove lock files whose owner process is gone | **Implemented.** Fixes collision 3, for both the shared folder and separate folders. |
| **Build new copies one at a time** | The first-time compile of each new copy takes turns, then the copy is marked built | **Implemented.** Fixes collision 4. Happens once per port. |
| Fixed gap between launches | Start each instance e.g. 20 s after the previous | Not tried: guesswork, since a slow launch can still overlap |
| Retry failed launches | Let collisions happen, restart the failed instance | Not tried: some failures hang instead of erroring, so it would need timeouts. Messy. |
| Separate Gradle cache per copy too | Point each copy at its own Gradle cache | Not needed: the lock cleanup and build queue cover what is still shared, and Gradle's cache is designed to be shared |

## Results

### 6 instances launched at the same moment (arena unless noted)

| Setup | Rounds | Rounds with a failed instance | Slowest launch per round |
| --- | --- | --- | --- |
| Shared folder, first test (arena) | 4 | 2 (world name, settings file) | ~100 s |
| Shared folder, first test (natural) | 4 | 0 | ~230 s |
| Shared folder, natural benchmark runs | 2 | 2 (23:04: remap crash right after a killed run; 18:38: cause not recorded) | – |
| Shared folder, arena benchmark run | 1 | 0 | – |
| Shared folder, alternating test | 11 | 1 (settings file) | 98 s average |
| **Shared folder total** | **22** | **5** (3 ordinary collisions, 1 after a killed run, 1 not recorded) | |
| Separate folders, first tests | 14 (13 arena, 1 natural) | 0 | ~100 s arena, 192 s natural |
| Separate folders, alternating test | 11 | 1 (leftover lock from a test stopped by hand) | 96 s average |
| **Separate folders total** | **25** | **1** (leftover lock, now cleaned up automatically) | |
| Shared folder, with a planted leftover lock | 4 | 0 (1 build rebuilt, 5 waited) | ~120 s |
| One at a time, shared folder (natural benchmark) | 1 | 0 | 620 s to start all 6 |

The alternating test ran shared and separate rounds back to back, swapping which
went first each round, so both saw the same conditions. If separate folders made no
difference, 24 clean rounds out of 25 would be very unlikely: at the shared
folder's ordinary collision rate (3 in 22), the chance of 24 clean rounds in a row
is about 3%.

### The final setup, checked end to end

Six instances started through `train.py`'s own code on its default ports
(8023 to 8073):

| Run | What happened | All 6 ready after |
| --- | --- | --- |
| First (no copies yet) | Created 6 copies, built them one at a time (~35 s each), launched all 6 together, played 100 steps | 351 s (one time only) |
| Second | Skipped building, removed each copy's old world, launched all 6 together, played 100 steps | 140 s |

The 140 s includes Python startup; the launch itself matches the ~100 s from the
launch tests.

### Natural world, 6 instances (benchmark)

Once all 6 could start (one at a time, before separate folders existed), the
natural world ran at **263 steps/s** total (44 per instance), against 165 with 4
instances. See BENCHMARK.md.

## What is implemented

### `instances.py`

`env_path_for(port)` returns the folder the instance on that port should run from,
ready to launch:

1. **Copy:** `C:\Users\njdon\craftground_instances\mc121_port<port>`, copied from
   CraftGround's installed folder if it does not exist yet (~50 MB; ~210 MB after
   the build below, mostly a C++ library CMake downloads). Old world saves
   and logs are left out, as are CMake build files, which store the original
   folder's location.
2. **Build once:** the first time, it compiles the mod and its native library in
   the copy (`gradlew classes compileCpp`), **one copy at a time** using a lock that
   all processes share. Then it writes a `.craftground_built` marker so it never
   builds again. The output goes to `build_first_use.log` in the copy.
   Before trusting the marker, it checks that CMake's cache (`CMakeCache.txt`)
   points at this folder. CMake stores the full folder path and refuses to build
   once a folder is moved or renamed, so a moved copy is rebuilt automatically (see
   "Moving or renaming a copy" below).
3. **Clean old worlds:** it deletes the copy's old world saves. CraftGround never
   removes them; the shared folder had 194 saves, about 1 GB.
4. **Clean stale Loom locks** (`clear_stale_loom_locks`): it deletes locks in
   `~/.gradle/caches/fabric-loom/` that say "disowned" or name a process that no
   longer exists. Locks of running builds are kept. An empty lock is only removed
   once it is a minute old, since a build might be writing it right then.

Folders are named by **port**, not instance number, because every running instance
needs its own port anyway. Training and an evaluation on different ports therefore
never share a folder.

### Changes to the scripts

- `tree_env.make_env` and `natural_env.make_env` take `env_path` (default `None` =
  CraftGround's shared folder).
- `train.py`, `evaluate.py` and `benchmark.py` give each instance
  `env_path_for(port)`, and call `clear_stale_loom_locks()` at startup. Each has a
  `--shared-folder` flag to go back to the old behaviour.
- `benchmark.py --serial-launch` still offers the one-at-a-time launch
  (`launch_lock.py`).
- `launch_test.py` is the test used here: `--separate-folders`, `--alternate`,
  `--serial`, `--plant-disowned`. It writes `results/launch_test.csv` and saves each
  instance's Gradle output to `launch_logs/` (ignored by git).

### Upkeep

- **New port or more instances:** those copies are created and built once on first
  use (~35 s each, one at a time).
- **After upgrading CraftGround:** delete `C:\Users\njdon\craftground_instances\`
  (or call `instances.refresh_instances()`) so the copies are made again from the
  new version.
- **Disk:** about 210 MB per built copy (Minecraft itself and its assets stay in
  Gradle's shared cache, so they are not copied). Ports 8023 to 8073 (training and evaluation) are
  built. The benchmark's default ports (8523 to 8573) build on first use.
- **Moving or renaming a copy:** don't. The test copies were first named
  `mc121_0` to `mc121_5` and later renamed to `mc121_port8523` and so on, with the
  built marker added by hand. CMake's cache still pointed at `mc121_0`, so every
  launch from them failed with *"CMakeCache.txt directory ... is different"*, and the
  marker stopped `instances.py` from rebuilding. It surfaced when a benchmark run on
  the default ports could not start Minecraft. Those copies were deleted, and
  `instances.py` now detects a moved copy and rebuilds it. To start over, delete a
  copy (or the whole folder) and it is made fresh on next use.

## Still open

- **The 23:04 remap crash in the shared folder** is only partly explained. It came
  right after a killed run (collision 3), but the planted-lock tests showed the
  shared folder usually copes with a leftover lock: the other builds wait. Something
  else had to coincide. The remaining suspect is a very narrow gap in how Loom
  writes its lock file: it creates the file empty, then writes the ID, and a build
  reading it in between treats the lock as free. That could not be forced on demand.
- **The 18:38 natural failure's cause was not recorded:** Gradle's output was
  discarded at the time.
- **Separate folders were tested mostly with the arena:** 24 arena rounds and 1
  natural round. The launch is the same for both worlds. Natural rounds with 6
  instances were limited by memory (~17 GB), which once got the test stopped.
- **CraftGround does not clean up leftover Minecraft processes on Windows.** Its
  cleanup looks in `/tmp`, which only exists on Linux. Runs and tests sometimes left
  a Minecraft instance running after exiting, which had to be killed by hand.
  `benchmark.py` kills its own leftovers; `train.py` and `evaluate.py` do not yet.
- **The startup collisions are not reported to CraftGround yet:** none of the
  four is in its GitHub issues, so a report there could help.
