# Getting Started with Slurm

Slurm is the job scheduler used on most HPC clusters. As a user, your work falls into a few simple categories: **look at the cluster**, **submit jobs**, **watch your jobs**, and **control them**. This tutorial covers the commands that handle 95% of everyday use, plus a few things that make life easier once you're comfortable.

---

## 1. Looking at the cluster: `sinfo`

```bash
sinfo
```

Your starting point. It lists the cluster's **partitions** (queues) and the state of their nodes:

```
PARTITION  AVAIL  TIMELIMIT   NODES  STATE   Nodelist
gpu        up      2-00:00:00      3   mix    gpu[01-02]
gpu        up      2-00:00:00      1   idle   gpu03
cpu        up    infinite         10   alloc  cpu[01-10]
cpu        up    infinite          4   idle   cpu[11-14]
```

- **PARTITION** — the queue jobs are submitted to.
- **AVAIL** — `up` = accepting jobs, `down` = disabled.
- **TIMELIMIT** — maximum runtime for jobs there.
- **NODES / STATE / Nodelist** — how many nodes are in each state, and their names (often compressed like `gpu[01-03]`).

### Node states


| State   | Meaning                                           |
| ------- | ------------------------------------------------- |
| `idle`  | Completely free                                   |
| `mix`   | Partially used — still has free cores or memory   |
| `alloc` | Fully allocated                                   |
| `drain` | Off-limits — no new jobs (maintenance or failure) |


> **Tip:** `mix` nodes often still have idle cores that can host small jobs — don't skip them when the cluster looks busy.

Useful variations:

```bash
sinfo -N                          # one line per node instead of per partition
sinfo -N -o "%N %T %C"            # node name, state, and CPU breakdown
sinfo -p gpu                      # just one partition
```

The `%C` format code shows CPUs as `Allocated/Idle/Other/Total`.

---

## 2. Submitting jobs: `sbatch`, `srun`, `salloc`

### `sbatch` — submit a batch script (the normal way)

```bash
sbatch myscript.sbatch
```

Your script is a normal shell script with a header telling Slurm what it needs:

```bash
#!/bin/bash
#SBATCH --job-name=train        # name shown in squeue
#SBATCH --partition=gpu         # which queue
#SBATCH --nodes=1               # number of nodes
#SBATCH --cpus-per-task=8       # cores per task
#SBATCH --gpus=1                # GPUs
#SBATCH --time=04:00:00         # walltime limit
#SBATCH --mem=32G               # memory per node
#SBATCH --output=slurm-%j.out   # stdout log (%j = job ID)

python train.py
```

Slurm queues the script and runs it when the resources are free. It prints your **job ID**, which you'll use for everything else.

You can override options on the command line: `sbatch --gpus=2 --time=08:00:00 myscript.sbatch`

### `srun` — run a single command inside an allocation

```bash
srun --cpus-per-task=4 python preprocess.py
```

Runs the command as a Slurm job and streams its output to your terminal. Inside a batch script, `srun` launches each parallel task/step; from your login shell it acts as a lightweight one-command submission.

### `salloc` — get an interactive shell on a compute node

```bash
salloc --nodes=1 --cpus-per-task=4 --time=01:00:00
```

You get a shell directly on a compute node once the resources are granted — ideal for debugging, profiling, or running an interactive tool like Jupyter or a Python REPL. Exit the shell (or hit the time limit) to release the node.

> Which to use? `sbatch` for production runs, `salloc`/`srun` for anything interactive or exploratory.

---

## 3. Watching your jobs: `squeue`

```bash
squeue                      # all jobs in the queue
squeue -u $USER             # only your jobs
squeue -j 12345             # one job by ID
squeue -p gpu               # one partition
squeue -w gpu01             # jobs on a specific node
```

Typical output:

```
  JOBID PARTITION   NAME     USER ST   TIME  NODES NODELIST(REASON)
    1234 gpu        train   alice  R  01:23      1 gpu01
    1235 gpu        eval    alice PD  0:00      1 (Priority)
```

- **ST** — state: `R` = running, `PD` = pending, `CG` = completing, `F` = failed.
- **TIME** — walltime used so far.
- **NODELIST(REASON)** — where the job runs, or *why* it's waiting (e.g. `Resources`, `Priority`, `QOSMaxGpuPerUserLimit`).

Watch a job live: `watch squeue -u $USER` (refreshes every 2 s).

---

## 4. Controlling jobs: `scancel`

```bash
scancel 12345               # cancel one job
scancel -u $USER            # cancel ALL your jobs (use with care!)
scancel -n train            # cancel by job name
scancel --state=PD -u $USER # cancel all your pending jobs
```

A job stuck or misbehaving? `scancel` is the tool. On a pending job it simply removes it from the queue.

---

## 5. Checking results after the fact: `sacct`

```bash
sacct                       # your jobs today
sacct -j 12345              # details of one job
sacct -j 12345 -o JobID,JobName,State,Elapsed,MaxRSS,AllocTRES%40
```

Especially useful for jobs that already finished — `squeue` only shows queued/running jobs. Key fields:

- **State** — `COMPLETED`, `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, etc.
- **Elapsed** — total walltime.
- **MaxRSS** — peak memory used (helps you request the right `--mem` next time).
- **AllocTRES** — resources actually granted (cores, GPUs, memory).

---

## 6. Inspecting details: `scontrol`

```bash
scontrol show job 12345     # full details of a job (resources, log paths, workdir, ...)
scontrol show node gpu01    # full details of a node (cores, memory, GPUs, state)
scontrol show partition     # full details of partitions
```

The "tell me everything" command — when a job misbehaves or you need the exact path of the log file Slurm chose, look here.

---

## 7. Testing before you submit

```bash
sbatch --test-only myscript.sbatch
```

Simulates the submission **without queuing it**, and tells you when and where the job would start (or why it wouldn't). A good sanity check for resource requests.

---

## 8. Running many similar jobs: arrays and dependencies

### Job arrays

```bash
sbatch --array=1-10 myscript.sbatch
```

Submits 10 copies of the same script at once. Inside the script, `$SLURM_ARRAY_TASK_ID` tells you which one you are:

```bash
seed=$SLURM_ARRAY_TASK_ID
python train.py --seed $seed
```

Output logs can use `%A` (array job ID) and `%a` (task ID): `--output=logs/%A_%a.out`. Cancel with `scancel 12345` (all) or `scancel 12345_7` (one).

### Dependencies — run jobs in sequence

```bash
id1=$(sbatch --parsable preprocess.sbatch)
id2=$(sbatch --parsable --dependency=afterok:$id1 train.sbatch)
```

- `afterok` — start only if the first job **succeeded** (most common).
- `afterany` — start after the first finishes, regardless of outcome.
- `afternotok` — start only if the first **failed** (e.g. "cleanup if it crashes").

`--parsable` prints just the job ID, which makes chaining in a shell script easy. A pending job waiting on a dependency shows the reason `(Dependency)` in `squeue`.

---

## 9. Inside your job: Slurm's environment variables

Slurm sets several variables your script (or code) can use:


| Variable                        | Meaning                                                 |
| ------------------------------- | ------------------------------------------------------- |
| `$SLURM_JOB_ID`                 | The job's ID                                            |
| `$SLURM_ARRAY_TASK_ID`          | Which array task this is                                |
| `$SLURM_JOB_CPUS_PER_NODE`      | Cores allocated on each node                            |
| `$SLURM_GPUS_PER_TASK`          | GPUs per task                                           |
| `$SLURM_SUBMIT_DIR`             | Directory where you submitted from                      |
| `$SLURM_PROCID` / `$LOCAL_RANK` | Process / local rank (useful with `srun` + PyTorch DDP) |


Two practical habits:

- **Work in `$SLURM_SUBMIT_DIR`**: jobs start there by default; use `cd $SLURM_SUBMIT_DIR` explicitly if your script changes directories.
- **Load modules in the script**, not your shell — the job environment should be reproducible on its own.

---

## 10. Common pitfalls

- **Wrong walltime** — jobs are killed at `--time` with no checkpoint. Check `sacct` for `TIMEOUT` and pad your estimate.
- **Too little memory** — `OUT_OF_MEMORY` kills. Check `MaxRSS` in `sacct` and request a bit more than peak.
- **Not enough cores requested** — your code may spawn more threads than allocated cores, causing contention. Set `--cpus-per-task` to match, and export `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` in the script.
- **Compressed node lists** — `gpu[01-03]` means `gpu01`, `gpu02`, `gpu03`; it's not a strange hostname.
- **Forgetting `--gpus`** — on many clusters, GPUs must be requested explicitly; a job without `--gpus` lands on a CPU-only node and CUDA fails.

---

## The everyday workflow

1. `sinfo` — check partitions and how busy they are.
2. `sbatch myscript.sbatch` — submit your job.
3. `squeue -u $USER` — watch it start; the REASON column tells you why it waits.
4. `scancel <id>` — kill it if something's wrong.
5. `sacct -j <id>` — after it ends, check it completed and how much memory it really used.

## Quick reference


| Goal                         | Command                                   |
| ---------------------------- | ----------------------------------------- |
| Cluster / partition overview | `sinfo`                                   |
| Per-node detail              | `sinfo -N` or `scontrol show node <node>` |
| Submit a batch script        | `sbatch myscript.sbatch`                  |
| Submit many variants         | `sbatch --array=1-10 myscript.sbatch`     |
| Chain jobs in sequence       | `sbatch --dependency=afterok:<id> ...`    |
| Interactive shell on a node  | `salloc`                                  |
| Run one command as a job     | `srun <command>`                          |
| Check the queue / your jobs  | `squeue` / `squeue -u $USER`              |
| Cancel a job                 | `scancel <id>`                            |
| Results of a finished job    | `sacct -j <id>`                           |
| Full job or node details     | `scontrol show job <id>`                  |
| Dry-run a submission         | `sbatch --test-only script.sbatch`        |
