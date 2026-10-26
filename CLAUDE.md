never do things I have not told you, like adding extension without me telling you.

DO NOT GO BEYOND WHAT I ASK FOR

Run  things in a non-blocking way; don't wait for batched jobs to finish
Always run on a free node, unless told otherwise, if a job can run on ngog, then run it there

When I ask you to do something, do it completley, do not save things for future promts. Just drive it to completion -- this is a top priority
When submitting slurm jobs, start the reply with the job id (e.g. "Job 12345: ...").
Compute nodes cannot see /tmp (or the scratchpad): scripts and outputs of slurm jobs must live in the project directory.
Slurm log files must start with the job id: #SBATCH --output=logs/%j_<name>.out (the job name itself can be anything). The logs folder holds only logs, never scripts; put job scripts next to the code they run (series_*/, diagnostics/).


please write a systematic plan of all tests and experiments and as you execute them, please save the data and plots, do not do adhoc tests