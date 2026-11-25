#!/usr/bin/env bash
# Setup worktrees + patch, then start the pipeline detached.
S=$(dirname $0); D=${SM75_ROOT:-$HOME/research/sm75-batch-invariance}; E=$D/evidence-p61
bash $S/setup_dp4a.sh || { echo SETUP FAILED; exit 1; }
setsid nohup bash $E/scripts/run_all.sh > $E/run_all.out 2>&1 < /dev/null &
sleep 2; echo "launched pid=$(pgrep -f 'run_all[.]sh')"
