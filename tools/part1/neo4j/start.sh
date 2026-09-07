#!/bin/zsh
# Start Neo4j (brew, community 2026.07.1) as a user process in the background.
# Writes two pid files under part1-tools/env:
#   neo4j.pid           the JVM that listens on bolt 7687 (the child; stop.sh kills this one)
#   neo4j-launcher.pid  the NeoBoot launcher that started it (parent, exits when the child exits)
# Config: /opt/homebrew/Cellar/neo4j/2026.07.1/libexec/conf/neo4j.conf
# Log:    part1-tools/env/neo4j.log (console output)
set -e
T=/Users/muralisid/github_other/part1-tools
cd $T/neo4j
if [ -f $T/env/neo4j.pid ] && kill -0 "$(cat $T/env/neo4j.pid)" 2>/dev/null; then
  echo "already running, pid $(cat $T/env/neo4j.pid)"; exit 0
fi
nohup /opt/homebrew/opt/neo4j/bin/neo4j console > $T/env/neo4j.log 2>&1 &
LAUNCHER=$!
echo $LAUNCHER > $T/env/neo4j-launcher.pid
# Wait for "Started." (up to 90 s), then record the JVM that owns port 7687.
for i in $(seq 1 90); do
  if grep -q "Started\." $T/env/neo4j.log 2>/dev/null; then break; fi
  sleep 1
done
CHILD=$(lsof -nP -iTCP:7687 -sTCP:LISTEN -t 2>/dev/null | head -1)
if [ -z "$CHILD" ]; then
  CHILD=$(pgrep -P $LAUNCHER | head -1)
fi
if [ -z "$CHILD" ]; then
  echo "neo4j did not come up; see $T/env/neo4j.log"; exit 1
fi
echo $CHILD > $T/env/neo4j.pid
echo "started: launcher pid $LAUNCHER, server pid $CHILD (bolt 7687), log $T/env/neo4j.log"
