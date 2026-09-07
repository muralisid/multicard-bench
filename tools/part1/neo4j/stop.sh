#!/bin/zsh
# Stop Neo4j started by start.sh. Sends SIGTERM to the server JVM (neo4j.pid,
# the process on bolt 7687) and waits for it and the launcher to exit.
# Shutdown takes about 10 to 15 seconds.
T=/Users/muralisid/github_other/part1-tools
if [ ! -f $T/env/neo4j.pid ]; then
  echo "no pid file"; exit 0
fi
PID=$(cat $T/env/neo4j.pid)
if kill "$PID" 2>/dev/null; then
  for i in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && echo "pid $PID still running after 60 s" || echo "stopped pid $PID"
else
  echo "no process for pid $PID"
fi
if [ -f $T/env/neo4j-launcher.pid ]; then
  L=$(cat $T/env/neo4j-launcher.pid)
  for i in $(seq 1 15); do
    kill -0 "$L" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$L" 2>/dev/null; then kill "$L" 2>/dev/null; echo "killed launcher pid $L"; fi
  rm -f $T/env/neo4j-launcher.pid
fi
# Wait until nothing listens on 7687 so a following start.sh can bind it.
for i in $(seq 1 60); do
  [ -z "$(lsof -nP -iTCP:7687 -sTCP:LISTEN -t 2>/dev/null)" ] && break
  sleep 1
done
rm -f $T/env/neo4j.pid
