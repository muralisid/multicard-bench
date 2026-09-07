#!/bin/zsh
# Stop the litellm proxy started by start.sh.
T=/Users/muralisid/github_other/part1-tools
if [ -f $T/env/litellm.pid ]; then
  kill "$(cat $T/env/litellm.pid)" 2>/dev/null && echo "stopped pid $(cat $T/env/litellm.pid)" || echo "no process for pid $(cat $T/env/litellm.pid)"
  rm -f $T/env/litellm.pid
else
  echo "no pid file"
fi
