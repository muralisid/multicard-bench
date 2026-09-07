#!/bin/zsh
# Start the litellm proxy on 127.0.0.1:4000 in the background.
# Needs VERTEX_AI_SERVICE_ACCOUNT_JSON in the environment (login shell has it).
set -e
T=/Users/muralisid/github_other/part1-tools
cd $T/litellm
if [ -f $T/env/litellm.pid ] && kill -0 "$(cat $T/env/litellm.pid)" 2>/dev/null; then
  echo "already running, pid $(cat $T/env/litellm.pid)"; exit 0
fi
export LITELLM_MASTER_KEY="$(cat $T/env/.proxy_key)"
export LITELLM_REQUEST_LOG=$T/env/requests.jsonl
nohup .venv/bin/litellm --config config.yaml --host 127.0.0.1 --port 4000 \
  > $T/env/litellm.log 2>&1 &
echo $! > $T/env/litellm.pid
echo "started pid $(cat $T/env/litellm.pid), log $T/env/litellm.log"
