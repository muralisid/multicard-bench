#!/usr/bin/env bash
# Fetch the Enron corpus from its canonical CMU distribution.
# Roughly 423 MB. The archive is streamed by the parser, never unpacked.
set -euo pipefail
DEST="data/raw/enron_mail_20150507.tar.gz"
URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
EXPECTED_PREFIX="b3da1b3fe0369ec3140bb4fb"

mkdir -p "$(dirname "$DEST")"
if [ -f "$DEST" ]; then
  echo "already present: $DEST"
else
  echo "downloading $URL"
  curl -sL -o "$DEST" "$URL"
fi

ACTUAL="$(shasum -a 256 "$DEST" | cut -c1-24)"
if [ "$ACTUAL" != "$EXPECTED_PREFIX" ]; then
  echo "checksum mismatch: got $ACTUAL, expected prefix $EXPECTED_PREFIX" >&2
  exit 1
fi
echo "verified: $DEST"
