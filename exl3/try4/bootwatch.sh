#!/bin/bash
# bootwatch.sh CUT PREPLOG (Mac): poll bootstatus.sh on Asusi every 30 s, print only lines never seen before; exit when the head is up.
CUT=${1:?cut}; PL=${2:?preplog}; SEEN=$(mktemp); n=0
while [ $n -lt 90 ]; do
  out=$(perl -e 'alarm 60; exec @ARGV' ssh -i $HOME/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o ConnectTimeout=8 tonyspark3@100.90.25.78 "bash ~/bootstatus.sh $CUT $PL" 2>/dev/null)
  [ -n "$out" ] || out="(status poll failed)"
  while IFS= read -r l; do grep -qxF -- "$l" "$SEEN" || { echo "$l"; printf '%s\n' "$l" >> "$SEEN"; }; done <<< "$out"
  echo "$out" | grep -q "^\[bluey\].*Application startup complete" && { echo "HEAD UP"; exit 0; }
  n=$((n+1)); sleep 30
done; echo "watch timeout after 45 min"
