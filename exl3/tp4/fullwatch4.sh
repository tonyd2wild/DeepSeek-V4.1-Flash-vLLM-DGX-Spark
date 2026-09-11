#!/bin/bash
# fullwatch4.sh CUT PREPLOG LABEL (Mac): ONE poller for a 4-Spark boot: bootstatus4 lines from Asusi plus the post-boot log on
# Bluey (the bench runs there against Reddie's endpoint). Prints only lines never seen before; exits when the post-boot log shows VISION DONE (or after 90 min).
CUT=${1:?cut}; PL=${2:?preplog}; L=${3:?label}; SEEN=$(mktemp); n=0
while [ $n -lt 180 ]; do
  out=$(perl -e 'alarm 60; exec @ARGV' ssh -i $HOME/.ssh/id_ed25519_spark -o IdentitiesOnly=yes -o ConnectTimeout=8 tonyspark3@100.90.25.78 "bash ~/bootstatus4.sh $CUT $PL; ssh -n -i ~/.ssh/id_ed25519_shared -o BatchMode=yes -o ConnectTimeout=6 tonyspark1@192.168.192.1 'grep -aE \"health OK|smoke|coherence|bench start|POST_BOOT DONE|VISION DONE|Traceback\" ~/post_boot_$L.log 2>/dev/null | cut -c1-200 | sed \"s/^/[post] /\"'" 2>/dev/null)
  [ -n "$out" ] || out="(status poll failed)"
  while IFS= read -r l; do grep -qxF -- "$l" "$SEEN" || { echo "$l"; printf '%s\n' "$l" >> "$SEEN"; }; done <<< "$out"
  echo "$out" | grep -q "VISION DONE" && { echo "ALL DONE"; exit 0; }
  n=$((n+1)); sleep 30
done; echo "fullwatch4 timeout after 90 min"
