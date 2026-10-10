#!/bin/bash
# Act on this Vast.ai instance with its own per-instance key, which Vast puts in PID 1's environment
# (CONTAINER_ID, CONTAINER_API_KEY). Never ship an account key to an instance.
#   bash vast_self.sh check|stop|destroy|label <text>
env1() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
CID=${CONTAINER_ID:-$(env1 CONTAINER_ID)}; KEY=${CONTAINER_API_KEY:-$(env1 CONTAINER_API_KEY)}
[ -n "$CID" ] && [ -n "$KEY" ] || { echo "vast_self: no CONTAINER_ID / CONTAINER_API_KEY"; exit 2; }
API=https://console.vast.ai/api/v0/instances/$CID/
J='Content-Type: application/json'
case "$1" in
  check)   curl -s -m 30 -o /dev/null -w "GET instance $CID: HTTP %{http_code}\n" -H "Authorization: Bearer $KEY" "$API" ;;
  label)   curl -s -m 30 -X PUT -H "Authorization: Bearer $KEY" -H "$J" -d "{\"label\": \"$2\"}" "$API"; echo ;;
  stop)    echo "$(date '+%F %T') stop $CID"; curl -s -m 30 -X PUT -H "Authorization: Bearer $KEY" -H "$J" -d '{"state": "stopped"}' "$API"; echo ;;
  destroy) echo "$(date '+%F %T') destroy $CID"; curl -s -m 30 -X DELETE -H "Authorization: Bearer $KEY" "$API"; echo ;;
esac
