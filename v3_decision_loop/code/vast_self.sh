#!/bin/bash
# Act on this Vast.ai instance with its own per-instance key, which Vast puts in PID 1's environment
# (CONTAINER_ID, CONTAINER_API_KEY). Never ship an account key to an instance. On 10 Oct 2026 the API answered the
# instance key erratically (the same request 401 one moment, 200 the next, on both API hosts), so every call is
# retried over both hosts and both ways of passing the key until one answers 200.
#   bash vast_self.sh check|stop|destroy|label <text>
env1() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
CID=${CONTAINER_ID:-$(env1 CONTAINER_ID)}; KEY=${CONTAINER_API_KEY:-$(env1 CONTAINER_API_KEY)}
[ -n "$CID" ] && [ -n "$KEY" ] || { echo "vast_self: no CONTAINER_ID / CONTAINER_API_KEY"; exit 2; }
J='Content-Type: application/json'
call() {   # method [json]
  for round in 1 2 3 4 5 6 7 8; do
    for host in cloud.vast.ai console.vast.ai; do
      for form in header param; do
        url="https://$host/api/v0/instances/$CID/"
        if [ $form = header ]; then
          code=$(curl -s -m 30 -o /tmp/vast_self.out -w "%{http_code}" -X "$1" -H "Authorization: Bearer $KEY" -H "$J" ${2:+-d "$2"} "$url")
        else
          code=$(curl -s -m 30 -o /tmp/vast_self.out -w "%{http_code}" -X "$1" -H "$J" ${2:+-d "$2"} "$url?api_key=$KEY")
        fi
        if [ "$code" = 200 ]; then
          if [ "$1" = GET ]; then echo "$1 $host ($form, round $round): HTTP 200"
          else echo "$1 $host ($form, round $round): HTTP 200 $(head -c 200 /tmp/vast_self.out)"; fi
          return 0
        fi
      done
    done
    sleep 5
  done
  echo "$1 instance $CID: no 200 after 8 rounds (last HTTP $code)"; return 1
}
case "$1" in
  check)   call GET ;;
  label)   call PUT "{\"label\": \"$2\"}" ;;
  stop)    echo "$(date '+%F %T') stop $CID"; call PUT '{"state": "stopped"}' ;;
  destroy) echo "$(date '+%F %T') destroy $CID"; call DELETE ;;
esac
