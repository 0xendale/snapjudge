#!/bin/sh
case "${SNAPJUDGE_BINARY:-}" in
  /*) ;;
  *) exit 1 ;;
esac
[ -x "$SNAPJUDGE_BINARY" ] || exit 1
[ -d "${1:-}" ] || exit 1
exec "$SNAPJUDGE_BINARY" mcp --stdio --workspace "$1"
