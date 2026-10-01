#!/bin/sh
case "${SNAPJUDGE_BINARY:-}" in
  /*) ;;
  *) exit 0 ;;
esac
[ -x "$SNAPJUDGE_BINARY" ] || exit 0
case "${1:-}" in
  SessionStart|UserPromptSubmit|PreToolUse|PostToolUse)
    exec "$SNAPJUDGE_BINARY" adapter claude-code --event "$1" ;;
  *) exit 0 ;;
esac
