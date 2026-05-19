#!/usr/bin/env bash

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

print_stage() {
  printf '==> %s\n' "$1"
}

require_commands() {
  local missing=()
  local command_name

  for command_name in "$@"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      missing+=("${command_name}")
    fi
  done

  if ((${#missing[@]} > 0)); then
    die "missing required command(s): ${missing[*]}"
  fi
}

require_value() {
  local option=$1
  local value=${2:-}

  [[ -n "${value}" && "${value}" != --* ]] || die "${option} requires a value"
}

expand_path() {
  local path=$1
  local home_prefix

  printf -v home_prefix '%b' '\x7e/'

  if [[ "${path:0:2}" == "${home_prefix}" ]]; then
    printf '%s/%s\n' "${HOME}" "${path#~/}"
    return
  fi

  printf '%s\n' "${path}"
}
