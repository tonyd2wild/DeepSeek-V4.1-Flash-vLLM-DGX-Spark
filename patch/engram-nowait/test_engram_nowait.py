#!/usr/bin/env python3
"""Dependency-free forced-path checks for the patched Engram row reader."""

import argparse
import ast
import errno
from pathlib import Path

_EXTRACT = {
    "_KaiNowaitUnsupported",
    "_kai_pread_rows",
    "_kai_pread_nowait",
    "_kai_pread_pending",
    "_kai_nowait_read",
    "_kai_parallel_read",
}


class _Future:
    def __init__(self, fn, *args):
        try:
            self.value = fn(*args)
            self.error = None
        except BaseException as exc:
            self.value = None
            self.error = exc

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _Pool:
    def submit(self, fn, *args):
        return _Future(fn, *args)


class _ScriptedOS:
    RWF_NOWAIT = 8

    def __init__(self, events):
        self.events = list(events)

    def preadv(self, _fd, buffers, _offset, flags=0):
        if not self.events:
            raise AssertionError("unexpected preadv call")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        if event == 0:
            return 0
        view = buffers[0]
        data = bytes(event)
        view[: len(data)] = data
        return len(data)


def _load(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _EXTRACT
    ]
    missing = _EXTRACT - {node.name for node in nodes}
    if missing:
        raise SystemExit(f"patched source is missing: {sorted(missing)}")
    namespace = {
        "_kai_errno": errno,
        "_KAI_NOWAIT": _ScriptedOS.RWF_NOWAIT,
        "_KAI_NOWAIT_UNSUPPORTED": frozenset(
            value
            for value in (
                errno.EINVAL,
                errno.ENOSYS,
                getattr(errno, "ENOTSUP", None),
                getattr(errno, "EOPNOTSUPP", None),
            )
            if value is not None
        ),
        "_KAI_THREADS": 2,
        "_KAI_CHUNK": 16,
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["_kai_pool"] = lambda: _Pool()
    return namespace


def _run(namespace, events, expected=b"row!"):
    fake_os = _ScriptedOS(events)
    namespace["_kai_os"] = fake_os
    output = bytearray(len(expected))
    namespace["_kai_parallel_read"](
        [(3, 4096, [0], len(expected), memoryview(output))]
    )
    if output != expected:
        raise AssertionError(f"output mismatch: {bytes(output)!r}")
    if fake_os.events:
        raise AssertionError(f"unused scripted events: {fake_os.events!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patched_engram", type=Path)
    args = parser.parse_args()
    namespace = _load(args.patched_engram)
    eagain = lambda: BlockingIOError(errno.EAGAIN, "not resident")
    unsupported = lambda: OSError(errno.EINVAL, "NOWAIT unsupported")

    cases = [
        ("cached-hit", [b"row!"]),
        ("one-eagain", [eagain(), b"row!"]),
        ("persistent-eagain", [eagain(), eagain(), b"row!"]),
        ("partial-read", [b"ro", b"w!"]),
        ("unsupported-operation", [unsupported(), b"row!"]),
        ("nowait-zero-fallback", [0, 0, b"row!"]),
    ]
    passed = 0
    for name, events in cases:
        _run(namespace, events)
        passed += 1
        print(f"PASS {name}")

    try:
        _run(namespace, [0, 0, 0])
    except OSError as exc:
        if "short read" not in str(exc):
            raise
    else:
        raise AssertionError("terminal blocking zero did not raise")
    passed += 1
    print("PASS terminal-short-read")
    print(f"{passed}/7 checks passed")


if __name__ == "__main__":
    main()
