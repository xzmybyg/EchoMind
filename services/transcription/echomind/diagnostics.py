"""Local fatal-error stacks only; never record credentials, audio or answers."""
import faulthandler
import os
from pathlib import Path
import time


_fault_file = None


def enable_fault_log(directory=None):
    global _fault_file
    if _fault_file is not None:
        return
    folder = Path(directory) if directory is not None else Path(
        os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'EchoMind/logs'
    handle = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'native-crash.log'
        if path.exists() and path.stat().st_size > 128 * 1024:
            path.replace(folder / 'native-crash.previous.log')
        handle = path.open('a', encoding='utf-8', buffering=1)
        handle.write(f'\nEchoMind start {time.strftime("%Y-%m-%d %H:%M:%S")} pid={os.getpid()}\n')
        faulthandler.enable(file=handle, all_threads=True)
        _fault_file = handle
    except (OSError, RuntimeError):
        if handle is not None:
            handle.close()
