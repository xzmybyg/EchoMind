"""Create a runtime-only GitHub release asset from a built desktop folder."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from PyInstaller.archive.readers import CArchiveReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "transcription"))
from echomind.updater import ASSET_NAME, COMPONENTS, version_tuple
from echomind.version import VERSION


def package_update(package: Path, output: Path):
    version_tuple(VERSION)
    for name in COMPONENTS:
        if not (package / name).exists():
            raise ValueError(f"Missing build component: {name}")
    embedded = CArchiveReader(str(package / "EchoMind.exe")).open_embedded_archive("PYZ.pyz").extract("echomind.version")
    if VERSION not in embedded.co_consts:
        raise ValueError("The EXE version does not match the source version; rebuild before packaging.")
    output.mkdir(parents=True, exist_ok=True)
    asset = output / ASSET_NAME
    with zipfile.ZipFile(asset, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("update.json", json.dumps({"version": VERSION, "format": 1}))
        for name in COMPONENTS:
            component = package / name
            paths = [component] if component.is_file() else sorted(component.rglob("*"))
            for path in paths:
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    with asset.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"Version: v{VERSION}\nAsset: {asset}\nSHA256: {checksum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package_update(args.package, args.output)
