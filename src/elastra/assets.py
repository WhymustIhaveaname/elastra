"""Download the third-party assets and build the ProtoMotions reference clips.

    elastra-download-assets                 # everything
    elastra-download-assets --skip-bones    # without the BONES-SEED clips

Every file is checked against the sha256 in ``conf/assets/default.yaml`` and a file
that is already present with the right hash is not downloaded again.  The
BONES-SEED clips need ``HF_TOKEN`` for an account that has accepted the dataset
license; they are only needed to run the ProtoMotions tracker.

Layout written under ``assets/`` (git-ignored)::

    protomotions/  g1_holo_compat.xml, meshes/, unified_pipeline.{onnx,yaml}, LICENSE.md
    host/          g1_23dof.urdf, meshes/, policies/{prone,supine}.onnx, LICENSE
    bones_seed/    the three get-up clips (CSV, not redistributable)
    reference_clips/{supine,side,prone}.npz   built from the CSVs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

from elastra.config import CONF_DIR, ROOT, repo_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected: str | None, *, headers: dict | None = None) -> None:
    if target.is_file() and (expected is None or sha256(target) == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            break
        except OSError as error:
            if attempt == 3:
                raise RuntimeError(f"download failed: {url}: {error}") from error
            time.sleep(2.0 * (attempt + 1))
    if expected is not None and sha256(partial) != expected:
        raise RuntimeError(f"{target.name}: sha256 {sha256(partial)} != expected {expected}")
    partial.replace(target)
    print(f"  {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}", flush=True)


def github_url(repository: str, commit: str, path: str, lfs: bool) -> str:
    owner_repo = repository.removeprefix("https://github.com/")
    host = "media.githubusercontent.com/media" if lfs else "raw.githubusercontent.com"
    return f"https://{host}/{owner_repo}/{commit}/{path}"


def download_repository_files(spec, target: Path, *, meshes_lfs: bool) -> None:
    for name, item in spec.files.items():
        fetch(
            github_url(spec.repository, spec.commit, item.path, bool(item.lfs)),
            target / name,
            item.sha256,
        )
    for name, digest in spec.meshes.items():
        fetch(
            github_url(spec.repository, spec.commit, f"{spec.mesh_path}/{name}", meshes_lfs),
            target / "meshes" / name,
            digest,
        )


def download_host_policies(spec, target: Path) -> None:
    for posture, item in spec.items():
        url = (
            "https://drive.usercontent.google.com/download?"
            f"id={item.google_drive_id}&export=download&confirm=t"
        )
        fetch(url, target / f"{posture}.onnx", item.sha256)


def download_bones_clips(spec, target: Path) -> None:
    wanted = {posture: target / Path(item.member).name for posture, item in spec.clips.items()}
    if all(
        path.is_file() and sha256(path) == spec.clips[posture].sha256
        for posture, path in wanted.items()
    ):
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "BONES-SEED is a gated dataset: accept its license on "
            f"https://huggingface.co/datasets/{spec.dataset} and export HF_TOKEN"
        )
    target.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{spec.dataset}/resolve/{spec.revision}/{spec.archive}"
    members = [item.member for item in spec.clips.values()]
    print(f"  streaming {spec.archive} until {len(members)} clips are found", flush=True)
    curl = subprocess.Popen(
        ["curl", "-sSL", "-H", f"Authorization: Bearer {token}", url],
        stdout=subprocess.PIPE,
    )
    scratch = target / "extract"
    scratch.mkdir(exist_ok=True)
    # --occurrence makes tar stop reading once every member has been seen
    tar = subprocess.run(
        ["tar", "-xzf", "-", "-C", str(scratch), "--occurrence=1", *members],
        stdin=curl.stdout,
        check=False,
    )
    curl.stdout.close()
    curl.terminate()
    curl.wait()
    if tar.returncode != 0:
        raise RuntimeError(f"tar exited with {tar.returncode}")
    for posture, item in spec.clips.items():
        source = scratch / item.member
        if sha256(source) != item.sha256:
            raise RuntimeError(f"{source.name}: sha256 mismatch")
        source.replace(wanted[posture])
    shutil.rmtree(scratch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="elastra-download-assets",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--assets",
        default=str(repo_path(OmegaConf.load(CONF_DIR / "paths" / "default.yaml").assets)),
        help="target directory (default: paths.assets of the configuration)",
    )
    parser.add_argument(
        "--skip-bones",
        action="store_true",
        help="do not fetch the BONES-SEED clips (ProtoMotions cannot run)",
    )
    args = parser.parse_args(argv)
    assets = Path(args.assets).resolve()
    manifest = OmegaConf.load(CONF_DIR / "assets" / "default.yaml")

    print("ProtoMotions G1 model and tracker", flush=True)
    download_repository_files(manifest.protomotions, assets / "protomotions", meshes_lfs=True)
    print("HoST G1 model", flush=True)
    download_repository_files(manifest.host, assets / "host", meshes_lfs=False)
    print("HoST policies", flush=True)
    download_host_policies(manifest.host_policies, assets / "host" / "policies")
    if args.skip_bones:
        print("skipping the BONES-SEED clips", flush=True)
        return 0
    print("BONES-SEED get-up clips", flush=True)
    download_bones_clips(manifest.bones_seed, assets / "bones_seed")

    from elastra.protomotions import build_reference_clips

    print("reference clips", flush=True)
    for path in build_reference_clips(assets):
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
