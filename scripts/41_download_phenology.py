"""Download DWD crop phenology observations (open data, CDC).

Two observer networks, both kept separately:
    annual_reporters     (Jahresmelder)  dense network, a year is published after the season
    immediate_reporters  (Sofortmelder)  smaller network, reported in near real time

For each network the newest historical file per crop plus the 'recent' file is
downloaded, together with station metadata and the phase / plant code tables.
Files already present with the same size as on the server are skipped.

Output: data/raw/phenology_dwd/
Usage:  python scripts/41_download_phenology.py
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/phenology_dwd"
CDC = "https://opendata.dwd.de/climate_environment/CDC"
BASE = f"{CDC}/observations_germany/phenology"
HELP = ["PH_Beschreibung_Phaenologie_Stationen_Jahresmelder.txt",
        "PH_Beschreibung_Phaenologie_Stationen_Sofortmelder.txt",
        "PH_Beschreibung_Phase.txt", "PH_Beschreibung_Pflanze.txt",
        "PH_Beschreibung_Phaenologie_Qualitaetsniveau.txt"]


def listing(url: str) -> list[str]:
    html = urllib.request.urlopen(url, timeout=60).read().decode("latin1")
    return re.findall(r'href="([^"]+\.txt)"', html)


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, method="HEAD")
    size = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", -1))
    if dest.exists() and dest.stat().st_size == size:
        return
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    if size >= 0 and tmp.stat().st_size != size:
        raise IOError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {size}")
    tmp.replace(dest)
    print(f"  {dest.name} ({size / 1e6:.1f} MB)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in HELP:
        fetch(f"{CDC}/help/{f}", OUT / f)
    for net in ("annual_reporters", "immediate_reporters"):
        for part in ("historical", "recent"):
            url = f"{BASE}/{net}/crops/{part}/"
            names = listing(url)
            if part == "historical":
                best: dict[str, tuple[int, str]] = {}
                for n in names:
                    m = re.match(r"PH_\w+?_Kulturpflanze_(.+?)(?:_(\d{4})_(\d{4}))?_hist\.txt$", n)
                    if m:
                        crop, end = m.group(1), int(m.group(3) or 9999)
                        if crop not in best or end > best[crop][0]:
                            best[crop] = (end, n)
                names = [v[1] for v in best.values()]
            else:
                names = [n for n in names if n.endswith("_akt.txt")]
            print(f"{net}/{part}: {len(names)} files")
            for n in names:
                fetch(url + n, OUT / n)


if __name__ == "__main__":
    main()
