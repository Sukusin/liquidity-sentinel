from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import requests

from .sample_data import ensure_sample_data
from .utils import read_csv


@dataclass(frozen=True)
class SourceUrls:
    cbr_required_reserves_excel: str = "https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx"
    cbr_required_reserves_page: str = "https://www.cbr.ru/hd_base/RReserves/"
    cbr_ruonia: str = "https://www.cbr.ru/hd_base/ruonia/"
    cbr_repo: str = "https://www.cbr.ru/hd_base/repo/"
    cbr_keyrate: str = "https://www.cbr.ru/hd_base/keyrate/"
    cbr_bliquidity: str = "https://www.cbr.ru/hd_base/bliquidity/"
    minfin_ofz: str = "https://minfin.gov.ru/ru/document/"
    nalog_calendar: str = "https://www.nalog.gov.ru/rn77/calendar/"
    roskazna_eks_deposits: str = "https://roskazna.gov.ru/finansovye-operacii/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"


def load_raw_data(data_dir: Path, create_sample_if_missing: bool = True) -> dict[str, pd.DataFrame]:
    """Load raw CSV files.

    Expected files:
    - reserves.csv
    - repo.csv
    - ofz.csv
    - tax_calendar.csv
    - treasury.csv
    - ground_truth.csv
    """
    if create_sample_if_missing:
        ensure_sample_data(data_dir)
    frames = {
        "reserves": read_csv(data_dir / "reserves.csv"),
        "repo": read_csv(data_dir / "repo.csv"),
        "ofz": read_csv(data_dir / "ofz.csv"),
        "tax_calendar": read_csv(data_dir / "tax_calendar.csv"),
        "treasury": read_csv(data_dir / "treasury.csv"),
        "ground_truth": read_csv(data_dir / "ground_truth.csv"),
    }
    return frames


def download_file(url: str, destination: Path, timeout: int = 30) -> Path:
    """Best-effort downloader for official sources.

    The project uses CSV inputs internally because official pages often change
    layout. This helper is intentionally simple: download official Excel/HTML,
    inspect it, then convert to the expected CSV schema.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "ru-liquidity-sentinel/0.1"})
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def official_source_urls() -> SourceUrls:
    return SourceUrls()
