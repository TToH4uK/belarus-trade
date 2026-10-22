"""Загрузка и очистка данных Белстата (экспорт/импорт важнейших видов продукции)."""
import re
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _num(x):
    """'1 196,8' -> 1196.8 ; обычные числа оставляет как есть."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("\xa0", "").replace(" ", "").replace(",", ".")
    return float(s)


def _split_name(label: str):
    """'Говядина, тыс. т' -> ('Говядина', 'тыс. т')"""
    label = re.sub(r"\s+", " ", label).strip()
    name, _, unit = label.rpartition(", ")
    return name.strip(), unit.strip()


def load_one(path: Path, flow: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    hdr = raw.index[raw.iloc[:, 1:].apply(lambda r: r.astype(str).str.fullmatch(r"2000(\.0)?").any(), axis=1)][0]
    years = [int(float(y)) for y in raw.iloc[hdr, 1:].dropna()]
    body = raw.iloc[hdr + 1:, : len(years) + 1].dropna(subset=[0])
    rows = []
    for _, r in body.iterrows():
        name, unit = _split_name(str(r[0]))
        for y, v in zip(years, r[1:]):
            if pd.notna(v):
                rows.append((flow, name, unit, y, _num(v)))
    return pd.DataFrame(rows, columns=["flow", "product", "unit", "year", "value"])


def load_all() -> pd.DataFrame:
    exp = load_one(DATA / "export_tovarov_vazh-2021.xlsx", "Экспорт")
    imp = load_one(DATA / "import_tovarov_vazh-2021.xlsx", "Импорт")
    return pd.concat([exp, imp], ignore_index=True)

