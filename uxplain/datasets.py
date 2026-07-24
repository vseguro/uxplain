"""
Real-world labour-force datasets used in the paper's illustrations.

Two national household surveys, loaded through a small \\pkg{scikit-learn}-style
interface: each loader returns a :class:`sklearn.utils.Bunch` (with ``data``,
``target``, ``frame``, ``feature_names`` and ``DESCR``), or the ready-to-model
``(X, y)`` pair when called with ``return_X_y=True``::

    from uxplain.datasets import fetch_pnadc, load_geih

    X, y = fetch_pnadc(return_X_y=True)                 # Brazil, regression
    X, y = load_geih("data/geih", return_X_y=True)      # Colombia, classification

- **PNAD Contínua** (IBGE, Brazil) — conformal *regression* on monthly labour
  income. :func:`fetch_pnadc` downloads the quarter from the IBGE public server
  and caches it, so nothing is shipped with the package.
- **GEIH** (DANE, Colombia) — conformal *classification* of occupational group.
  :func:`load_geih` reads a copy the user has downloaded from DANE; see the note
  on data terms below.

Data terms and provenance
-------------------------
No survey data is bundled with or redistributed by this package; the loaders
either fetch from the producer's own server or read a local copy the user
obtained themselves. When you use these datasets you must cite the producer:

- IBGE microdata are public and may be reproduced with attribution
  (see :data:`PNADC_CITATION`).
- DANE authorises use, transformation and analysis of its anonymised microdata
  **provided the source is cited** (:data:`GEIH_CITATION`); it does not
  authorise redistribution or commercial use, which is exactly why
  :func:`load_geih` never downloads or ships the data. Obtain a month from
  https://microdatos.dane.gov.co and point the loader at it.

Labour income is a deliberately good showcase for the regression side: its
spread grows with education and with informality, so the *width* of a conformal
interval is driven by different features than its centre — exactly the
distinction the package is built to expose.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import re
import urllib.request
import zipfile

import pandas as pd
from sklearn.utils import Bunch

__all__ = [
    "PNADC_VARIABLES",
    "GEIH_VARIABLES",
    "PNADC_FEATURES",
    "GEIH_FEATURES",
    "PNADC_CITATION",
    "GEIH_CITATION",
    "default_cache_dir",
    "fetch_pnadc",
    "load_geih",
]

_IBGE_ROOT = (
    "https://ftp.ibge.gov.br/Trabalho_e_Rendimento/"
    "Pesquisa_Nacional_por_Amostra_de_Domicilios_continua/Trimestral/Microdados"
)
_IBGE_DICT = f"{_IBGE_ROOT}/Documentacao/Dicionario_e_input_20221031.zip"

PNADC_CITATION = (
    "Instituto Brasileiro de Geografia e Estatistica (IBGE), Pesquisa Nacional "
    "por Amostra de Domicilios Continua (PNAD Continua), microdata. "
    "Available from https://www.ibge.gov.br/."
)
GEIH_CITATION = (
    "Fuente: Departamento Administrativo Nacional de Estadistica (DANE), "
    "Gran Encuesta Integrada de Hogares (GEIH), bases anonimizadas: "
    "www.dane.gov.co."
)

#: PNAD Contínua source columns kept by :func:`fetch_pnadc`, mapped to readable
#: names.
PNADC_VARIABLES: dict[str, str] = {
    "UF": "state",
    "V1022": "urban",             # urban / rural
    "V2007": "sex",
    "V2009": "age",
    "V2010": "race",
    "VD3004": "education",        # highest level attained
    "VD4001": "in_labour_force",
    "VD4002": "employed",
    "VD4009": "job_position",     # employee w/ or w/o formal contract, etc.
    "VD4012": "pension_contrib",  # contributes to social security
    "VD4020": "income",           # monthly labour income (target)
    "VD4035": "hours",            # hours actually worked
    "V1028": "weight",            # calibrated survey weight
}

#: GEIH source columns expected by :func:`load_geih`, mapped to the same schema.
#:
#: These are the names used since the 2022 questionnaire redesign, which renamed
#: several fields (sex moved from ``P6020`` to ``P3271``, education from
#: ``P6210`` to ``P3042``, and the occupation code became ``OFICIO_C8``). Pass an
#: explicit mapping to :func:`load_geih` for earlier years.
GEIH_VARIABLES: dict[str, str] = {
    "P3271": "sex",
    "P6040": "age",
    "P3042": "education",
    "P6800": "hours",
    "INGLABO": "income",          # monthly labour income
    "OFICIO_C8": "occupation",    # 4-digit occupation code
    "RAMA2D_R4": "industry",      # 2-digit industry code
    "P6920": "pension_contrib",   # proxy for formal employment
    "FEX_C18": "weight",          # survey expansion factor
}

#: Default modelling features for each survey (the ones used in the paper).
PNADC_FEATURES: list[str] = [
    "age", "hours", "education", "sex", "race", "urban", "informal",
]
GEIH_FEATURES: list[str] = [
    "age", "hours", "education", "sex", "income", "informal",
]

#: Person key that links GEIH modules to each other.
GEIH_MERGE_KEYS: tuple[str, ...] = ("DIRECTORIO", "SECUENCIA_P", "ORDEN")

#: Major occupational groups: the leading digit of the ISCO-08 (CIUO-08 A.C.)
#: code. Collapsing the 4-digit occupation to its major group yields a target
#: with a workable number of classes for conformal classification.
CIUO_MAJOR_GROUPS: dict[str, str] = {
    "0": "Armed forces",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians",
    "4": "Clerical support",
    "5": "Services and sales",
    "6": "Skilled agricultural",
    "7": "Craft and trades",
    "8": "Plant and machine operators",
    "9": "Elementary occupations",
}

# Both surveys code social-security contribution as 1 = contributes. Defining
# informality this way keeps the two labels comparable.
_CONTRIBUTES_CODE = "1"


def default_cache_dir() -> Path:
    """Directory used to cache downloaded microdata (override with
    ``UXPLAIN_DATA_HOME``)."""
    root = os.environ.get("UXPLAIN_DATA_HOME")
    return Path(root) if root else Path.home() / ".cache" / "uxplain"


# ---------------------------------------------------------------------------
# Shared Bunch / (X, y) packaging
# ---------------------------------------------------------------------------

def _finalize(frame, features, target, *, return_X_y, subsample, random_state,
              descr, citation):
    """Coerce the modelling columns, drop unusable rows, and package the result.

    Returns ``(X, y)`` when ``return_X_y`` is set, otherwise a
    :class:`sklearn.utils.Bunch`.
    """
    features = list(features)
    missing = [c for c in features + [target] if c not in frame.columns]
    if missing:
        raise KeyError(f"Requested columns not in the data: {missing}")

    # Features are coerced to float (not int): partial dependence and several
    # explainers reject integer columns, and float avoids implicit rounding.
    work = frame.copy()
    for column in features:
        if work[column].dtype == bool:
            work[column] = work[column].astype(float)
        else:
            work[column] = pd.to_numeric(work[column], errors="coerce").astype(float)

    subset = list(dict.fromkeys(features + [target]))  # dedupe if target ∈ feats
    work = work.dropna(subset=subset)
    if subsample is not None and subsample < len(work):
        work = work.sample(n=subsample, random_state=random_state)
    work = work.reset_index(drop=True)

    X = work[features]
    y = work[target].to_numpy()
    if return_X_y:
        return X, y
    return Bunch(
        data=X, target=y, frame=work, feature_names=features,
        target_name=target, DESCR=descr, citation=citation,
    )


# ---------------------------------------------------------------------------
# PNAD Contínua (IBGE, Brazil)
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(dest)
    return dest


def _parse_sas_layout(text: str) -> dict[str, tuple[int, int]]:
    """Read the IBGE SAS input file into ``{name: (start, width)}``."""
    layout: dict[str, tuple[int, int]] = {}
    pattern = re.compile(r"^@(\d+)\s+(\w+)\s+\$?(\d+)\.")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            start, name, width = match.groups()
            layout[name] = (int(start), int(width))
    return layout


def _pnadc_layout(cache_dir: Path) -> dict[str, tuple[int, int]]:
    archive = _download(_IBGE_DICT, cache_dir / "pnadc_dictionary.zip")
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".sas"))
        return _parse_sas_layout(zf.read(name).decode("latin-1"))


def _quarter_url(year: int, quarter: int) -> str:
    listing_url = f"{_IBGE_ROOT}/{year}/"
    with urllib.request.urlopen(listing_url) as response:
        listing = response.read().decode("utf-8", errors="replace")
    prefix = f"PNADC_{quarter:02d}{year}"
    names = re.findall(rf'href="({re.escape(prefix)}\w*\.zip)"', listing)
    if not names:
        raise FileNotFoundError(
            f"No PNAD Continua archive for {year} Q{quarter} at {listing_url}"
        )
    return listing_url + sorted(names)[-1]


def _pnadc_frame(year, quarter, cache_dir, employed_only, min_income):
    """Download and parse one quarter of PNAD Contínua into a tidy frame."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    layout = _pnadc_layout(cache_dir)
    absent = [v for v in PNADC_VARIABLES if v not in layout]
    if absent:
        raise KeyError(f"Variables absent from the IBGE layout: {absent}")

    # Resolve the remote name only when the archive is not already cached, so a
    # second run needs no network at all.
    archive = cache_dir / f"pnadc_{year}q{quarter}.zip"
    if not (archive.exists() and archive.stat().st_size > 0):
        archive = _download(_quarter_url(year, quarter), archive)

    wanted = sorted(PNADC_VARIABLES, key=lambda v: layout[v][0])
    colspecs = [(layout[v][0] - 1, layout[v][0] - 1 + layout[v][1])
                for v in wanted]
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        with zf.open(name) as handle:
            frame = pd.read_fwf(
                io.TextIOWrapper(handle, encoding="latin-1"),
                colspecs=colspecs, names=wanted, dtype=str,
            )

    frame = frame.rename(columns=PNADC_VARIABLES)
    for column in ("age", "income", "hours", "weight"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["informal"] = frame["pension_contrib"].str.strip() != _CONTRIBUTES_CODE

    if employed_only:
        frame = frame[(frame["employed"] == "1")
                      & (frame["income"] >= min_income)]
    return frame.reset_index(drop=True)


def fetch_pnadc(
    year: int = 2024,
    quarter: int = 1,
    *,
    features: list[str] | None = None,
    target: str = "income",
    return_X_y: bool = False,
    subsample: int | None = None,
    random_state: int | None = None,
    employed_only: bool = True,
    min_income: float = 1.0,
    cache_dir: str | Path | None = None,
):
    """
    Load one quarter of PNAD Contínua (IBGE, Brazil).

    The ~200 MB archive is downloaded from the IBGE public server on first use
    and cached, so later calls need no network. Nothing is shipped with the
    package. When you use this data, cite the source (:data:`PNADC_CITATION`).

    Parameters
    ----------
    year, quarter
        Reference quarter, e.g. ``2024, 1``.
    features
        Modelling columns. Defaults to :data:`PNADC_FEATURES`.
    target
        Response column. ``"income"`` (regression, default). Other useful
        choices: ``"informal"`` (classification).
    return_X_y
        If ``True``, return ``(X, y)`` instead of a
        :class:`~sklearn.utils.Bunch`.
    subsample
        If given, randomly keep this many rows (useful to keep an illustration
        fast; the conformal guarantee does not depend on the sample size).
    random_state
        Seed for ``subsample``.
    employed_only, min_income
        Restrict to employed respondents with a labour income at least
        ``min_income`` (zeros mean "no income", not low income).
    cache_dir
        Download cache. Defaults to :func:`default_cache_dir`.

    Returns
    -------
    sklearn.utils.Bunch or tuple
        A ``Bunch`` with ``data``, ``target``, ``frame`` (the full parsed
        table), ``feature_names``, ``target_name``, ``DESCR`` and
        ``citation``; or ``(X, y)`` when ``return_X_y=True``.
    """
    frame = _pnadc_frame(year, quarter, cache_dir, employed_only, min_income)
    descr = (f"PNAD Continua {year} Q{quarter} (IBGE). "
             f"Target: {target}. {PNADC_CITATION}")
    return _finalize(
        frame, features or PNADC_FEATURES, target,
        return_X_y=return_X_y, subsample=subsample, random_state=random_state,
        descr=descr, citation=PNADC_CITATION,
    )


# ---------------------------------------------------------------------------
# GEIH (DANE, Colombia)
# ---------------------------------------------------------------------------

def _is_wanted_geih_module(name: str) -> bool:
    """True for the two person-level GEIH modules :func:`load_geih` needs.

    ``"No ocupados"`` is disjoint from ``"Ocupados"`` (merging either in would
    empty the result), so the match on the employed module is exact.
    """
    stem = Path(name).stem.strip().lower()
    return stem == "ocupados" or stem.startswith("caracter")


def _read_geih_module(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str,
                       encoding="latin-1")


def _geih_frame(path, variables, min_income):
    """Merge the GEIH modules under ``path`` into one tidy person-level frame."""
    variables = variables or GEIH_VARIABLES

    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.is_dir():
            found = sorted(set(path.rglob("*.csv")) | set(path.rglob("*.CSV")))
            if not found:
                raise FileNotFoundError(f"No CSV files found under {path}")
            paths = [p for p in found if _is_wanted_geih_module(p.name)]
            if not paths:
                raise FileNotFoundError(
                    "Neither the 'Ocupados' nor the 'Caracteristicas generales' "
                    f"module was found under {path}. Files present: "
                    f"{[p.name for p in found]}"
                )
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in path]

    frame = None
    for module in (_read_geih_module(p) for p in paths):
        keep = [c for c in module.columns
                if c in variables or c in GEIH_MERGE_KEYS]
        module = module[keep]
        if frame is None:
            frame = module
            continue
        shared = [k for k in GEIH_MERGE_KEYS
                  if k in frame.columns and k in module.columns]
        if not shared:
            raise KeyError(
                f"Cannot merge GEIH modules: none of {GEIH_MERGE_KEYS} are "
                "present in both files. Pass the modules with the person key."
            )
        new_cols = [c for c in module.columns
                    if c not in frame.columns or c in shared]
        frame = frame.merge(module[new_cols], on=shared, how="inner")

    present = {src: dst for src, dst in variables.items()
               if src in frame.columns}
    if "income" not in present.values():
        raise KeyError(
            f"No income column found. Columns present: "
            f"{sorted(frame.columns)[:25]}. Pass an explicit `variables` mapping."
        )
    frame = frame[list(present)].rename(columns=present)

    for column in ("age", "income", "hours", "weight"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].str.replace(",", ".", regex=False), errors="coerce",
            )
    if "pension_contrib" in frame.columns:
        frame["informal"] = (frame["pension_contrib"].astype(str).str.strip()
                             != _CONTRIBUTES_CODE)
    if "occupation" in frame.columns:
        leading = frame["occupation"].astype(str).str.strip().str.zfill(4).str[0]
        frame["occupation_group"] = leading.map(CIUO_MAJOR_GROUPS)

    frame = frame[frame["income"].fillna(0) >= min_income]
    return frame.reset_index(drop=True)


def load_geih(
    path: str | Path | list[str | Path],
    *,
    features: list[str] | None = None,
    target: str = "occupation_group",
    return_X_y: bool = False,
    subsample: int | None = None,
    random_state: int | None = None,
    variables: dict[str, str] | None = None,
    min_income: float = 1.0,
):
    """
    Load a locally downloaded GEIH extract (DANE, Colombia).

    This function never downloads anything: DANE authorises using and analysing
    its anonymised microdata with attribution but not redistributing it, so you
    obtain a month yourself and point ``path`` at it. Cite the source when you
    use the data (:data:`GEIH_CITATION`).

    To get the data: open https://microdatos.dane.gov.co, choose *Gran Encuesta
    Integrada de Hogares (GEIH)* for a month, download and unzip it, and pass the
    folder that contains the CSV modules. A month ships one file per
    questionnaire module; this loader merges the *Ocupados* module (income,
    hours, occupation, pension contribution) with *Características generales,
    seguridad social en salud y educación* (sex, age, education) on the person
    key ``DIRECTORIO``/``SECUENCIA_P``/``ORDEN``.

    Parameters
    ----------
    path
        A directory holding the module CSVs (scanned recursively; only the two
        person-level modules are merged), a list of CSV paths, or a single CSV.
    features
        Modelling columns. Defaults to :data:`GEIH_FEATURES`.
    target
        Response column. ``"occupation_group"`` (classification, default); the
        4-digit occupation collapsed to its ISCO-08 major group. Other choices:
        ``"income"`` (regression), ``"informal"`` (binary classification).
    return_X_y
        If ``True``, return ``(X, y)`` instead of a
        :class:`~sklearn.utils.Bunch`.
    subsample, random_state
        Optionally keep a random subset of ``subsample`` rows.
    variables
        Override the source-column mapping (DANE renames fields between years).
        Defaults to :data:`GEIH_VARIABLES`.
    min_income
        Drop incomes below this value.

    Returns
    -------
    sklearn.utils.Bunch or tuple
        Same shape as :func:`fetch_pnadc`.
    """
    frame = _geih_frame(path, variables, min_income)
    descr = (f"GEIH (DANE), local extract. Target: {target}. {GEIH_CITATION}")
    return _finalize(
        frame, features or GEIH_FEATURES, target,
        return_X_y=return_X_y, subsample=subsample, random_state=random_state,
        descr=descr, citation=GEIH_CITATION,
    )
