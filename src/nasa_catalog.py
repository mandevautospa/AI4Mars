"""
src/nasa_catalog.py
===================
Functions for discovering the AI4Mars dataset via:
  - NASA's open data catalog (CKAN REST API)
  - Zenodo REST API (where the actual files are hosted)

These functions are pure data-fetching utilities.
They do NOT print anything by themselves so that they can be reused
in both notebooks (where you want pretty output) and scripts (where you
might want to process results silently).

Use the helper functions at the bottom when you want a human-readable summary.
"""

import requests


# ---------------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------------

NASA_CKAN_BASE: str = "https://data.nasa.gov/api/3/action"
ZENODO_API_BASE: str = "https://zenodo.org/api/records"


# ---------------------------------------------------------------------------
# NASA CKAN functions
# ---------------------------------------------------------------------------

def search_nasa_datasets(query: str, rows: int = 10) -> dict:
    """Search the NASA open data catalog (CKAN) for datasets matching *query*.

    Parameters
    ----------
    query : str
        Free-text search string, e.g. ``"AI4Mars Mars terrain"``.
    rows : int
        Maximum number of results to return (default 10).

    Returns
    -------
    dict
        Parsed JSON response from the CKAN ``package_search`` endpoint.
        The relevant results are in ``response["result"]["results"]``.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status code.
    """
    url = f"{NASA_CKAN_BASE}/package_search"
    params = {"q": query, "rows": rows}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_nasa_dataset(dataset_id: str) -> dict:
    """Fetch full metadata for a single NASA CKAN dataset by its ID or name.

    Parameters
    ----------
    dataset_id : str
        The CKAN dataset name or UUID, e.g. ``"ai4mars-mars-terrain"``.

    Returns
    -------
    dict
        Parsed JSON response from the CKAN ``package_show`` endpoint.

    Raises
    ------
    requests.HTTPError
        If the dataset is not found or the request fails.
    """
    url = f"{NASA_CKAN_BASE}/package_show"
    params = {"id": dataset_id}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Zenodo functions
# ---------------------------------------------------------------------------

def get_zenodo_record(record_id: int) -> dict:
    """Fetch metadata for a single Zenodo record.

    The AI4Mars terrain-segmentation record ID is **15995036**.
    See: https://zenodo.org/records/15995036

    Parameters
    ----------
    record_id : int
        Zenodo record identifier.

    Returns
    -------
    dict
        Full Zenodo record metadata including title, description, and file list.

    Raises
    ------
    requests.HTTPError
        If the record is not found or the request fails.
    """
    url = f"{ZENODO_API_BASE}/{record_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def list_zenodo_files(record: dict) -> list:
    """Extract the file list from a Zenodo record dictionary.

    Parameters
    ----------
    record : dict
        A record dictionary returned by :func:`get_zenodo_record`.

    Returns
    -------
    list[dict]
        Each element has keys: ``key`` (filename), ``size`` (bytes),
        ``links.self`` (direct download URL).
    """
    return record.get("files", [])


# ---------------------------------------------------------------------------
# Human-readable summary helpers (safe to call from notebooks)
# ---------------------------------------------------------------------------

def print_nasa_search_results(search_response: dict) -> None:
    """Print a short summary of NASA CKAN search results.

    Parameters
    ----------
    search_response : dict
        The dict returned by :func:`search_nasa_datasets`.
    """
    results = search_response.get("result", {}).get("results", [])
    print(f"Found {len(results)} dataset(s):\n")
    for i, ds in enumerate(results):
        title = ds.get("title", "N/A")
        name = ds.get("name", "N/A")
        print(f"  [{i}] title : {title}")
        print(f"       name  : {name}\n")


def print_zenodo_files(files: list) -> None:
    """Print a table of Zenodo file names, sizes, and download links.

    Parameters
    ----------
    files : list[dict]
        The list returned by :func:`list_zenodo_files`.
    """
    print(f"{'Filename':<50} {'Size (MB)':>10}  Download URL")
    print("-" * 120)
    for f in files:
        name = f.get("key", "?")
        size_mb = f.get("size", 0) / 1_000_000
        url = f.get("links", {}).get("self", "N/A")
        print(f"{name:<50} {size_mb:>10.1f}  {url}")
