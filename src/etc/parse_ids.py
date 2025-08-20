import gzip
import pandas as pd
import requests
from pathlib import Path


# Step 1. Download MetaNetX tables (once)
chem_prop_url = "https://www.metanetx.org/cgi-bin/mnxget/mnxref/chem_prop.tsv.gz"
chem_xref_url = "https://www.metanetx.org/cgi-bin/mnxget/mnxref/chem_xref.tsv.gz"


data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "resources"
data_dir.mkdir(parents=True, exist_ok=True)

for url in [chem_prop_url, chem_xref_url]:
    fname = url.split("/")[-1]
    out_path = data_dir / fname
    r = requests.get(url, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)