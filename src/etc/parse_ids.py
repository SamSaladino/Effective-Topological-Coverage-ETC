import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import ast

class XMLParser:
    """Parse HUMAN1 SBML and extract identifiers present in the file.

    This parser only reads local XML and extracts identifiers present in the
    SBML file; it does not perform any web requests.
    """

    def __init__(self, path: str):
        self.path = path
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        self.ns = {
            "": "http://www.sbml.org/sbml/level3/version1/core",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }
        self.data = []
        self.df = pd.DataFrame()

        # Collect identifiers.org resources for each species
        core_ns = "http://www.sbml.org/sbml/level3/version1/core"
        for species in self.root.findall(f".//{{{core_ns}}}species"):
            metabolite_id = species.get("id")
            identifiers = [
                li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
                for li in species.findall(".//rdf:RDF//rdf:li", self.ns)
            ]
            self.data.append([metabolite_id, identifiers])

    def extract_data(self) -> pd.DataFrame:
        """Build base DataFrame with cleaned/split identifiers."""
        self.df = pd.DataFrame(self.data, columns=["Metabolite_ID", "Identifiers"])
        self.df.Identifiers = self.df.Identifiers.astype(str)
        self.df.Identifiers = self.df.Identifiers.str.replace(
            "https://identifiers.org/", "", regex=False
        )
        self.df["Identifiers"] = self.df["Identifiers"].apply(ast.literal_eval)
        self.df.Identifiers = self.df.Identifiers.apply(self.split_and_clean)
        return self.df

    def split_and_clean(self, identifiers):
        """Split identifiers.org paths on '/' -> [['namespace','value'], ...]."""
        return [item.split("/") for item in identifiers]

    def _first_of(self, values, default=None):
        """Return the first non-empty value from a list of candidates.
        
        Skips None, empty strings, empty lists, and empty dicts.
        """
        for value in values:
            if value not in (None, "", [], {}):
                return value
        return default

    def _normalize_chebi(self, chebi_raw):
        """Normalize CHEBI identifier from [namespace, value] pair.
        
        Extracts the value part from split identifier pair.
        """
        if isinstance(chebi_raw, list) and len(chebi_raw) > 1:
            return chebi_raw[1]  # Return the value part (e.g., 'CHEBI:12345')
        return chebi_raw

    def to_identifier_df(self) -> pd.DataFrame:
        """Return DataFrame indexed by HUMAN1_ID with common identifier columns.

        Columns include chebi, smiles, inchikey, kegg, metanetx and vmhmetabolite
        when available in the SBML file.
        """
        if self.df.empty:
            self.extract_data()

        rows = []
        for metabolite_id, identifiers in zip(self.df.Metabolite_ID, self.df.Identifiers):
            chebi_raw = self._first_of(identifiers, {"chebi"})
            chebi = self._normalize_chebi(chebi_raw)

            smiles = self._first_of(identifiers, {"smiles", "smile"})
            inchikey = self._first_of(
                identifiers, {"inchikey", "inchi.key", "inchi_key"}
            )
            kegg = self._first_of(identifiers, {"kegg.compound"})
            metanetx = self._first_of(identifiers, {"metanetx.chemical"})
            vmhmetabolite = self._first_of(identifiers, {"vmhmetabolite"})
            hmdb = self._first_of(identifiers, {"hmdb"})
            lipidmaps = self._first_of(identifiers, {"lipidmaps"})
            pubchem = self._first_of(identifiers, {"pubchem.compound"})

            chebi_val = chebi if isinstance(chebi, str) and chebi else np.nan
            smiles_val = smiles if isinstance(smiles, str) and smiles else np.nan
            inchikey_val = (
                inchikey if isinstance(inchikey, str) and inchikey else np.nan
            )
            kegg_val = kegg if isinstance(kegg, str) and kegg else np.nan
            metanetx_val = (
                metanetx if isinstance(metanetx, str) and metanetx else np.nan
            )
            vmh_val = (
                vmhmetabolite
                if isinstance(vmhmetabolite, str) and vmhmetabolite
                else np.nan
            )
            hmdb_val = hmdb if isinstance(hmdb, str) and hmdb else np.nan
            lipidmaps_val = (
                lipidmaps if isinstance(lipidmaps, str) and lipidmaps else np.nan
            )
            pubchem_val = (
                pubchem if isinstance(pubchem, str) and pubchem else np.nan
            )

            rows.append(
                {
                    "HUMAN1_ID": metabolite_id,
                    "chebi": chebi_val,
                    "smiles": smiles_val,
                    "inchikey": inchikey_val,
                    "kegg": kegg_val,
                    "metanetx": metanetx_val,
                    "vmhmetabolite": vmh_val,
                    "hmdb": hmdb_val,
                    "lipidmaps": lipidmaps_val,
                    "pubchem": pubchem_val,
                }
            )

        return pd.DataFrame(rows).set_index("HUMAN1_ID")