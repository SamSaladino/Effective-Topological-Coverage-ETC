import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import ast
import libchebipy
import re
import requests
class XMLParser:
    """Parse HUMAN1 SBML and extract identifiers present in the file (no web calls).

    Output DataFrame (index: HUMAN1_ID) columns:
    - chebi (normalized to 'CHEBI:xxxx')
    - smiles        (create the column empty)
    - inchikey      (create the column empty)
    - kegg          (from 'kegg.compound')
    - metanetx      (from 'metanetx.chemical')
    - vmhmetabolite (from 'vmhmetabolite')
    """

    def __init__(self, path: str):
        self.path = path
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        self.ns = {
            '': 'http://www.sbml.org/sbml/level3/version1/core',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        }
        self.data = []
        self.df = pd.DataFrame()

        # Collect identifiers.org resources for each species
        for species in self.root.findall('.//{http://www.sbml.org/sbml/level3/version1/core}species'):
            metabolite_id = species.get('id')
            identifiers = [
                li.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                for li in species.findall('.//rdf:RDF//rdf:li', self.ns)
            ]
            self.data.append([metabolite_id, identifiers])

    def extract_data(self) -> pd.DataFrame:
        """Build base DataFrame with cleaned/split identifiers."""
        self.df = pd.DataFrame(self.data, columns=['Metabolite_ID', 'Identifiers'])
        self.df.Identifiers = self.df.Identifiers.astype(str)
        self.df.Identifiers = self.df.Identifiers.str.replace('https://identifiers.org/', '', regex=False)
        self.df['Identifiers'] = self.df['Identifiers'].apply(ast.literal_eval)
        self.df.Identifiers = self.df.Identifiers.apply(self.split_and_clean)
        return self.df

    def split_and_clean(self, identifiers):
        """Split identifiers.org paths on '/' -> [['namespace','value'], ...]."""
        return [item.split('/') for item in identifiers]

    def to_identifier_df(self) -> pd.DataFrame:
        """Return DataFrame indexed by HUMAN1_ID with chebi, smiles, inchikey, kegg, metanetx, vmhmetabolite (XML-only)."""
        if self.df.empty:
            self.extract_data()

        rows = []
        for metabolite_id, identifiers in zip(self.df.Metabolite_ID, self.df.Identifiers):
            chebi_raw = self._first_of(identifiers, {"chebi"})
            chebi = self._normalize_chebi(chebi_raw)

            smiles = self._first_of(identifiers, {"smiles", "smile"})
            inchikey = self._first_of(identifiers, {"inchikey", "inchi.key", "inchi_key"})
            kegg = self._first_of(identifiers, {"kegg.compound"})
            metanetx = self._first_of(identifiers, {"metanetx.chemical"})
            vmhmetabolite = self._first_of(identifiers, {"vmhmetabolite"})

            rows.append({
                "HUMAN1_ID": metabolite_id,
                "chebi": chebi if isinstance(chebi, str) and chebi else np.nan,
                "smiles": smiles if isinstance(smiles, str) and smiles else np.nan,
                "inchikey": inchikey if isinstance(inchikey, str) and inchikey else np.nan,
                "kegg": kegg if isinstance(kegg, str) and kegg else np.nan,
                "metanetx": metanetx if isinstance(metanetx, str) and metanetx else np.nan,
                "vmhmetabolite": vmhmetabolite if isinstance(vmhmetabolite, str) and vmhmetabolite else np.nan,
            })

        return pd.DataFrame(rows).set_index("HUMAN1_ID")

    # Helpers
    def _first_of(self, identifiers, names):
        """Return first identifier value whose namespace matches names (case-insensitive)."""
        names_lc = {n.lower() for n in names}
        for j in identifiers:
            if len(j) >= 2 and j[0].lower() in names_lc:
                return j[1]
        return np.nan

    def _normalize_chebi(self, value):
        """Normalize to 'CHEBI:xxxx' if present."""
        if not isinstance(value, str) or not value:
            return None
        token = value.split(':')[-1]
        return f"CHEBI:{token}"
    
