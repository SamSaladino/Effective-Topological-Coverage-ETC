import numpy as np
import pandas as pd
import pytest

from etc.parse_ids import XMLParser

@pytest.fixture
def toy_sbml_file(tmp_path):
    sbml_content = """<?xml version="1.0" encoding="UTF-8"?>
<sbml
    xmlns="http://www.sbml.org/sbml/level3/version1/core"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    level="3"
    version="1">

    <model id="toy_model">
        <listOfCompartments>
            <compartment id="c" constant="true"/>
        </listOfCompartments>

        <listOfSpecies>
            <species
                id="M_glucose_c"
                compartment="c"
                boundaryCondition="false"
                hasOnlySubstanceUnits="false"
                constant="false">

                <annotation>
                    <rdf:RDF>
                        <rdf:Description>
                            <rdf:Bag>
                                <rdf:li
                                    rdf:resource="https://identifiers.org/chebi/CHEBI:17634"/>
                                <rdf:li
                                    rdf:resource="https://identifiers.org/kegg.compound/C00031"/>
                                <rdf:li
                                    rdf:resource="https://identifiers.org/hmdb/HMDB0000122"/>
                                <rdf:li
                                    rdf:resource="https://identifiers.org/metanetx.chemical/MNXM41"/>
                            </rdf:Bag>
                        </rdf:Description>
                    </rdf:RDF>
                </annotation>
            </species>

            <species
                id="M_unknown_c"
                compartment="c"
                boundaryCondition="false"
                hasOnlySubstanceUnits="false"
                constant="false"/>
        </listOfSpecies>
    </model>
</sbml>
"""

    path = tmp_path / "toy_model.xml"
    path.write_text(sbml_content, encoding="utf-8")

    return path

def test_to_identifier_df_extracts_expected_identifiers(
    toy_sbml_file,
):
    parser = XMLParser(str(toy_sbml_file))

    identifier_df = parser.to_identifier_df()

    assert isinstance(identifier_df, pd.DataFrame)

    assert list(identifier_df.index) == [
        "M_glucose_c",
        "M_unknown_c",
    ]

    assert identifier_df.loc["M_glucose_c", "chebi"] == "17634"
    assert identifier_df.loc["M_glucose_c", "kegg"] == "C00031"
    assert identifier_df.loc["M_glucose_c", "hmdb"] == "HMDB0000122"

    assert (
        identifier_df.loc["M_glucose_c", "metanetx"]
        == "MNXM41"
    )

    assert pd.isna(
        identifier_df.loc["M_unknown_c", "chebi"]
    )

    assert pd.isna(
        identifier_df.loc["M_unknown_c", "kegg"]
    )

def test_get_chebi_numbers_uses_human1_ids_as_keys(
    toy_sbml_file,
):
    parser = XMLParser(str(toy_sbml_file))
    parser.extract_data()

    chebi_mapping = parser.get_chebi_numbers()

    assert chebi_mapping == {
        "M_glucose_c": "17634",
    }

def test_to_identifier_df_calls_extract_data_automatically(
    toy_sbml_file,
):
    parser = XMLParser(str(toy_sbml_file))

    assert parser.df.empty

    result = parser.to_identifier_df()

    assert not result.empty
    assert "M_glucose_c" in result.index

def test_xml_parser_rejects_malformed_xml(tmp_path):
    malformed_file = tmp_path / "malformed.xml"
    malformed_file.write_text(
        "<sbml><model>",
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        XMLParser(str(malformed_file))
        
import xml.etree.ElementTree as ET
def test_xml_parser_rejects_malformed_xml(tmp_path):
    malformed_file = tmp_path / "malformed.xml"
    malformed_file.write_text(
        "<sbml><model>",
        encoding="utf-8",
    )

    with pytest.raises(ET.ParseError):
        XMLParser(str(malformed_file))