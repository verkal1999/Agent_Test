# PyLC1_Converter.py

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional


def parse_pou_blocks(
    xml_path: str,
    output_path: str = "generated_code_0.py",
    target_pou_name: Optional[str] = None,
) -> None:
    """
    Liest eine TwinCAT PLCopen-XML-Datei ein und erzeugt eine Python-Hilfsdatei
    mit genau EINEM POU und den zugehörigen Funktionsbaustein-Blöcken (B1, B2, ...).

    - target_pou_name = None:
        Es wird der letzte <pou>-Knoten aus der XML verwendet (abwärtskompatibel).
    - target_pou_name = "NameDesPOU":
        Es wird gezielt dieser POU ausgewertet.

    Die Hilfsdatei (generated_code_0.py) enthält:
        POU = {
            'pou_name': ...,
            'pou_type': ...,
            'input_vars': [... "Name:Typ" ...],
            'output_vars': [...],
            'local_vars': [...],
            'input_ids':  [{'Expression': ..., 'InVariable': ...}, ...],
            'output_ids': [{'Expression': ..., 'OutVariable': ...}, ...]
        }

        B1, B2, ... = Dicts mit Blockinformationen (localId, typeName, inputVariables, ...)
                      inkl. optionalem 'stcode' für EXECUTE-Blöcke.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {
        "plcopen": "http://www.plcopen.org/xml/tc6_0200",
        "xhtml": "http://www.w3.org/1999/xhtml",
    }

    # alle POU-Knoten holen
    all_pous = root.findall(".//plcopen:pou", ns)
    if not all_pous:
        raise ValueError("Keine <pou>-Elemente in der XML-Datei gefunden.")

    # passenden POU auswählen
    selected_pou = None
    if target_pou_name is None:
        # wie bisher: letzten POU nehmen
        selected_pou = all_pous[-1]
    else:
        for pou in all_pous:
            if pou.get("name") == target_pou_name:
                selected_pou = pou
                break
        if selected_pou is None:
            raise ValueError(f"POU '{target_pou_name}' wurde in der XML-Datei nicht gefunden.")

    pou_name = selected_pou.get("name", "UnnamedPOU")
    pou_type = selected_pou.get("pouType", "")

    # Hilfsfunktion zum Sammeln von Variablen-Listen (Name:Typ)
    def collect_vars(xpath: str) -> List[str]:
        result: List[str] = []
        for var in selected_pou.findall(xpath, ns):
            var_name = var.get("name")
            data_type = "UNKNOWN"
            type_elem = var.find("plcopen:type", ns)
            if type_elem is not None:
                # z. B. <plcopen:type><plcopen:BOOL/></plcopen:type>
                for child in type_elem:
                    data_type = child.tag.split("}", 1)[-1]
                    break
            if var_name:
                result.append(f"{var_name}:{data_type}")
        return result

    input_vars = collect_vars(".//plcopen:inputVars/plcopen:variable")
    output_vars = collect_vars(".//plcopen:outputVars/plcopen:variable")
    local_vars = collect_vars(".//plcopen:localVars/plcopen:variable")

    # Input-IDs (Mapping localId -> Ausdruck)
    input_ids: List[Dict[str, Optional[str]]] = []
    for in_var in selected_pou.findall(".//plcopen:inVariable", ns):
        local_id = in_var.get("localId")
        expr_elem = in_var.find("plcopen:expression", ns)
        expr_text = expr_elem.text.strip() if expr_elem is not None and expr_elem.text else None
        if local_id is not None:
            input_ids.append(
                {
                    "Expression": expr_text,
                    "InVariable": local_id,
                }
            )

    # Output-IDs (Mapping localId -> Ausdruck + Quelle)
    output_ids: List[Dict[str, Optional[str]]] = []
    for out_var in selected_pou.findall(".//plcopen:outVariable", ns):
        local_id = out_var.get("localId")

        # Expression wie bisher
        expr_elem = out_var.find("plcopen:expression", ns)
        expr_text = expr_elem.text.strip() if expr_elem is not None and expr_elem.text else None

        # NEU: Quelle (refLocalId aus connectionPointIn)
        source_local_id = None
        cp_in = out_var.find("plcopen:connectionPointIn", ns)
        if cp_in is not None:
            conn = cp_in.find("plcopen:connection", ns)
            if conn is not None:
                source_local_id = conn.get("refLocalId")

        if local_id is not None:
            output_ids.append(
                {
                    "Expression": expr_text,
                    "OutVariable": local_id,
                    "SourceLocalId": source_local_id,   # <--- neu
                }
            )

    # Blöcke dieses POU
    blocks: List[Dict[str, object]] = []
    for idx, block in enumerate(selected_pou.findall(".//plcopen:block", ns), start=1):
        block_local_id = block.get("localId")
        type_name = block.get("typeName")

        # optionale Positionsinformation
        pos_elem = block.find("plcopen:position", ns)
        position = dict(pos_elem.attrib) if pos_elem is not None else {}

        # Eingangsverbindungen (refLocalId)
        input_variables: List[str] = []
        for var in block.findall(".//plcopen:inputVariables/plcopen:variable", ns):
            conn = var.find(".//plcopen:connection", ns)
            if conn is not None and "refLocalId" in conn.attrib:
                input_variables.append(conn.get("refLocalId"))  # type: ignore[arg-type]

        # formale Parameter (optional)
        variable_params: List[str] = []
        for var in block.findall(".//plcopen:variable", ns):
            formal = var.get("formalParameter")
            if formal:
                variable_params.append(formal)

        # Verbindungspunkte (In)
        conn_points_in: List[str] = []
        conn_ref_local_ids: List[Optional[str]] = []
        for cp in block.findall(".//plcopen:connectionPointIn", ns):
            conn_points_in.append(cp.tag.split("}", 1)[-1])
            conn = cp.find("plcopen:connection", ns)
            ref = conn.get("refLocalId") if conn is not None else None
            conn_ref_local_ids.append(ref)

        # EXECUTE: ST-Code auslesen (optional)
        stcode = None
        if type_name == "EXECUTE":
            data = block.find(
                ".//plcopen:addData/plcopen:data[@name='http://www.3s-software.com/plcopenxml/stcode']",
                ns,
            )
            if data is not None:
                st_elem = data.find(".//STCode")
                if st_elem is not None and st_elem.text:
                    # Zeilen sauber übernehmen
                    lines = [line.rstrip("\r\n") for line in st_elem.text.splitlines()]
                    stcode = "\n".join(lines)

        blocks.append(
            {
                "name": f"B{idx}",
                "block_localId": block_local_id,
                "typeName": type_name,
                "pou_name": pou_name,
                "block_position": position,
                "inputVariables": input_variables,
                "variable": variable_params,
                "connectionpointIn": conn_points_in,
                "connection_refLocalId": conn_ref_local_ids,
                "stcode": stcode,
            }
        )

    # Python-Hilfsmodul erzeugen
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated intermediate representation from PLCopen XML\n\n")
        f.write("POU = {\n")
        f.write(f"    'pou_name': '{pou_name}',\n")
        f.write(f"    'pou_type': '{pou_type}',\n")
        f.write(f"    'input_vars': {input_vars},\n")
        f.write(f"    'output_vars': {output_vars},\n")
        f.write(f"    'local_vars': {local_vars},\n")
        f.write(f"    'input_ids': {input_ids},\n")
        f.write(f"    'output_ids': {output_ids},\n")
        f.write("}\n\n")

        for block in blocks:
            # repr(block) → gültiges Python-Dict inkl. None
            f.write(f"{block['name']} = {repr(block)}\n")

        f.write("\n")


if __name__ == "__main__":
    # Beispiel:
    # python PyLC1_Converter.py export.xml VSG_AS_CompressorControl
    import sys

    xml = sys.argv[1] if len(sys.argv) > 1 else "export.xml"
    target = sys.argv[2] if len(sys.argv) > 2 else None
    parse_pou_blocks(xml_path=xml, output_path="generated_code_0.py", target_pou_name=target)
