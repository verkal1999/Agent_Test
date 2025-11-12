// NodeIdUtils.h – Hilfsfunktion zur Zerlegung von OPC-UA-NodeIds
//
// parseNsAndId(...):
//   - akzeptiert die Kurzform  "OPCUA.<...>" und setzt dann
//       nsOut    = 4
//       idTypeOut= 's'
//       idStrOut = voller String ("OPCUA.x")
//   - akzeptiert die Langform  "ns=<n>;<t>=<id>" (OPC UA NodeId-Notation):
//       Beispiel: "ns=4;s=OPCUA.DiagnoseFinished"
//       -> nsOut    = 4
//          idTypeOut= 's'
//          idStrOut = "OPCUA.DiagnoseFinished"
//   - Rückgabe: true bei erfolgreichem Parsen, false bei unpassendem Format.
//
// Verwendung in der MPA:
//   - InventorySnapshotUtils: um browsed NodeIds in (ns, type, id)-Tripel zu zerlegen.
//   - ReactionManager: um KG/JSON-IDs (z. B. "ns=4;s=OPCUA.x") in NodeKey-Strukturen
//     zu überführen und damit gegen InventorySnapshot-Werte zu vergleichen.

#pragma once
#include <string>
#include <cstdint>

inline bool parseNsAndId(const std::string& full,
                         uint16_t& nsOut,
                         std::string& idStrOut,
                         char& idTypeOut) {
    nsOut = 4; idStrOut.clear(); idTypeOut = '?';

    // Kurzform: “OPCUA.…”
    if (full.rfind("OPCUA.", 0) == 0) {
        idTypeOut = 's';
        idStrOut  = full;
        return true;
    }
    // Langform: ns=<n>;<t>=<id>
    if (full.rfind("ns=", 0) != 0) return false;
    const size_t semi = full.find(';');
    if (semi == std::string::npos) return false;

    try {
        nsOut = static_cast<uint16_t>(std::stoi(full.substr(3, semi - 3)));
    } catch (...) {
        nsOut = 4;
    }
    if (semi + 2 >= full.size()) return false;
    idTypeOut = full[semi + 1];
    if (full[semi + 2] != '=') return false;
    idStrOut = full.substr(semi + 3);
    return true;
}
