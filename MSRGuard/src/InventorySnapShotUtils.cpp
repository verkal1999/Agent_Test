// InventorySnapShotUtils.cpp
// Hilfsfunktionen, um aus dem PLCMonitor einen InventorySnapshot zu erzeugen,
// der sowohl Struktur (rows) als auch aktuelle Werte für relevante Variablen enthält.

#include "InventorySnapShotUtils.h"
#include <iostream>

bool parseNsAndId(const std::string &nodeId, uint16_t &ns, std::string &id, char &typeChar);

// Diese Funktion baut den Snapshot sofort, indem sie alle Variablen unterhalb von root
// browsed und dann ihre Werte mit den passenden Read-Funktionen einliest.
bool buildInventorySnapshotNow(PLCMonitor &mon, const std::string &root, InventorySnapshot &s) {
    s = InventorySnapshot{};
    mon.dumpPlcInventory(s.rows, root.c_str());

    // Schleife: iteriert über alle Elemente in s.rows.
    for (const auto &r : s.rows) {
        if (r.nodeClass != "Variable") continue;

        uint16_t ns = 0;
        std::string id;
        char type = 0;
        if (!parseNsAndId(r.nodeId, ns, id, type))
            continue;
        if (type != 's')
            continue;

        bool isBool   = (r.dtypeOrSig.find("Boolean") != std::string::npos);
        bool isString = (r.dtypeOrSig.find("String")  != std::string::npos) ||
                        (r.dtypeOrSig.find("STRING")  != std::string::npos);
        bool isI16    = (r.dtypeOrSig.find("Int16")   != std::string::npos);
        bool isF64    = (r.dtypeOrSig.find("Double")  != std::string::npos);
        bool isF32    = (r.dtypeOrSig.find("Float")   != std::string::npos);

        NodeKey k;
        k.ns   = ns;
        k.type = 's';
        k.id   = id;

        if (isBool) {
            bool v{};
            if (mon.readBoolAt(id, ns, v))
                s.bools.emplace(k, v);
        }
        if (isString) {
            std::string v;
            if (mon.readStringAt(id, ns, v))
                s.strings.emplace(k, v);
        }
        if (isI16) {
            UA_Int16 v{};
            if (mon.readInt16At(id, ns, v))
                s.int16s.emplace(k, static_cast<int16_t>(v));
        }
        if (isF64) {
            UA_Double v{};
            if (mon.readDoubleAt(id, ns, v))
                s.floats.emplace(k, static_cast<double>(v));
        }
        if (isF32) {
            UA_Float v{};
            if (mon.readFloatAt(id, ns, v))
                s.floats.emplace(k, static_cast<double>(v));
        }
    }

    return true;
}
