// Plan.h – Beschreibung von Reaktionsplänen und Operationen
//
// Ein Plan ist die zentrale Ausführungseinheit:
//  - Er enthält eine geordnete Liste von Operationen (ops), die von
//    ICommandForce-Implementierungen (PLCCommandForce, KgIngestionForce, …)
//    sequentiell abgearbeitet wird.
//  - Die Operationen sind bewusst „kleine Bausteine“ (WriteBool, CallMethod,
//    WaitMs, BlockResource, KGIngestion, …), um flexibel von KG/JSON zu C++
//    mappen zu können (siehe PlanJsonUtils).
//
// correlationId : verbindet Plan mit D2/D3, KG-Ergebnissen, Ingestion usw.
// resourceId    : logische Ressource/Anlage, auf die sich der Plan bezieht.
#pragma once
#include <string>
#include <vector>
#include <map>  
#include <any>             
#include "common_types.h"    // UAValue, UAValueMap (neu)

/// Primitive, aus denen ein Reaktionsplan besteht.
enum class OpType {
    WriteBool,
    PulseBool,
    WriteInt32,
    // Aufrufe (semantisch getrennt):
    CallMethod,             // generischer Method-Call (falls genutzt)
    CallMonitoringActions,  // <— MonActionForce-Domain
    CallSystemReaction,     // <— SystemReactionForce-Domain
    WaitMs,
    ReadCheck,
    BlockResource,
    RerouteOrders,
    UnblockResource,
    KGIngestion,             // KG-Ingestion (nicht-PLC)
    WriteCSV
};

/// Eine einzelne Operation (Schritt) in einem Plan.
struct Operation {
    OpType type;

    // Allgemein (für Write*/ReadCheck):
    std::string   nodeId;                 // z. B. "DiagnoseFinished"
    unsigned short ns   = 1;              // Namespace Index

    // ---- CallMethod-spezifisch ----
    std::string   callObjNodeId;          // Object-NodeId (StringId)
    std::string   callMethNodeId;         // Method-NodeId (StringId)
    unsigned short callNsObj  = 0;        // 0 = "nicht gesetzt" -> 'ns' nutzen
    unsigned short callNsMeth = 0;        // 0 = "nicht gesetzt" -> 'ns' nutzen

    // Typisierte Argumente (nur für CallMethod genutzt):
    UAValueMap    inputs;                 // index -> UAValue (bool,int16,int32,float,double,string)
    UAValueMap    expOuts;                // erwartete Outputs (index -> UAValue)

    // Legacy/sonstige:
    // WriteBool:  "true"/"false"
    // WriteInt32: "42"
    // CallMethod: wird künftig ignoriert (nur noch inputs/expOuts)
    std::string   arg;

    // Zeitfeld:
    // WaitMs:      Wartezeit
    // PulseBool:   Pulsbreite
    // CallMethod:  Call-Timeout (Client->config->timeout)
    int           timeoutMs = 0;

    
    // Opaque Cargo für spezielle OpTypes (z. B. KGIngestion)
    // Hier legen wir ein std::shared_ptr<KgIngestionParams> ab.
    std::any      attach;
};

/// Gesamter Reaktionsplan …
struct Plan {
    std::string              correlationId;
    std::string              resourceId;
    std::vector<Operation>   ops;
    bool                     abortRequired  = false;
    bool                     degradeAllowed = false;
};