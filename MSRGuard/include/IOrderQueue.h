// IOrderQueue – Abstraktion für Auftrags-/Ressourcensteuerung
//
// In der MPA dient diese Schnittstelle dazu, aus Reaktionsplänen heraus
// Ressourcen zu blockieren, Aufträge umzurouten oder wieder freizugeben, ohne
// dass PLCCommandForce die konkrete Implementierung der Produktions-IT kennen muss.
#pragma once
#include "Plan.h"
struct IOrderQueue {
    virtual ~IOrderQueue() = default;

    // Blockiert eine Ressource (z. B. Maschine/Anlage), sodass keine neuen Aufträge
    // mehr eingeplant werden. Implementierung ist domänenspezifisch.
    virtual bool blockResource(const std::string& /*resId*/) { return true; }

    // Routet Aufträge nach Kriterien (criteriaJson) auf eine andere Ressource um.
    // Im Prototyp ist das ein Stub (immer true).
    virtual bool reroute(const std::string& /*resId*/, const std::string& /*criteriaJson*/) { return true; }

    // Hebt eine Blockierung wieder auf; die Ressource kann wieder verplant werden.
    virtual bool unblockResource(const std::string& /*resId*/) { return true; }
};