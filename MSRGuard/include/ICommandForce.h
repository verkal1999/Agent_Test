// ICommandForce – Abstraktion für „Befehlskräfte“
// 
// In der MPA beschreibt ein Plan (Plan/Operation) eine Folge von Operationen,
// die von einer ICommandForce konkret ausgeführt werden:
//  - PLCCommandForce  : führt SPS-nahe Operationen aus (WriteBool, PulseBool, …).
//  - KgIngestionForce : führt KG-Ingestions-Operationen aus.
//  - WriteCSVForce    : schreibt CSV-Dateien (TimeBlogger).
//
// Die Factory (CommandForceFactory) liefert zur passenden Operation eine
// ICommandForce-Instanz, die execute(const Plan&) synchron ausführt.
#pragma once
#include "Plan.h"

struct ICommandForce {
    virtual ~ICommandForce() = default;
    // synchroner Ablauf; Rückgabe 1 = OK, 0 = Fehler
    virtual int execute(const Plan& p) = 0;
};