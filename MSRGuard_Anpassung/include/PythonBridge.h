// PythonBridge.h
#pragma once
#include <string>
#include <vector>

struct SuggestionResult {
    std::vector<std::string> monitoringActions;
    std::vector<std::string> systemReactions;
};

class PythonBridge {
public:
    // Persistiert einen unbekannten Fehler mit isUnknownFailure=true
    static void ingestUnknownFailure(const std::string &id,
                                     const std::string &processName,
                                     const std::string &summary);

    // Ruft den Python-Agenten an und gibt Vorschläge zurück
    static SuggestionResult askExcHAgent(const std::string &id,
                                         const std::string &processName,
                                         const std::string &summary);
};