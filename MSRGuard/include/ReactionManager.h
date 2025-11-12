#pragma once
#include "ReactiveObserver.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <ostream>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "MonActionForce.h"
#include "SystemReactionForce.h"
#include "CommandForceFactory.h"
#include "PLCCommandForce.h"
#include "Event.h"
#include "PLCMonitor.h"
#include "Plan.h"
#include "InventorySnapshot.h"   // NodeKey, InventorySnapshot, D2Snapshot

class EventBus;

class ReactionManager : public ReactiveObserver {
public:
    enum class LogLevel { Error=0, Warn=1, Info=2, Debug=3, Trace=4, Verbose=5 };

    ReactionManager(PLCMonitor& mon, EventBus& bus);
    ~ReactionManager();

    void onEvent(const Event& ev) override;

    void setLogLevel(LogLevel lvl) { logLevel_.store(static_cast<int>(lvl), std::memory_order_relaxed); }
    LogLevel getLogLevel() const   { return static_cast<LogLevel>(logLevel_.load(std::memory_order_relaxed)); }
    bool isEnabled(LogLevel lvl) const { return static_cast<int>(lvl) <= logLevel_.load(std::memory_order_relaxed); }

    // ---- Vergleich/Normalisierung aus KG ------------------------------------
    enum class KgValKind { Bool, Int16, Float64, String };

    struct KgExpect {
        NodeKey     key;
        KgValKind   kind{KgValKind::Bool};
        bool        expectedBool{false};
        int16_t     expectedI16{0};
        double      expectedF64{0.0};
        std::string expectedStr;
    };

    struct ComparisonItem {
        NodeKey     key;
        bool        ok{false};
        std::string detail;
    };
    struct ComparisonReport {
        bool allOk{true};
        std::vector<ComparisonItem> items;
    };

    struct KgCandidate {
        std::string              potFM;    // IRI/ID der potenziellen FailureMode
        std::vector<KgExpect>    expects;  // zu prüfende Istwerte (nur gegen Cache!)
    };

private:
    // --- Umgebung
    PLCMonitor& mon_;
    EventBus&   bus_;

    // --- Worker
    std::jthread worker_;
    std::mutex               job_mx_;
    std::condition_variable  job_cv_;
    std::queue<std::function<void(std::stop_token)>> jobs_;

    // --- Logging
    std::atomic<int> logLevel_{static_cast<int>(LogLevel::Info)};
    const char* toCStr(LogLevel lvl) const;
    std::ostream& log(LogLevel lvl) const;

    // --- Hilfen
    static std::string makeCorrelationId(const char* evName);
    static std::ostream& nullout();
    mutable std::ostream* nullout_ = nullptr;

    // Inventar-Logging (nur Debug/Info)
    void logInventoryVariables(const InventorySnapshot& inv) const;

    // Cache-Helper
    static std::string getStringFromCache(const InventorySnapshot& inv, uint16_t ns, const std::string& id);
    static std::string getLastExecutedSkill(const InventorySnapshot& inv); // **nur Cache**, kein UA-Read

    // KG-Anbindung (Python)
    std::string fetchFailureModeParameters(const std::string& skillName);
    std::string fetchMonitoringActionForFM(const std::string& fmIri);
    std::string fetchSystemReactionForFM(const std::string& fmIri);

    // Normalisieren & Entscheiden
    static std::vector<KgExpect>    normalizeKgResponse(const std::string& rowsJson);
    std::vector<KgCandidate> normalizeKgPotFM(const std::string& rowsJson);
    ComparisonReport         compareAgainstCache(const InventorySnapshot& inv, const std::vector<KgExpect>& expects);
    std::vector<KgCandidate> selectPotFMByChecks(const InventorySnapshot& inv,
                                                        const std::vector<KgCandidate>& cands);

    // Plan-Erstellung & -Ausführung
    Plan buildPlanFromComparison(const std::string& corr, const ComparisonReport& rep) const;
    void createCommandForceForPlanAndAck(const Plan& plan,
                                         bool checksOk,
                                         const std::string& processNameForFail);
};
