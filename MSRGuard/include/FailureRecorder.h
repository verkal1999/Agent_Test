#pragma once

#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <nlohmann/json.hpp>

#include "ReactiveObserver.h"
#include "EventBus.h"
#include "Event.h"
#include "Plan.h"
#include "InventorySnapshot.h"    // InventorySnapshot / D2Snapshot
#include "KGIngestionParams.h"    // KgIngestionParams (siehe oben)

class FailureRecorder : public ReactiveObserver,
                        public std::enable_shared_from_this<FailureRecorder> {
public:
    explicit FailureRecorder(EventBus& bus) : bus_(bus) {}

    void subscribeAll();
    void onEvent(const Event& ev) override;

private:
    void resetCorrUnlocked(const std::string& corr);
    std::unordered_set<std::string> activeCorr_;
    using json = nlohmann::json;

    EventBus& bus_;

    // Recorder-interner Cache je correlationId:
    std::mutex mx_;
    std::unordered_map<std::string, std::string>               snapshotJsonByCorr_;
    std::unordered_map<std::string, std::vector<std::string>>  monReactsByCorr_;
    std::unordered_map<std::string, std::vector<std::string>>  sysReactsByCorr_;
    std::unordered_set<std::string>                             ingestionStarted_;
    std::unordered_map<std::string, std::string> failureModeByCorr_;
    // optional: FailureModesByCorr_ kannst du später genauso hinzufügen
    bool tryMarkIngestion(const std::string& corr);
    // Helpers
    static json        snapshotToJson(const InventorySnapshot& inv); // (legacy) unbenutzt hier
    static std::string now_ts();
    static std::string wrapSnapshot(const std::string& js);
    static std::string findStringInSnap(const json& snap, const char* nodeId);
    static json        snapshotToJson_flat(const InventorySnapshot& inv);

    std::shared_ptr<KgIngestionParams>
    buildParams(const std::string& corr, const std::string& process, const std::string& summary);

    void startIngestionWith(std::shared_ptr<KgIngestionParams> prm);
};
