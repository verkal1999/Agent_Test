#pragma once
#include "PLCCommandForce.h"
#include "CommandForceFactory.h"
#include "EventBus.h"
#include <pybind11/embed.h>
#include "PythonWorker.h"
#include <nlohmann/json.hpp>
#include <optional>

class KgIngestionForce final : public ICommandForce {
public:
    explicit KgIngestionForce(EventBus& bus) : bus_(bus) {}
    int execute(const Plan& p) override;

private:
    EventBus& bus_;

    struct Parameters {
        // Meta
        std::string corr;
        std::string process;
        std::string summary;
        std::string resourceId;
        // Zeit/Name
        std::string ts;               // "YYYY-MM-DD_HH-mm-ss"
        std::string individualName;   // corr + "_" + ts
        // Snapshot
        std::string snapshotWrapped;  // "==InventorySnapshot==...==InventorySnapshot=="
        std::string lastSkill;
        std::string lastProcess;
        // Optionales Zusatzwissen
        std::vector<std::string> sysReacts;    // IRIs
        std::vector<std::string> monReacts;    // IRIs
        std::vector<std::string> failureModes; // IRIs

        nlohmann::json toJson() const {
            using nlohmann::json;
            return json{
                {"corr", corr},
                {"process", process},
                {"summary", summary},
                {"resourceId", resourceId},
                {"ts", ts},
                {"individualName", individualName},
                {"snapshot", snapshotWrapped},
                {"lastSkill", lastSkill},
                {"lastProcess", lastProcess},
                {"sysReactions", sysReacts},
                {"monReactions", monReacts},
                {"failureModes", failureModes}
            };
        }
    };

    // helpers
    static std::string getStr(const UAValueMap& m, int idx);
    static std::string now_ts(); // "YYYY-MM-DD_HH-mm-ss"
    static std::string wrapSnapshot(const std::string& js) {
        return "==InventorySnapshot==" + js + "==InventorySnapshot==";
    }
    static std::string findStringInSnapshot(const nlohmann::json& snap, const char* nodeId);

    static Parameters buildParams(const Plan& p);
};

