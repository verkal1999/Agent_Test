#pragma once
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <chrono>
#include <memory>
#include <any>
#include "ReactiveObserver.h"
#include "Event.h"     // Event, EventType (ev*-Typen)  
#include "Acks.h"      // Ack-Structs mit correlationId  
#include "WriteCsvParams.h"

class EventBus;

class TimeBlogger
  : public ReactiveObserver,
    public std::enable_shared_from_this<TimeBlogger> {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using DurationMs = std::chrono::milliseconds;

    explicit TimeBlogger(EventBus& bus);

    // Abonnieren NACH make_shared() aufrufen (nicht im Konstruktor)!
    void subscribeAll();

    void onEvent(const Event& ev) override;

    // (optional) manuelle Marken
    void mark(const std::string& corrId, const std::string& label);
    bool delta(const std::string& corrId, const std::string& fromLabel,
               const std::string& toLabel, DurationMs& out) const;
    void finish(const std::string& corrId);

private:
    void printAndCollect_(const std::string& corr,
                      const std::string& evName,
                      long long durMs,
                      long long sumMs);

        struct Timeline {
        Clock::time_point t0{};
        Clock::time_point lastTs{};
        bool hasLast{false};
        std::string lastEventName;
        std::unordered_map<std::string, Clock::time_point> marks;
        std::vector<::CsvRow> csvRows;
        long long sumMs{0};
    };
    

    // Helfer
    static std::string extractCorrId_(const Event& ev);
    static const char* toName_(EventType t);
    void recordSegment_(Timeline& tl, const std::string& from, const std::string& to);
    void handleEvent_(const Event& ev, const std::string& corrId, const char* evName);

private:
    EventBus& bus_;
    mutable std::mutex mx_;
    std::unordered_map<std::string, Timeline> tlByCorr_;
};
