// TimeBlogger.cpp
// Komponente, die Zeiten zwischen Events misst, pro correlationId eine Timeline aufbaut
// und diese später in eine CSV-Datei schreiben kann, um das Verhalten des Systems zu analysieren.
#include "TimeBlogger.h"
#include "EventBus.h"   
#include "AckLogger.h"   
#include <iomanip>
#include <sstream>
#include "CommandForceFactory.h"
#include "ICommandForce.h"
#include "WriteCsvParams.h"
#include <filesystem>
//using namespace std;

using CsvRow = ::CsvRow;
TimeBlogger::TimeBlogger(EventBus& bus) : bus_(bus) {}

void TimeBlogger::subscribeAll() {
    auto self = shared_from_this();
    bus_.subscribe(EventType::evD2,               self, 3);
    bus_.subscribe(EventType::evD1,               self, 3);
    bus_.subscribe(EventType::evD3,               self, 3);
    bus_.subscribe(EventType::evMonActPlanned,    self, 3);
    bus_.subscribe(EventType::evMonActDone,       self, 3);
    bus_.subscribe(EventType::evKGResult,         self, 3);
    bus_.subscribe(EventType::evKGTimeout,        self, 3);
    bus_.subscribe(EventType::evMonActFinished,   self, 3);
    bus_.subscribe(EventType::evSysReactFinished, self, 3);
    bus_.subscribe(EventType::evProcessFail,      self, 3);
    bus_.subscribe(EventType::evSRDone,           self, 3);
    bus_.subscribe(EventType::evIngestionPlanned, self, 3);
    bus_.subscribe(EventType::evIngestionDone,    self, 3);
    bus_.subscribe(EventType::evUnknownFM,        self, 3);
    bus_.subscribe(EventType::evGotFM,            self, 3);
}

const char* TimeBlogger::toName_(EventType t) {
    switch (t) {
        case EventType::evD1: return "evD1";
        case EventType::evD2: return "evD2";
        case EventType::evD3: return "evD3";
        case EventType::evSRPlanned: return "evSRPlanned";
        case EventType::evSRDone: return "evSRDone";
        case EventType::evProcessFail: return "evProcessFail";
        case EventType::evMonActPlanned: return "evMonActPlanned";
        case EventType::evMonActDone: return "evMonActDone";
        case EventType::evKGResult: return "evKGResult";
        case EventType::evKGTimeout: return "evKGTimeout";
        case EventType::evIngestionPlanned: return "evIngestionPlanned";
        case EventType::evIngestionDone: return "evIngestionDone";
        case EventType::evMonActFinished: return "evMonActFinished";
        case EventType::evSysReactFinished: return "evSysReactFinished";
        case EventType::evUnknownFM: return "evUnknownFM";
        case EventType::evGotFM: return "evGotFM";
    }
    return "ev";
}

std::string TimeBlogger::extractCorrId_(const Event& ev) {
    // Acks
    if (auto p = std::any_cast<ReactionPlannedAck>(&ev.payload))    return p->correlationId;
    if (auto p = std::any_cast<ReactionDoneAck>(&ev.payload))       return p->correlationId;
    if (auto p = std::any_cast<ProcessFailAck>(&ev.payload))        return p->correlationId;
    if (auto p = std::any_cast<IngestionPlannedAck>(&ev.payload))   return p->correlationId;
    if (auto p = std::any_cast<IngestionDoneAck>(&ev.payload))      return p->correlationId;
    if (auto p = std::any_cast<MonActFinishedAck>(&ev.payload))     return p->correlationId;
    if (auto p = std::any_cast<SysReactFinishedAck>(&ev.payload))   return p->correlationId;
    if (auto p = std::any_cast<UnknownFMAck>(&ev.payload))          return p->correlationId;
    if (auto p = std::any_cast<GotFMAck>(&ev.payload))              return p->correlationId;


    if (auto p = std::any_cast<PLCSnapshotPayload>(&ev.payload))    return p->correlationId; // Event.h 

    return "ev-" + std::to_string(static_cast<int>(ev.type));
}

void TimeBlogger::onEvent(const Event& ev) {
    const char* evName = toName_(ev.type);
    const std::string corr = extractCorrId_(ev);

    DurationMs dtSincePrev{0};
    long long durMs = 0;
    long long sumMs = 0;

    {
        std::lock_guard<std::mutex> lk(mx_);
        auto& tl = tlByCorr_[corr];
        const auto now = Clock::now();

        if (tl.hasLast) dtSincePrev = std::chrono::duration_cast<DurationMs>(now - tl.lastTs);
        else            dtSincePrev = DurationMs{0};

        if (tl.marks.empty()) tl.t0 = now;
        tl.marks[evName] = now;
        tl.lastEventName = evName;
        tl.lastTs        = now;
        tl.hasLast       = true;

        durMs    = static_cast<long long>(dtSincePrev.count());
        tl.sumMs += durMs;
        sumMs    = tl.sumMs;
    }
    printAndCollect_(corr, evName, durMs, sumMs);
    handleEvent_(ev, corr, evName);
    if (ev.type == EventType::evIngestionDone) {
        finish(corr);
    }
}

void TimeBlogger::handleEvent_(const Event& ev, const std::string& corrId, const char* evName) {
    std::lock_guard<std::mutex> lk(mx_);
    auto it = tlByCorr_.find(corrId);
    if (it == tlByCorr_.end()) return;
    auto& tl = it->second;

    // Beispiel: bei Fail zusätzlich evD2→evProcessFail als Segment mitloggen
    if (ev.type == EventType::evProcessFail) {
        if (tl.marks.count("evD2")) recordSegment_(tl, "evD2", "evProcessFail");
    }
}

void TimeBlogger::recordSegment_(Timeline& tl, const std::string& from, const std::string& to) {
    auto itA = tl.marks.find(from);
    auto itB = tl.marks.find(to);
    if (itA == tl.marks.end() || itB == tl.marks.end()) return;
    auto ms = std::chrono::duration_cast<DurationMs>(itB->second - itA->second);
    // Optional: speichern, falls du später eine Sammelausgabe möchtest
    // tl.segments.emplace_back(from + "->" + to, ms);
    std::cout << "[Time][seg] " << from << "->" << to << " = " << ms.count() << " ms\n";
}

void TimeBlogger::mark(const std::string& corrId, const std::string& label) {
    std::lock_guard<std::mutex> lk(mx_);
    auto& tl = tlByCorr_[corrId];
    const auto now = Clock::now();
    if (tl.marks.empty()) tl.t0 = now;
    tl.marks[label] = now;
    tl.lastEventName = label;
    tl.lastTs        = now;
    tl.hasLast       = true;
}

bool TimeBlogger::delta(const std::string& corrId, const std::string& fromLabel,
                        const std::string& toLabel, DurationMs& out) const {
    std::lock_guard<std::mutex> lk(mx_);
    auto ti = tlByCorr_.find(corrId);
    if (ti == tlByCorr_.end()) return false;
    const auto& tl = ti->second;
    auto itA = tl.marks.find(fromLabel);
    auto itB = tl.marks.find(toLabel);
    if (itA == tl.marks.end() || itB == tl.marks.end()) return false;
    out = std::chrono::duration_cast<DurationMs>(itB->second - itA->second);
    return true;
}

void TimeBlogger::finish(const std::string& corrId) {
  std::vector<::CsvRow> rows;
  {
    std::lock_guard<std::mutex> lk(mx_);
    auto it = tlByCorr_.find(corrId);
    if (it != tlByCorr_.end()) {
      rows = std::move(it->second.csvRows);
      tlByCorr_.erase(it);
    }
  }
  if (rows.empty()) return;

  auto prm = std::make_shared<WriteCsvParams>();
  prm->rows = std::move(rows);

  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories("logs/time", ec); // erzeugt fehlende Ordner, bestehende ok. 
  prm->outFile = (fs::path("logs/time") / "timeblog_DefaultParams.csv").string();

  Plan p;
  p.correlationId = corrId;
  p.resourceId    = "TimeBlogger";

  Operation op;
  op.type   = OpType::WriteCSV;
  op.attach = prm; // später per std::any_cast<shared_ptr<WriteCsvParams>> in der Force holen. :contentReference[oaicite:1]{index=1}

  p.ops.push_back(std::move(op));

  if (auto cf = CommandForceFactory::createForOp(p.ops.front(), /*mon*/nullptr, bus_)) {
    (void)cf->execute(p);
  }
}

void TimeBlogger::printAndCollect_(const std::string& corr,
                                   const std::string& evName,
                                   long long durMs,
                                   long long sumMs)
{
    std::cout << "[Time][" << corr << "] "
              << evName << " (type=" << evName << ") "
              << "Δ=" << durMs << " ms\n"
              << "sum= " << sumMs << " ms\n";
    std::lock_guard<std::mutex> lk(mx_);
    tlByCorr_[corr].csvRows.push_back(CsvRow{corr, evName, durMs, sumMs});
}