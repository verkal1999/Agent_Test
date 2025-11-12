// main.cpp
// Einstiegspunkt der Anwendung: initialisiert Python, den EventBus, den PLCMonitor
// und verbindet die Komponenten gemäß deinem MPA-Draft (Trigger D1/D2/D3, KG-Abfragen usw.).
#include "PLCMonitor.h"
#include "EventBus.h"
#include "ReactionManager.h"
#include "AckLogger.h"
#include "PythonRuntime.h"
#include "PythonWorker.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <pybind11/embed.h>
#include "InventorySnapshot.h"
#include "InventorySnapshotUtils.h"
#include "FailureRecorder.h"
#include "TimeBlogger.h"


namespace py = pybind11;

int main() {
    // Interpreter starten
    py::scoped_interpreter guard{};

    // *** WICHTIG: GIL für den Main-Thread freigeben, damit andere Threads (PythonWorker)
    //     Python ausführen können. Der GIL bleibt bis zum Ende von main() freigegeben.
    static std::unique_ptr<py::gil_scoped_release> main_gil_release;
    main_gil_release = std::make_unique<py::gil_scoped_release>();
    //py::gil_scoped_release main_gil_release;

    // PythonWorker starten
    PythonWorker::instance().start();

    PythonWorker::instance().call([]{
        namespace py = pybind11;

        // 1) Pfad setzen – KEIN py::str(char*), sondern sicher casten
        py::module_ sys  = py::module_::import("sys");
        py::list    path = sys.attr("path").cast<py::list>();

        // Hier den Ordner eintragen, der *KG_Interface.py* oder *KG_Interface/__init__.py* enthält:
        // Tipp: UTF-8 Literal + std::string vermeidet char*-Spezialfälle.
        const std::string src_dir = R"(C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\Implementierung_MPA\MSRGuard\src)";
        path.insert(0, py::cast(src_dir));

        // 2) Optional: venv-Site-Packages hinzufügen (falls benutzt)
        // py::module_::import("site").attr("addsitedir")(py::str(u8R"(C:\pfad\zu\venv\Lib\site-packages)"));

        // 3) Diagnose: zeig uns die ersten Pfade
        py::print("[Py] exe=", sys.attr("executable"), " prefix=", sys.attr("prefix"));
        py::print("[Py] sys.path[0..2] = ", sys.attr("path").attr("__getitem__")(0),
                                    ", ", sys.attr("path").attr("__getitem__")(1),
                                    ", ", sys.attr("path").attr("__getitem__")(2));

        // 4) Vorab prüfen, ob das Modul gefunden wird
        py::object spec = py::module_::import("importlib.util").attr("find_spec")("KG_Interface");
        if (spec.is_none()) {
            py::print("[KG] find_spec('KG_Interface') -> None ; sys.path=", sys.attr("path"));
            throw std::runtime_error("KG_Interface not found on sys.path");
        }

        // 5) Warm-Up: tatsächlicher Import
        py::module_::import("KG_Interface");
        std::cout << "[KG] warm-up import done\n";
    });

    PLCMonitor::Options opt;
    opt.endpoint       = "opc.tcp://DESKTOP-LNJR8E0:4840";
    opt.username       = "VDAdmin";
    opt.password       = "123456";
    opt.certDerPath    = R"(..\..\certificates\client_cert.der)";
    opt.keyDerPath     = R"(..\..\certificates\client_key.der)";
    opt.applicationUri = "urn:DESKTOP-LNJR8E0:Test:opcua-client";
    opt.nsIndex        = 4;

    PLCMonitor mon(opt);
    if (!mon.connect()) {
        std::cerr << "[Client] connect() failed\n";
        return 1;
    }
    std::cout << "[Client] connected\n";

    // 6) EventBus + ReactionManager + Logger + Abos
    EventBus bus;
    auto rm        = std::make_shared<ReactionManager>(mon, bus);
    rm->setLogLevel(ReactionManager::LogLevel::Info);
    auto subD2     = bus.subscribe_scoped(EventType::evD2, rm, 4);
    //auto subD1 = bus.subscribe_scoped(EventType::evD1, rm, 4);
    //auto subD3 = bus.subscribe_scoped(EventType::evD3, rm, 4);
    auto ackLogger = std::make_shared<AckLogger>();
    auto subPlan   = bus.subscribe_scoped(EventType::evSRPlanned, ackLogger, 1);
    auto subDone   = bus.subscribe_scoped(EventType::evSRDone,    ackLogger, 1);
    auto subPlan2   = bus.subscribe_scoped(EventType::evMonActPlanned, ackLogger, 1);
    auto subDone2   = bus.subscribe_scoped(EventType::evMonActDone,    ackLogger, 1);
    auto subProcessFail   = bus.subscribe_scoped(EventType::evProcessFail,    ackLogger, 1);
    auto subKGRes  = bus.subscribe_scoped(EventType::evKGResult,        rm, 4);
    auto subKGTo   = bus.subscribe_scoped(EventType::evKGTimeout,       rm, 4);
    auto rec = std::make_shared<FailureRecorder>(bus);
    rec->subscribeAll();   // registriert Observer für alle EventTypes
    auto subIngPlan = bus.subscribe_scoped(EventType::evIngestionPlanned, ackLogger, 1);
    auto subIngDone = bus.subscribe_scoped(EventType::evIngestionDone,    ackLogger, 1);
    auto subUnknown = bus.subscribe_scoped(EventType::evUnknownFM, ackLogger, 1);

    // 7) Trigger-Subscription → Event
    mon.subscribeBool("OPCUA.TriggerD3", opt.nsIndex, 0.0, 10,
        [&](bool b, const UA_DataValue& dv) {
            static std::atomic<bool> initialized{false};
            static std::atomic<bool> prev{false};

            std::cout << "[TrigD3] b=" << (b ? "true" : "false")
                    << " sourceTs=" << static_cast<UA_UInt64>(dv.sourceTimestamp)
                    << " serverTs=" << static_cast<UA_UInt64>(dv.serverTimestamp)
                    << "\n";

            if (!initialized.exchange(true)) { prev = b; return; }
            if (!b) { prev = false; return; }
            if (prev.exchange(true)) return;

            mon.post([&]{
                InventorySnapshot inv;
                const bool ok = buildInventorySnapshotNow(mon, "PLC", inv);
                std::cout << "[Debug] Snapshot D3 = " << (ok ? "OK":"FAIL") << "\n";
                dumpInventorySnapshot(inv);

                const auto now = std::chrono::steady_clock::now();
                const std::string corr = "evD3-" + std::to_string(now.time_since_epoch().count());

                bus.post({ EventType::evD3, now, std::any{ D2Snapshot{ corr, std::move(inv) } } });
                bus.post({ EventType::evUnknownFM, now,
                        std::any{ UnknownFMAck{ corr, "UnknownFM", "Triggered by D3" } } });
            });
        });
    mon.subscribeBool("OPCUA.TriggerD1", opt.nsIndex, 0.0, 10,
        [&](bool b, const UA_DataValue& dv) {
            static std::atomic<bool> initialized{false};
            static std::atomic<bool> prev{false};

            std::cout << "[TrigD1] b=" << (b ? "true" : "false")
                    << " sourceTs=" << static_cast<UA_UInt64>(dv.sourceTimestamp)
                    << " serverTs=" << static_cast<UA_UInt64>(dv.serverTimestamp)
                    << "\n";

            if (!initialized.exchange(true)) { prev = b; return; }
            if (!b) { prev = false; return; }
            if (prev.exchange(true)) return;

            mon.post([&]{
                InventorySnapshot inv;
                const bool ok = buildInventorySnapshotNow(mon, "PLC", inv);
                std::cout << "[Debug] Snapshot D1 = " << (ok ? "OK":"FAIL") << "\n";
                dumpInventorySnapshot(inv);

                const auto now = std::chrono::steady_clock::now();
                const std::string corr = "evD1-" + std::to_string(now.time_since_epoch().count());

                bus.post({ EventType::evD1, now, std::any{ D2Snapshot{ corr, std::move(inv) } } });
                bus.post({ EventType::evUnknownFM, now,
                        std::any{ UnknownFMAck{ corr, "UnknownFM", "Triggered by D1" } } });
            });
        });
    mon.subscribeBool("OPCUA.TriggerD2", opt.nsIndex, 0.0, 10,
        [&](bool b, const UA_DataValue& dv) {
            static std::atomic<bool> initialized{false};
            static std::atomic<bool> prev{false};

            std::cout << "[TrigD2] b=" << (b ? "true" : "false")
                    << " sourceTs=" << static_cast<UA_UInt64>(dv.sourceTimestamp)
                    << " serverTs=" << static_cast<UA_UInt64>(dv.serverTimestamp)
                    << "\n";

            if (!initialized.exchange(true)) { prev = b; return; }
            if (!b) { prev = false; return; }
            if (prev.exchange(true)) return;

            mon.post([&]{
                InventorySnapshot inv;
                const bool ok = buildInventorySnapshotNow(mon, "PLC", inv);
                std::cout << "[Debug] Snapshot D2 = " << (ok ? "OK":"FAIL") << "\n";
                dumpInventorySnapshot(inv);

                const auto now = std::chrono::steady_clock::now();
                const std::string corr = "evD2-" + std::to_string(now.time_since_epoch().count());

                bus.post({ EventType::evD2, now, std::any{ D2Snapshot{ corr, std::move(inv) } } });
                bus.post({ EventType::evUnknownFM, now,
                        std::any{ UnknownFMAck{ corr, "UnknownFM", "Triggered by D2" } } });
            });
        });
    /*
    std::atomic<bool> d2Prev{false};
    mon.subscribeBool("OPCUA.TriggerD2", opt.nsIndex, 0.0, 10, [&](bool b, const UA_DataValue&) {
        static std::atomic<bool> d2Prev{false};
        if (!b) { d2Prev = false; return; }
        if (d2Prev.exchange(true)) return;

        mon.post([&]{
            InventorySnapshot inv;
            const bool ok = buildInventorySnapshotNow(mon, "PLC", inv);
            std::cout << "[Debug] buildInventorySnapshotNow = " << (ok ? "OK":"FAIL") << "\n";
            dumpInventorySnapshot(inv);  // <— kompletter Dump hier

            const std::string corr = "evD2-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            bus.post({ EventType::evD2, std::chrono::steady_clock::now(),
                    std::any{ D2Snapshot{ corr, std::move(inv) } } });
        });
    }); */
    std::cout << "[Client] subscribed: ns=4;s=TriggerD2\n";
        
    auto tb = std::make_shared<TimeBlogger>(bus);
    tb->subscribeAll();

    // 8) Main-Loop
    for (;;) {
        mon.runIterate(50);   
        mon.processPosted(16);
        bus.process(16);
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // (Nie erreicht) PythonWorker::instance().stop();
}