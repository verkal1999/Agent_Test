// PLCCommandForce (ICommandForce-Implementierung für SPS-Operationen)
// - Implementiert execute(const Plan&) und führt sequentiell die Operationen aus p.ops aus.
// - Unterstützte OpTypes: WriteBool, PulseBool, WriteInt32, WaitMs, ReadCheck,
//   BlockResource, RerouteOrders, UnblockResource (vgl. MPA_Draft CommandForceFactory).
// - Die eigentliche Kommunikation mit der SPS erfolgt über PLCMonitor (post/postDelayed).
// - Rückgabewert 1/0 signalisiert Erfolg/Fehlschlag der ausgeführten Plan-Schritte.
// Die Instanz wird über CommandForceFactory::create(UseMonitor, ...) bzw.
// createForOp(...) vom ReactionManager und weiteren Komponenten genutzt.
#include "PLCCommandForce.h"
#include "PLCMonitor.h"

#include <string>    // std::stoi
#include <iostream>
#include <thread>
#include <chrono>

PLCCommandForce::PLCCommandForce(PLCMonitor& mon, IOrderQueue* oq)
    : mon_(mon), oq_(oq) {}

int PLCCommandForce::execute(const Plan& p) {
    bool ok = true;

    for (const auto& op : p.ops) {
        switch (op.type) {
        case OpType::WriteBool: {
            const bool value = (op.arg == "true" || op.arg == "1");
            mon_.post([&m = mon_, op, value]{
                const bool wr = m.writeBool(op.nodeId, op.ns, value);
                std::cout << "[PLCCommandForce] WriteBool " << op.nodeId
                          << " ns=" << op.ns << " value=" << (value ? "true" : "false")
                          << " -> " << (wr ? "OK" : "FAIL") << "\n";
            });
            break;
        }
        case OpType::PulseBool: {
        // Pulsbreite in Millisekunden: aus op.timeoutMs oder Default 100 ms
        const int widthMs = (op.timeoutMs > 0) ? op.timeoutMs : 100;

        // Optional: Preclear (LOW setzen, um sicher always eine steigende Flanke zu erzeugen)
        // Falls gewünscht, aus op.arg ableiten, z. B. "preclear"
        const bool doPreclear = (op.arg == "preclear");

        PLCMonitor* pm = &mon_;   // robust in Threads verwenden
        const auto nodeId = op.nodeId;
        const auto ns     = op.ns;

        if (doPreclear) {
            pm->post([pm, nodeId, ns]{
                const bool wr = pm->writeBool(nodeId, ns, false);
                std::cout << "[PLCCommandForce] PulseBool PRECLEAR " << nodeId
                        << " ns=" << ns << " -> " << (wr ? "OK" : "FAIL") << "\n";
            });
            // Kleine Entprell-/SPS-Zeit, damit LOW sicher ankommt
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        // HIGH setzen (steigende Flanke auslösen)
        pm->post([pm, nodeId, ns]{
            const bool wr = pm->writeBool(nodeId, ns, true);
            std::cout << "[PLCCommandForce] PulseBool HI " << nodeId
                    << " ns=" << ns << " -> " << (wr ? "OK" : "FAIL") << "\n";
        });

        // Nach widthMs wieder auf LOW – sauber über Monitor-Timer
        pm->postDelayed(widthMs, [pm, nodeId, ns]{
            const bool wr = pm->writeBool(nodeId, ns, false);
            std::cout << "[PLCCommandForce] PulseBool LO " << nodeId
                    << " ns=" << ns << " -> " << (wr ? "OK" : "FAIL") << "\n";
        });

        break;
    }

        case OpType::WriteInt32: {
            // TODO: Implementiere writeInt32 in PLCMonitor, dann hier aktivieren
            int value = 0;
            try { value = std::stoi(op.arg); } catch (...) { value = 0; ok = false; }
            std::cout << "[PLCCommandForce] WriteInt32 TODO node=" << op.nodeId
                      << " ns=" << op.ns << " value=" << value << " (not implemented)\n";
            // mon_.post([&m = mon_, op, value]{ m.writeInt32(op.nodeId, op.ns, value); });
            break;
        }

        case OpType::CallMethod: {
            // TODO: op.arg als JSON der Method-Argumente parsen und callMethod aufrufen
            std::cout << "[PLCCommandForce] CallMethod TODO node=" << op.nodeId
                      << " ns=" << op.ns << " args='" << op.arg << "' (not implemented)\n";
            // mon_.post([&m = mon_, op]{ m.callMethod(...); });
            break;
        }

        case OpType::WaitMs: {
            const int ms = (op.timeoutMs > 0) ? op.timeoutMs : 0; // makro-sicher
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            break;
        }

        case OpType::ReadCheck: {
            std::cout << "[PLCCommandForce] ReadCheck TODO node=" << op.nodeId
                      << " ns=" << op.ns << " expect='" << op.arg
                      << "' timeoutMs=" << op.timeoutMs << " (not implemented)\n";
            // Optional: synchron lesen & prüfen -> ok = ok && result;
            break;
        }

        case OpType::BlockResource: {
            if (oq_) ok = oq_->blockResource(op.nodeId) && ok;
            else std::cout << "[PLCCommandForce] BlockResource(" << op.nodeId << ") (noop)\n";
            break;
        }

        case OpType::RerouteOrders: {
            if (oq_) ok = oq_->reroute(op.nodeId, op.arg) && ok;
            else std::cout << "[PLCCommandForce] RerouteOrders(" << op.nodeId
                           << ", criteria=" << op.arg << ") (noop)\n";
            break;
        }

        case OpType::UnblockResource: {
            if (oq_) ok = oq_->unblockResource(op.nodeId) && ok;
            else std::cout << "[PLCCommandForce] UnblockResource(" << op.nodeId << ") (noop)\n";
            break;
        }
        }
    }

    return ok ? 1 : 0;
}
