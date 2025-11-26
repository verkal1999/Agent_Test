// EventBus.cpp
// Ereignisbus, der Observer registriert und Events asynchron an alle Listener verteilt.
// Kern des Event-getriebenen Frameworks in deiner MPA, entkoppelt Sender und Empfänger.

#include "EventBus.h"
#include <algorithm> // sort, remove_if



SubscriptionToken EventBus::subscribe(EventType t,
                                      const std::shared_ptr<ReactiveObserver>& obs,
                                      int priority) {
    SubscriptionToken tok;
    if (!obs) return tok;

    // Priority clamp
    if (priority < kMinPriority) priority = kMinPriority;
    if (priority > kMaxPriority) priority = kMaxPriority;

    std::lock_guard<std::mutex> lk(mx_);
    const auto id = nextId_.fetch_add(1, std::memory_order_relaxed);
    listeners_[t].push_back(Entry{
        std::weak_ptr<ReactiveObserver>(obs),
        id,
        priority
    });
    tok = SubscriptionToken{ t, id };
    return tok;
}

void EventBus::unsubscribe(const SubscriptionToken& tok) {
    if (!tok) return;
    std::lock_guard<std::mutex> lk(mx_);
    auto it = listeners_.find(tok.type);
    if (it == listeners_.end()) return;
    auto& vec = it->second;
    vec.erase(std::remove_if(vec.begin(), vec.end(),
             [&](const Entry& e){ return e.id == tok.id; }),
             vec.end());
}

void EventBus::post(const Event& ev) {
    {
        std::lock_guard<std::mutex> lk(q_mx_);
        queue_.push(ev);
    }
    // Der Event wird nur in die Queue gestellt, verarbeitet wird er später in process().
}

void EventBus::dispatch_one(const Event& ev) {
    std::vector<Entry> copy;
    {
        std::lock_guard<std::mutex> lk(mx_);
        auto it = listeners_.find(ev.type);
        if (it == listeners_.end()) return;
        copy = it->second;
    }

    std::sort(copy.begin(), copy.end(),
        [](const Entry& a, const Entry& b){
            if (a.priority != b.priority)
                return a.priority > b.priority;
            return a.id < b.id; // ältere zuerst
        });

    bool anyDead = false;

    // Schleife: iteriert über alle Elemente in copy.
    for (const auto& e : copy) {
        if (auto sp = e.wp.lock()) {
            sp->onEvent(ev);
        } else {
            anyDead = true;
        }
    }

    if (anyDead) sweep_dead(ev.type);
}

void EventBus::process(size_t maxEvents) {
    // Schleife: klassische Zählschleife, die über Indizes oder Zähler läuft.
    for (size_t i = 0; i < maxEvents; ++i) {
        Event ev;
        {
            std::lock_guard<std::mutex> lk(q_mx_);
            if (queue_.empty()) break;
            ev = queue_.front();
            queue_.pop();
        }
        dispatch_one(ev);
    }
}

void EventBus::sweep_dead(EventType t) {
    std::lock_guard<std::mutex> lk(mx_);
    auto it = listeners_.find(t);
    if (it == listeners_.end()) return;
    auto& vec = it->second;
    vec.erase(std::remove_if(vec.begin(), vec.end(),
             [](const Entry& e){ return e.wp.expired(); }),
             vec.end());
}

// -------- Subscription (RAII) ----------
void Subscription::unsubscribe() {
    if (bus_ && tok_) {
        bus_->unsubscribe(tok_);
        tok_ = {};
        bus_ = nullptr;
    }
}