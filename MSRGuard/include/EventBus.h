#pragma once
#include "Event.h"
#include "ReactiveObserver.h"
#include <unordered_map>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <atomic>
#include <cstdint>

struct EventTypeHash {
    size_t operator()(EventType t) const noexcept {
        return static_cast<size_t>(t);
    }
};

// Eindeutiger Schlüssel für eine Subscription:
struct SubscriptionToken {
    EventType type{};
    std::uint64_t id{0};
    explicit operator bool() const noexcept { return id != 0; }
};

// Optionales RAII-Handle (löscht Subscription im Destruktor):
class EventBus; // fwd
class Subscription {
public:
    Subscription() = default;
    Subscription(EventBus* bus, SubscriptionToken tok)
        : bus_(bus), tok_(tok) {}
    Subscription(const Subscription&) = delete;
    Subscription& operator=(const Subscription&) = delete;
    Subscription(Subscription&& other) noexcept { swap(other); }
    Subscription& operator=(Subscription&& other) noexcept {
        if (this != &other) { unsubscribe(); swap(other); }
        return *this;
    }
    ~Subscription() { unsubscribe(); }

    SubscriptionToken token() const noexcept { return tok_; }
    void unsubscribe();

private:
    void swap(Subscription& o) noexcept { std::swap(bus_, o.bus_); std::swap(tok_, o.tok_); }
    EventBus* bus_{nullptr};
    SubscriptionToken tok_{};
};

// --------------------------------------------------------------------------------------------

class EventBus {
public:
    EventBus() = default;

    // Prioritätenbereich: 1..4 (Konvention: 4 z. B. TaskManager)
    static constexpr int kMinPriority = 1;
    static constexpr int kMaxPriority = 4;

    // Observer registrieren; Rückgabe: Token (oder nutze RAII-Variante unten)
    SubscriptionToken subscribe(EventType t,
                                const std::shared_ptr<ReactiveObserver>& obs,
                                int priority = kMinPriority);

    // Komfort: RAII-Handle zurückgeben (automatisch unsubscribe im Destruktor)
    Subscription subscribe_scoped(EventType t,
                                  const std::shared_ptr<ReactiveObserver>& obs,
                                  int priority = kMinPriority) {
        return Subscription(this, subscribe(t, obs, priority));
    }

    // Manuelles Abbestellen
    void unsubscribe(const SubscriptionToken& tok);

    // Ereignisse einreihen (thread-sicher, asynchron)
    void post(Event ev);

    // Sofort verteilen (synchron; vorsichtig bzgl. Reentranz)
    void post_now(const Event& ev);

    // Warteschlange bearbeiten; maxEvents = Schutz gegen Starvation
    void process(size_t maxEvents = 32);

    // Queue leeren (optional)
    void clear_queue();

private:
    struct Entry {
        std::weak_ptr<ReactiveObserver> wp;
        std::uint64_t id{0};     // Anmelde-Reihenfolge (kleiner = älter)
        int priority{kMinPriority};
    };

    void dispatch_one(const Event& ev);
    void sweep_dead(EventType t); // tote weak_ptrs wegräumen

    std::mutex mx_;
    std::deque<Event> q_;
    std::unordered_map<EventType, std::vector<Entry>, EventTypeHash> listeners_;
    std::atomic<std::uint64_t> nextId_{1};

    friend class Subscription;
};
