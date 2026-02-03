#include "EventBus.h"
#include <algorithm> // sort, remove_if

SubscriptionToken EventBus::subscribe(EventType t,
                                      const std::shared_ptr<ReactiveObserver>& obs,
                                      int priority) {
    SubscriptionToken tok;
    if (!obs) return tok;

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

void EventBus::post(Event ev) {
    std::lock_guard<std::mutex> lk(mx_);
    q_.push_back(std::move(ev));
}

void EventBus::post_now(const Event& ev) {
    dispatch_one(ev);
}

void EventBus::process(size_t maxEvents) {
    for (size_t i = 0; i < maxEvents; ++i) {
        Event ev;
        {
            std::lock_guard<std::mutex> lk(mx_);
            if (q_.empty()) break;
            ev = std::move(q_.front());
            q_.pop_front();
        }
        dispatch_one(ev);
    }
}

void EventBus::clear_queue() {
    std::lock_guard<std::mutex> lk(mx_);
    q_.clear();
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
            if (a.priority != b.priority) return a.priority > b.priority; // höher zuerst
            return a.id < b.id; // ältere zuerst
        });

    bool anyDead = false;

    for (const auto& e : copy) {
        if (auto sp = e.wp.lock()) {
            sp->onEvent(ev);
        } else {
            anyDead = true;
        }
    }

    if (anyDead) sweep_dead(ev.type);
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

void Subscription::unsubscribe() {
    if (bus_ && tok_) {
        bus_->unsubscribe(tok_);
        tok_ = {};
        bus_ = nullptr;
    }
}
