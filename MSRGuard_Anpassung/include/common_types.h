// common_types.h – Grundtypen für OPC-UA-Werte
//
// UAValue    : schlanker Variant-Typ für die typisierten OPC-UA-Werte (bool, int16,
//              int32, float, double, string). Dient als neutrale Zwischenschicht
//              zwischen KG-JSON, Plan/Operation und PLCMonitor.
// UAValueMap : ordnet einen ganzzahligen Index (z. B. Argumentposition) einem UAValue zu.
//
// tagOf(...) : liefert eine einfache String-Repräsentation des Variant-Tags.
// almostEqual / equalUA : Vergleich von UAValue, insbesondere tolerant für
//              Fließkommazahlen (Floating-Point-Fehler gemäß MPA_Draft).
#pragma once
#include <variant>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

// Variant-Repräsentation eines einzelnen UA-Wertes (entspricht der kleinen
// Teilmenge an Datentypen, die im Plan / bei CallMethod genutzt werden soll).
using UAValue    = std::variant<std::monostate, bool, int16_t, int32_t, float, double, std::string>;
using UAValueMap = std::map<int, UAValue>;

// Liefert eine einfache Typkennung als Text (z. B. "bool", "int16", "double").
inline const char* tagOf(const UAValue& v) {
  switch (v.index()) {
    case 1: return "bool";
    case 2: return "int16";
    case 3: return "int32";
    case 4: return "float";
    case 5: return "double";
    case 6: return "string";
    default: return "null";
  }
}
// Toleranter Vergleich von double-Werten nach relative/absolute Toleranz.
// Wird von equalUA für float/double verwendet.
inline bool almostEqual(double a, double b, double relTol = 1e-6, double absTol = 1e-9) {
  const double diff  = std::fabs(a - b);
  const double scale = (std::max)(std::fabs(a), std::fabs(b)); // Klammern gegen Windows-Makros
  const double thr   = (std::max)(absTol, relTol * scale);
  return diff <= thr;
}

// Vergleich zweier UAValue (Float/Double tolerant)
inline bool equalUA(const UAValue& a, const UAValue& b,
                    double relTol = 1e-6, double absTol = 1e-9) {
  if (a.index() != b.index()) return false;
  return std::visit([&](auto&& x)->bool{
    using T = std::decay_t<decltype(x)>;
    const T& y = std::get<T>(b);
    if constexpr (std::is_same_v<T,float>)  return almostEqual((double)x, (double)y, relTol, absTol);
    if constexpr (std::is_same_v<T,double>) return almostEqual(x, y, relTol, absTol);
    else return x == y;
  }, a);
}