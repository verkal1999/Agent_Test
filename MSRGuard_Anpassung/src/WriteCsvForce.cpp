// WriteCSVForce (ICommandForce-Implementierung für CSV-Export)
// - Führt execute(const Plan&) für genau eine Operation mit WriteCsvParams aus.
// - Die Parameter (Dateipfad, Zeilen, Header-Flag) liegen typisiert in Operation::attach.
// - Öffnet/erzeugt die Datei, legt fehlende Verzeichnisse an und schreibt die Zeilen
//   CSV-konform (csvEscape: doppelte Quotes, CRLF gem. RFC 4180).
// - Zugriff ist über einen Prozess-weiten Mutex serialisiert, damit TimeBlogger & Co.
//   threadsicher in dieselbe Datei loggen können.

#include "WriteCSVForce.h"
#include "WriteCsvParams.h"
#include <any>
#include <fstream>
#include <filesystem>
#include <mutex>

namespace {
  std::string csvEscape(const std::string& s) {
    bool need = s.find_first_of(",\"\r\n") != std::string::npos;
    if (!need) return s;
    std::string out; out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) { if (c=='"') out.push_back('"'); out.push_back(c); }
    out.push_back('"');
    return out;
  }
  std::mutex& csv_mutex() { static std::mutex m; return m; } // prozessweiter Schutz
}

int WriteCSVForce::execute(const Plan& p) {
  if (p.ops.empty()) return 0;
  const Operation& op = p.ops.front();

  std::shared_ptr<WriteCsvParams> prm;
  try {
    prm = std::any_cast<std::shared_ptr<WriteCsvParams>>(op.attach);
  } catch (...) { return 0; }

  // Fester Pfad kommt schon aus TimeBlogger -> prm->outFile
  if (prm->outFile.empty()) return 0;

  namespace fs = std::filesystem;
  std::error_code ec;
  if (auto parent = fs::path(prm->outFile).parent_path(); !parent.empty())
    fs::create_directories(parent, ec);

  std::lock_guard<std::mutex> lk(csv_mutex());

  bool newOrEmpty = !fs::exists(prm->outFile)
                 || (fs::is_regular_file(prm->outFile) && fs::file_size(prm->outFile, ec) == 0);

  std::ofstream ofs(prm->outFile, std::ios::out | std::ios::app); // APPEND!
  if (!ofs.is_open()) return 0;

  if (newOrEmpty) {
    ofs << "corrrelID,EventType,duration,DurSum\r\n"; // Header nur einmal (CRLF lt. RFC 4180)
  }
  for (const auto& r : prm->rows) {
    ofs  << csvEscape(r.corrId)    << ','
         << csvEscape(r.eventType) << ','
         << r.durationMs           << ','
         << r.durSumMs             << "\r\n";
  }
  ofs.flush(); // Buffer → OS
  return ofs.good() ? 1 : 0;
}