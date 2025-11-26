// WriteCsvParams – CSV-Parameterbündel für TimeBlogger & WriteCSVForce
//
//   - CsvRow beschreibt eine aufgezeichnete Zeitmessung pro Ereignis:
//       * corrId     : CorrelationId der Reaktionskette
//       * eventType  : Name des Events (z. B. "evD2", "evSRDone", …)
//       * durationMs : Zeitdifferenz seit dem letzten markierten Event
//       * durSumMs   : aufsummierte Gesamtzeit seit Beginn der Korrelation
//   - WriteCsvParams bündelt alle Zeilen und den Zielpfad für die CSV-Datei:
//       * outFile    : z. B. "logs/time/timeblog_DefaultParams.csv"
//       * rows       : Sequenz von CsvRow-Einträgen
//       * withHeader : steuert, ob ein Header geschrieben werden soll
//
// TimeBlogger:
//   - sammelt CsvRow in Timeline::csvRows,
//   - erzeugt beim Abschluss einer Korrelation ein std::shared_ptr<WriteCsvParams>,
//   - setzt prm->outFile und prm->rows,
//   - verpackt das Objekt in Operation::attach (OpType::WriteCSV) und
//     lässt die WriteCSVForce den eigentlichen Dateischreibvorgang ausführen.
#pragma once
#include <string>
#include <vector>
struct CsvRow {
  std::string corrId;
  std::string eventType;
  long long   durationMs{0};
  long long   durSumMs{0};
};
struct WriteCsvParams {
  std::string outFile;           
  std::vector<CsvRow> rows;      
  bool withHeader{true};         
};
