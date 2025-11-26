#pragma once
#include "ICommandForce.h"
#include <memory>

struct WriteCsvParams; // fwd

class WriteCSVForce final : public ICommandForce {
public:
  int execute(const Plan& p) override; // 1 = OK, 0 = Fehler
private:
  static bool writeCsv(const WriteCsvParams& prm);
};