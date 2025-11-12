// PythonRuntime (globale pybind11-Laufzeitwache)
// - Hält einen statischen std::unique_ptr<pybind11::scoped_interpreter>, damit der
//   eingebettete Python-Interpreter (PythonWorker, KG-Brücke usw.) pro Prozess nur
//   einmal gestartet und bis Programmende am Leben gehalten wird.
// - Details siehe MPA_Draft: PythonWorker / eingebetteter Interpreter.
#include "PythonRuntime.h"
std::unique_ptr<pybind11::scoped_interpreter> PythonRuntime::guard_;