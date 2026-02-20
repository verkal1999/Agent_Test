# ua_test_server

Lokaler OPC-UA-Testserver fuer Integrations- und Verbindungstests.

## Dateien und Verantwortung
- `ua_test_server_secure.cpp`: Implementierung eines sicheren OPC-UA-Testservers (inkl. Zertifikate).
- `CMakeLists.txt`: Builddefinition fuer den Testserver.
- `CMakePresets.json`: Vordefinierte Build-/Tool-Presets.

## Unterordner
- `certs/`: Server-Zertifikat und Schluessel fuer Security-Tests.
