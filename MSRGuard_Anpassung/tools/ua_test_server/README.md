# ua_test_server

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains a local secure OPC UA test server used during development and integration testing.

## What this folder contains

- `ua_test_server_secure.cpp`: secure OPC UA test server implementation
- `CMakeLists.txt`: build definition for the test server
- `CMakePresets.json`: build presets
- `certs/`: local server certificates for security-related tests

## Current status

This is a utility component for testing. It is not part of the main benchmark pipeline, but it helps validate the runtime integration used in the thesis.
