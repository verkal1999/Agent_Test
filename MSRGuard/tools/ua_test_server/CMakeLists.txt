cmake_minimum_required(VERSION 3.20)
project(ua_test_server_secure C CXX)

# C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC Runtime konsistent halten
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# open62541 Optionen (wie zuvor)
set(UA_ENABLE_SUBSCRIPTIONS ON CACHE BOOL "" FORCE)
set(UA_ENABLE_SUBSCRIPTION_EVENTS ON CACHE BOOL "" FORCE)
set(UA_ENABLE_ENCRYPTION OPENSSL CACHE STRING "" FORCE)
set(UA_ENABLE_OPENSSL    ON      CACHE BOOL   "" FORCE)
set(UA_ENABLE_MBEDTLS    OFF     CACHE BOOL   "" FORCE)
find_package(OpenSSL REQUIRED)

# Pfad zum Repo-Root ermitteln (dieses CMake liegt in tools/ua_test_server)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

# open62541 als Subdir (wie zuvor, nur relativ zu diesem CMake)
add_subdirectory("${ROOT_DIR}/open62541" "${CMAKE_BINARY_DIR}/open62541")

# Executable (Server)
add_executable(ua_test_server_secure
  ${CMAKE_CURRENT_LIST_DIR}/ua_test_server_secure.cpp
)

# Includes für open62541
target_include_directories(ua_test_server_secure PRIVATE
  "${ROOT_DIR}/open62541/include"
  "${ROOT_DIR}/open62541/plugins/include"
  "${CMAKE_BINARY_DIR}/open62541/src_generated"
)

# Link mit open62541 + OpenSSL
target_link_libraries(ua_test_server_secure PRIVATE
  open62541
  OpenSSL::SSL OpenSSL::Crypto
)

# Binär-Output in build-server/bin/... (praktisch zum Starten)
set_target_properties(ua_test_server_secure PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Server-Zertifikate neben die EXE spiegeln
add_custom_command(TARGET ua_test_server_secure POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:ua_test_server_secure>/certs"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_LIST_DIR}/certs/server_cert.der"
          "$<TARGET_FILE_DIR:ua_test_server_secure>/certs/server_cert.der"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_LIST_DIR}/certs/server_key.der"
          "$<TARGET_FILE_DIR:ua_test_server_secure>/certs/server_key.der"
)
