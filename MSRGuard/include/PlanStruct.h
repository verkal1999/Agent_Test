/*
 * PlanStruct.h
 *
 * Defines the simple plan and operation structures used by the ExcH‑Agent.
 * These mirror the structures found in the original MSRGuard project but are
 * simplified here for demonstration.  Each operation corresponds to an action
 * the MSRGuard executor can perform (e.g., calling a method via OPC UA,
 * writing a boolean variable, pulsing a boolean, or ingesting a new skill).
 */
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

// Enumeration of supported operation types.  In a real implementation this
// should align with the definitions in the existing Plan.h file【830125908068710†L25-L71】.
enum class OpType {
    WriteBool,
    PulseBool,
    CallMethod,
    CallMonitoringActions,
    KGIngestion
};

// Forward declaration of a variant type used for arguments.  In the actual
// MSRGuard project this is a UAValue variant that can hold different OPC UA
// datatypes【594010340563552†L7-L14】.  Here we use a simple std::string map to
// represent arguments for demonstration.
using ArgMap = std::unordered_map<std::string, std::string>;

// Represents a single operation in a reaction plan.
struct Operation {
    OpType type;                     // Type of operation
    std::string nodeId;              // OPC UA node or resource identifier
    std::string methodName;          // Method name for CallMethod operations
    ArgMap args;                     // Arguments for the method call
    uint32_t timeoutMs = 0;          // Optional timeout in milliseconds
    std::string skillFile;           // Path to the generated skill file (KGIngestion)
};

// Represents a complete reaction plan.  A plan contains a correlation ID to
// trace the request, a resource ID for the affected device, a list of
// operations, and flags indicating whether the execution should abort or
// degrade【830125908068710†L25-L71】.
struct Plan {
    std::string correlationId;
    std::string resourceId;
    std::vector<Operation> operations;
    bool abort = false;
    bool degrade = false;
};
