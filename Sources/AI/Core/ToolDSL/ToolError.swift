// Copyright © Anthony DePasquale

import Foundation

/// Errors that can occur during tool execution.
public enum ToolError: Error, LocalizedError {
  /// A required parameter was not provided.
  case missingRequiredParameter(String)

  /// A parameter has an invalid type.
  case invalidParameterType(parameter: String, expected: String, got: String)

  /// Parameter validation failed (e.g., value out of range).
  case validationFailed(parameter: String, reason: String)

  /// The requested tool was not found.
  case unknownTool(String)

  public var errorDescription: String? {
    switch self {
      case let .missingRequiredParameter(name):
        "Missing required parameter: \(name)"
      case let .invalidParameterType(param, expected, got):
        "Invalid type for '\(param)': expected \(expected), got \(got)"
      case let .validationFailed(param, reason):
        "Validation failed for '\(param)': \(reason)"
      case let .unknownTool(name):
        "Unknown tool: \(name)"
    }
  }
}
