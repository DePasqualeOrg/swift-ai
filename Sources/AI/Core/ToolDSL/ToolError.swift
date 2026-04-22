// Copyright © Anthony DePasquale

import Foundation

/// Framework-side errors raised when the dispatcher can't even invoke a tool —
/// missing required parameter, type mismatch, unknown tool, etc.
///
/// Distinct from the author-thrown `ToolError` protocol below, which carries
/// rich content from inside the tool's `perform()` body. The naming split
/// mirrors swift-mcp.
public enum ToolDispatchError: Error, LocalizedError {
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

// MARK: - Author-thrown `ToolError` protocol

/// Author-thrown errors that carry rich content (multi-block text + binary,
/// or structured JSON payloads) from inside a tool's `perform()` body.
///
/// Collapses swift-ai's two error paths — `throw` (loses rich content) and
/// `ToolResult.error(...)` (bypasses throw entirely) — into one `throw`-based
/// flow. The dispatcher in `Tools.call()` catches conformers and emits a
/// `ToolResult` with `isError: true` carrying both `content` and
/// `structuredContent`.
///
/// Example:
/// ```swift
/// struct WeatherUnavailable: ToolError {
///     let city: String
///     var content: [ToolResult.Content] {
///         [.text("No data for \(city)")]
///     }
/// }
/// ```
public protocol ToolError: LocalizedError {
  /// Content blocks the model sees on the failure path. Mirrors success-path
  /// `ToolOutput` returns — text, image, audio, file, resource, JSON.
  var content: [ToolResult.Content] { get }

  /// Optional structured-channel payload, parallel to `ToolResult.structuredContent`.
  /// `nil` by default — text/blob-only errors don't have to populate it.
  /// Rich errors that want to thread typed payloads through Gemini's
  /// `functionResponse.response` and MCP's `structuredContent` override this.
  var structuredContent: Value? { get }
}

public extension ToolError {
  var structuredContent: Value? {
    nil
  }

  /// Default joins `.text` blocks from `content` so plain `Error.localizedDescription`
  /// callers see something useful. Falls back to the type name when no `.text`
  /// blocks exist (rich-only payloads).
  var errorDescription: String? {
    let texts = content.compactMap { block -> String? in
      if case let .text(text) = block {
        return text
      }
      return nil
    }
    if texts.isEmpty {
      return "\(Self.self)"
    }
    return texts.joined(separator: "\n")
  }
}
