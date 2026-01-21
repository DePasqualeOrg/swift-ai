// Copyright © Anthony DePasquale

import AI
import MCP

// MARK: - AI.Value ↔ MCP.Value Conversions

public extension AI.Value {
  /// Creates an AI Value from an MCP Value.
  ///
  /// Note: MCP's `.data` case is converted to a base64-encoded string,
  /// since AI.Value doesn't have a native data representation.
  init(_ mcpValue: MCP.Value) {
    switch mcpValue {
      case .null:
        self = .null
      case let .bool(b):
        self = .bool(b)
      case let .int(i):
        self = .int(i)
      case let .double(d):
        self = .double(d)
      case let .string(s):
        self = .string(s)
      case let .data(mimeType, data):
        // Encode as data URL string
        let base64 = data.base64EncodedString()
        let dataURL = if let mimeType {
          "data:\(mimeType);base64,\(base64)"
        } else {
          "data:application/octet-stream;base64,\(base64)"
        }
        self = .string(dataURL)
      case let .array(arr):
        self = .array(arr.map { AI.Value($0) })
      case let .object(obj):
        self = .object(obj.mapValues { AI.Value($0) })
    }
  }

  /// Converts this Value to an MCP Value.
  var mcpValue: MCP.Value {
    switch self {
      case .null:
        .null
      case let .bool(b):
        .bool(b)
      case let .int(i):
        .int(i)
      case let .double(d):
        .double(d)
      case let .string(s):
        .string(s)
      case let .array(arr):
        .array(arr.map { $0.mcpValue })
      case let .object(obj):
        .object(obj.mapValues { $0.mcpValue })
    }
  }
}

public extension MCP.Value {
  /// Creates an MCP Value from an AI Value.
  init(_ value: AI.Value) {
    self = value.mcpValue
  }

  /// Converts this Value to an AI Value.
  var aiValue: AI.Value {
    AI.Value(self)
  }
}

// MARK: - Dictionary Conversions

public extension [String: AI.Value] {
  /// Converts a dictionary of AI Values to MCP Values.
  var mcpValues: [String: MCP.Value] {
    mapValues { $0.mcpValue }
  }
}

public extension [String: MCP.Value] {
  /// Converts a dictionary of MCP Values to AI Values.
  var aiValues: [String: AI.Value] {
    mapValues { $0.aiValue }
  }
}
