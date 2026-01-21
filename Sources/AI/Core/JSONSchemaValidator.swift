// Copyright © Anthony DePasquale

import Foundation
import JSONSchema

#if canImport(os)
import os
#endif

/// A protocol for validating JSON values against JSON Schema.
///
/// This protocol abstracts JSON Schema validation to allow different implementations
/// to be used. The default implementation uses the swift-json-schema library.
public protocol JSONSchemaValidator: Sendable {
  /// Validates an instance value against a JSON Schema.
  ///
  /// - Parameters:
  ///   - instance: The value to validate.
  ///   - schema: The JSON Schema to validate against.
  /// - Throws: An error if validation fails.
  func validate(_ instance: Value, against schema: Value) throws
}

/// Errors thrown during JSON Schema validation.
public enum JSONSchemaValidationError: Error, LocalizedError {
  /// The schema itself is invalid.
  case invalidSchema(String)
  /// The instance does not conform to the schema.
  case validationFailed(String)

  public var errorDescription: String? {
    switch self {
      case let .invalidSchema(message):
        "Invalid JSON Schema: \(message)"
      case let .validationFailed(message):
        message
    }
  }
}

/// Default JSON Schema validator using swift-json-schema library.
///
/// This validator supports JSON Schema draft-2020-12 and provides comprehensive
/// validation including type checking, format validation, and constraint enforcement.
///
/// Compiled schemas are cached for performance when validating multiple instances
/// against the same schema (e.g., repeated tool calls).
public final class DefaultJSONSchemaValidator: JSONSchemaValidator, @unchecked Sendable {
  #if canImport(os)
  private let cache = OSAllocatedUnfairLock(initialState: [Value: Schema]())
  #else
  private let lock = NSLock()
  private var cacheStorage: [Value: Schema] = [:]
  #endif

  public init() {}

  public func validate(_ instance: Value, against schema: Value) throws {
    let jsonInstance = instance.toJSONValue()

    // Check cache first, compile if not found
    let validator = try getOrCompileSchema(schema)

    let result = validator.validate(jsonInstance)

    if !result.isValid {
      let message = formatValidationErrors(result.errors)
      throw JSONSchemaValidationError.validationFailed(message)
    }
  }

  private func getOrCompileSchema(_ schema: Value) throws -> Schema {
    #if canImport(os)
    return try cache.withLock { cache -> Schema in
      if let cached = cache[schema] {
        return cached
      }
      let compiled = try compileSchema(schema)
      cache[schema] = compiled
      return compiled
    }
    #else
    lock.lock()
    defer { lock.unlock() }
    if let cached = cacheStorage[schema] {
      return cached
    }
    let compiled = try compileSchema(schema)
    cacheStorage[schema] = compiled
    return compiled
    #endif
  }

  private func compileSchema(_ schema: Value) throws -> Schema {
    let jsonSchema = schema.toJSONValue()
    do {
      return try Schema(
        rawSchema: jsonSchema,
        context: .init(dialect: .draft2020_12)
      )
    } catch {
      throw JSONSchemaValidationError.invalidSchema("\(error)")
    }
  }

  /// Formats validation errors into a human-readable message.
  private func formatValidationErrors(_ errors: [ValidationError]?) -> String {
    guard let errors, !errors.isEmpty else {
      return "Validation failed"
    }

    // Collect all error messages, including nested ones
    var messages: [String] = []
    collectErrorMessages(errors, into: &messages)

    if messages.count == 1 {
      return messages[0]
    }

    return messages.joined(separator: "; ")
  }

  /// Recursively collects error messages from validation errors.
  private func collectErrorMessages(_ errors: [ValidationError], into messages: inout [String]) {
    for error in errors {
      let path = error.instanceLocation.description
      let keyword = error.keyword

      // Build a descriptive message
      let message: String = if error.message.isEmpty {
        if path.isEmpty || path == "/" {
          "validation failed for '\(keyword)'"
        } else {
          "validation failed for '\(keyword)' at \(path)"
        }
      } else {
        if path.isEmpty || path == "/" {
          error.message
        } else {
          "\(error.message) at \(path)"
        }
      }
      messages.append(message)

      // Collect nested errors
      if let nestedErrors = error.errors {
        collectErrorMessages(nestedErrors, into: &messages)
      }
    }
  }
}
