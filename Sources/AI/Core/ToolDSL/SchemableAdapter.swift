// Copyright © Anthony DePasquale

import Foundation
import JSONSchema
import JSONSchemaBuilder

/// Bridges `JSONSchemaComponent` output (produced by `@Schemable`) into the
/// internal `[String: Value]` representation used by the tool DSL and
/// strict-mode normalizer, and parses JSON arguments into Swift values.
///
/// These entry points are called from `@Tool` macro-emitted code at the
/// user's call site, so they're exposed via SPI.
@_spi(ToolMacroSupport) public enum SchemableAdapter {
  /// Converts a `JSONSchemaComponent` into `[String: Value]` by encoding the
  /// component's `Schema` definition to JSON and decoding it into `Value`.
  ///
  /// Throws if the component produces a non-object schema (JSON Schema allows
  /// boolean schemas, but tools always model parameters as objects) or if the
  /// Codable roundtrip fails.
  public static func valueDictionary(
    from component: some JSONSchemaComponent,
  ) throws -> [String: Value] {
    let schema = component.definition()
    let value = try Value(schema)
    guard case let .object(dictionary) = value else {
      throw AIError.invalidRequest(
        message: "@Schemable component produced a non-object schema: \(value)",
      )
    }
    return dictionary
  }

  /// Builds the `[String: Value]` schema used for `StructuredOutput` result
  /// types. Calls `valueDictionary(from:)` and then post-processes the result
  /// so every property appears in `required` — recursively, at every nested
  /// object schema inside the tree.
  ///
  /// JSONSchemaBuilder's default for optional stored properties produces
  /// `type: ["T", "null"]` (nullable union) and leaves the property out of
  /// `required`. For structured tool output, the wire contract says every key
  /// is always present (optional Swift values encode as explicit `null`), and
  /// code-mode consumers generate typed interfaces from this schema —
  /// `required + nullable union` produces `field: T | null` (always defined),
  /// whereas `optional + nullable` produces `field?: T | null` (caller has to
  /// handle `undefined`). Mirrors swift-mcp's `structuredOutputSchemaDictionary`.
  public static func structuredOutputSchemaDictionary(
    from component: some JSONSchemaComponent,
  ) throws -> [String: Value] {
    let dictionary = try valueDictionary(from: component)
    guard case let .object(rewritten) = promoteRequired(.object(dictionary)) else {
      return dictionary
    }
    return rewritten
  }

  /// Recursively promotes every nested object schema's `required` list to
  /// include all of its declared properties. Leaves non-object schemas
  /// untouched.
  static func promoteRequired(_ value: Value) -> Value {
    switch value {
      case var .object(dict):
        if case let .object(properties)? = dict["properties"] {
          let rewrittenProperties = properties.mapValues(promoteRequired)
          dict["properties"] = .object(rewrittenProperties)
          dict["required"] = .array(rewrittenProperties.keys.sorted().map(Value.string))
        }
        if let items = dict["items"] { dict["items"] = promoteRequired(items) }
        if case let .array(prefix)? = dict["prefixItems"] {
          dict["prefixItems"] = .array(prefix.map(promoteRequired))
        }
        if let addl = dict["additionalProperties"] {
          dict["additionalProperties"] = promoteRequired(addl)
        }
        if case let .object(patterns)? = dict["patternProperties"] {
          dict["patternProperties"] = .object(patterns.mapValues(promoteRequired))
        }
        for key in ["oneOf", "anyOf", "allOf"] {
          if case let .array(variants)? = dict[key] {
            dict[key] = .array(variants.map(promoteRequired))
          }
        }
        for key in ["$defs", "definitions"] {
          if case let .object(defs)? = dict[key] {
            dict[key] = .object(defs.mapValues(promoteRequired))
          }
        }
        return .object(dict)
      case let .array(items):
        return .array(items.map(promoteRequired))
      default:
        return value
    }
  }

  /// Parses a `Value` into the component's `Output` type, throwing a `ToolError`
  /// with a human-readable message on failure. Path information is preserved
  /// in the error so an LLM agent retrying the tool call gets enough detail
  /// to fix the input.
  public static func parse<Component: JSONSchemaComponent>(
    _ component: Component,
    from value: Value,
    parameterName: String,
  ) throws -> Component.Output {
    let jsonValue = value.toJSONValue()
    switch component.parse(jsonValue) {
      case let .valid(output):
        return output
      case let .invalid(issues):
        let detail = issues.map(\.description).joined(separator: "; ")
        throw ToolDispatchError.invalidParameterType(
          parameter: parameterName,
          expected: String(describing: Component.Output.self),
          got: "\(value) — \(detail)",
        )
    }
  }
}
