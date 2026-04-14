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
        throw ToolError.invalidParameterType(
          parameter: parameterName,
          expected: String(describing: Component.Output.self),
          got: "\(value) — \(detail)",
        )
    }
  }
}
