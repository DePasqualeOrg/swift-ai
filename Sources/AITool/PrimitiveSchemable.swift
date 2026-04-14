// Copyright © Anthony DePasquale

import Foundation
import JSONSchema
import JSONSchemaBuilder

// Retroactive Schemable conformances for primitives used as tool parameters.
// These are the building blocks the `@Tool` macro emits for typed parameters;
// user-defined tool parameter types are expected to adopt `@Schemable`
// themselves (directly or via the `@Schemable` macro).

extension String: @retroactive Schemable {
  public static var schema: JSONString {
    JSONString()
  }
}

extension Int: @retroactive Schemable {
  public static var schema: JSONInteger {
    JSONInteger()
  }
}

extension Double: @retroactive Schemable {
  public static var schema: JSONNumber {
    JSONNumber()
  }
}

extension Bool: @retroactive Schemable {
  public static var schema: JSONBoolean {
    JSONBoolean()
  }
}

extension Array: @retroactive Schemable where Element: Schemable {
  public static var schema: JSONArray<Element.Schema> {
    JSONArray { Element.schema }
  }
}

extension Optional: @retroactive Schemable where Wrapped: Schemable {
  public static var schema: JSONComponents.AnySchemaComponent<Wrapped.Schema.Output?> {
    Wrapped.schema.orNull(style: .type)
  }
}

// MARK: - Date / Data

extension Date: @retroactive Schemable {
  public static var schema: JSONComponents.CompactMap<JSONString, Date> {
    JSONString()
      .format("date-time")
      .compactMap { string -> Date? in
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: string) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: string)
      }
  }
}

extension Data: @retroactive Schemable {
  public static var schema: JSONComponents.CompactMap<JSONString, Data> {
    JSONString()
      .contentEncoding("base64")
      .compactMap { Data(base64Encoded: $0) }
  }
}

// MARK: - Dictionary

/// A JSON schema component representing a dictionary with `String` keys and
/// a uniform `Value` component for every value.
///
/// Lives here rather than upstream because JSONSchemaBuilder's
/// `additionalProperties` modifier produces a `(Void, T)` tuple output rather
/// than `[String: T]`; we need the dictionary shape for `Dictionary: Schemable`.
public struct JSONDictionary<ValueComponent: JSONSchemaComponent>: JSONSchemaComponent {
  public var schemaValue: SchemaValue
  private let valueComponent: ValueComponent

  public init(valueComponent: ValueComponent) {
    self.valueComponent = valueComponent
    var schemaValue: SchemaValue = .object([:])
    schemaValue["type"] = .string(JSONType.object.rawValue)
    schemaValue["additionalProperties"] = valueComponent.schemaValue.value
    self.schemaValue = schemaValue
  }

  public func parse(_ value: JSONValue) -> Parsed<[String: ValueComponent.Output], ParseIssue> {
    guard case let .object(dictionary) = value else {
      return .invalid([.typeMismatch(expected: .object, actual: value)])
    }
    var result: [String: ValueComponent.Output] = [:]
    var errors: [ParseIssue] = []
    for (key, jsonValue) in dictionary {
      switch valueComponent.parse(jsonValue) {
        case let .valid(parsed):
          result[key] = parsed
        case let .invalid(nested):
          errors.append(contentsOf: nested)
      }
    }
    guard errors.isEmpty else { return .invalid(errors) }
    return .valid(result)
  }
}

extension Dictionary: @retroactive Schemable where Key == String, Value: Schemable {
  public static var schema: JSONDictionary<Value.Schema> {
    JSONDictionary(valueComponent: Value.schema)
  }
}
