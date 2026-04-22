// Copyright © Anthony DePasquale

import Foundation

/// A type whose Swift value can be wrapped into `ToolResult.structuredContent`
/// as a JSON-schema-described value.
///
/// This is the element-level marker: anything that can appear inside an array,
/// an optional, a dictionary value, or as the wrapped value of a tool return
/// conforms.
///
/// The library conforms exactly these built-in types:
///
/// - Scalars: `Int`, `Double`, `Bool`, `String`, `Date`.
/// - Collections: `Array<Element>`, `Optional<Wrapped>`,
///   `Dictionary<String, Value>` — each conditionally, when the inner type
///   is itself `WrappableValue`.
/// - `@Schemable @StructuredOutput` structs — via a constrained extension.
///
/// Sized-int variants (`Int32`, `Int64`, `UInt`, …), `Float`, `Decimal`,
/// `URL`, and other common value types **do not conform**. The sealed set
/// keeps the JSON wire shape uncontroversial — JSON has one integer type
/// and one number type, and mapping Swift's richer numeric hierarchy onto
/// that is a policy question better resolved by the author than silently
/// by the library. Wrap in a `@StructuredOutput` struct whose Swift type
/// matches the intended wire shape.
///
/// Tool-output-level machinery lives on `PrimitiveToolOutput` (for types
/// that wrap under `"result"`) and `Dictionary: ToolOutput` (for the
/// unwrapped top-level-object path).
public protocol WrappableValue: Encodable, Sendable {
  /// JSON schema for the value itself, *not* wrapped in an object.
  ///
  /// - `Int` → `{"type": "integer"}`, not `{"type": "object", "properties": {"result": ...}}`.
  /// - `[Int]` → `{"type": "array", "items": {"type": "integer"}}`.
  /// - `Int?` → `{"type": ["integer", "null"]}`.
  /// - A `@StructuredOutput` struct → the struct's full `outputJSONSchema`.
  ///
  /// `PrimitiveToolOutput.toToolResult()` wraps this schema under `"result"`
  /// when producing the tool's `outputSchema`. Authors don't implement this
  /// directly for conforming built-in types — the library supplies it.
  static var valueSchema: Value { get }

  /// The value as a `Value`, for use as an array element, dictionary value,
  /// or to be wrapped under `"result"` in `structuredContent`.
  func asJSONValue() throws -> Value

  /// The value rendered as a single text block for `content[0].text`.
  ///
  /// Scalars stringify (`Int(42)` → `"42"`, `Bool(true)` → `"true"`);
  /// `String` passes through verbatim; compound values JSON-encode `self`
  /// directly. `Optional<Wrapped>.none` renders as the literal `"null"`;
  /// `.some(value)` JSON-encodes the unwrapped value (`Int?.some(42)` →
  /// `"42"`, `String?.some("hello")` → `"\"hello\""`).
  func asDisplayText() throws -> String
}

// MARK: - Primitive conformances

extension Int: WrappableValue {
  public static var valueSchema: Value {
    .object(["type": .string("integer")])
  }

  public func asJSONValue() throws -> Value {
    .int(self)
  }

  public func asDisplayText() throws -> String {
    String(self)
  }
}

extension Double: WrappableValue {
  public static var valueSchema: Value {
    .object(["type": .string("number")])
  }

  public func asJSONValue() throws -> Value {
    if isNaN || isInfinite {
      throw AIError.invalidRequest(
        message: "Double value \(self) is not representable in JSON. NaN, Infinity, and -Infinity have no JSON form — return a sentinel value or a @StructuredOutput struct that expresses the case explicitly.",
      )
    }
    return .double(self)
  }

  public func asDisplayText() throws -> String {
    String(self)
  }
}

extension Bool: WrappableValue {
  public static var valueSchema: Value {
    .object(["type": .string("boolean")])
  }

  public func asJSONValue() throws -> Value {
    .bool(self)
  }

  public func asDisplayText() throws -> String {
    String(self)
  }
}

extension String: WrappableValue {
  public static var valueSchema: Value {
    .object(["type": .string("string")])
  }

  public func asJSONValue() throws -> Value {
    .string(self)
  }

  public func asDisplayText() throws -> String {
    self
  }
}

extension Date: WrappableValue {
  public static var valueSchema: Value {
    .object([
      "type": .string("string"),
      "format": .string("date-time"),
    ])
  }

  public func asJSONValue() throws -> Value {
    let data = try AIEncoding.defaultEncoder().encode(self)
    return try JSONDecoder().decode(Value.self, from: data)
  }

  public func asDisplayText() throws -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.string(from: self)
  }
}

// MARK: - Compound conformances

extension Array: WrappableValue where Element: WrappableValue {
  public static var valueSchema: Value {
    .object([
      "type": .string("array"),
      "items": Element.valueSchema,
    ])
  }

  public func asJSONValue() throws -> Value {
    try .array(map { try $0.asJSONValue() })
  }

  public func asDisplayText() throws -> String {
    try prettyPrintedJSON(self)
  }
}

extension Optional: WrappableValue where Wrapped: WrappableValue {
  public static var valueSchema: Value {
    Value.promoteToNullable(Wrapped.valueSchema)
  }

  public func asJSONValue() throws -> Value {
    switch self {
      case .none:
        .null
      case let .some(value):
        try value.asJSONValue()
    }
  }

  public func asDisplayText() throws -> String {
    switch self {
      case .none:
        "null"
      case let .some(value):
        try prettyPrintedJSON(value)
    }
  }
}

// MARK: - StructuredOutput bridge

public extension WrappableValue where Self: StructuredOutput {
  static var valueSchema: Value {
    outputJSONSchema
  }

  func asJSONValue() throws -> Value {
    let data = try Self.encoder.encode(self)
    return try JSONDecoder().decode(Value.self, from: data)
  }

  func asDisplayText() throws -> String {
    try prettyPrintedJSON(self)
  }
}

// MARK: - Shared helpers

extension Value {
  /// Promotes a value schema to allow `null` in addition to whatever type(s)
  /// it already declares. Falls back to `anyOf` for composition-only schemas.
  static func promoteToNullable(_ schema: Value) -> Value {
    guard case var .object(fields) = schema else {
      return anyOfNullable(schema)
    }
    guard let existingType = fields["type"] else {
      return anyOfNullable(schema)
    }
    switch existingType {
      case let .string(name):
        if name == "null" {
          return schema
        }
        fields["type"] = .array([.string(name), .string("null")])
        return .object(fields)
      case let .array(elements):
        if elements.contains(.string("null")) {
          return schema
        }
        fields["type"] = .array(elements + [.string("null")])
        return .object(fields)
      default:
        return anyOfNullable(schema)
    }
  }

  private static func anyOfNullable(_ schema: Value) -> Value {
    .object([
      "anyOf": .array([schema, .object(["type": .string("null")])]),
    ])
  }
}

/// Pretty-prints an `Encodable` value through `AIEncoding.defaultEncoder()`.
/// Used by compound `WrappableValue` conformers (`Array`, `Optional`,
/// `Dictionary`, `@StructuredOutput` structs).
func prettyPrintedJSON(_ value: some Encodable) throws -> String {
  let encoder = AIEncoding.defaultEncoder()
  encoder.outputFormatting.insert(.prettyPrinted)
  let data = try encoder.encode(value)
  guard let text = String(data: data, encoding: .utf8) else {
    throw AIError.invalidRequest(message: "Failed to render encoded value as UTF-8 text")
  }
  return text
}
