// Copyright © Anthony DePasquale

import Foundation

/// Types that can be used as tool parameters.
///
/// This protocol provides the information needed to:
/// 1. Generate JSON Schema for the parameter type
/// 2. Parse `Value` into the Swift type
///
/// Built-in conformances include:
/// - `String`, `Int`, `Double`, `Bool` - Basic types
/// - `Date` - Parsed as ISO 8601 strings
/// - `Data` - Base64-encoded strings
/// - `Optional<T>` where T: ParameterValue
/// - `Array<T>` where T: ParameterValue
/// - `Dictionary<String, T>` where T: ParameterValue
///
/// ## Creating Custom Types
///
/// To use a custom type as a tool parameter, conform it to `ParameterValue`:
///
/// ```swift
/// struct Money: ParameterValue {
///     let amount: Double
///     let currency: String
///
///     static var jsonSchemaType: String { "object" }
///
///     static var jsonSchemaProperties: [String: Value] {
///         [
///             "properties": .object([
///                 "amount": .object(["type": .string("number")]),
///                 "currency": .object(["type": .string("string")])
///             ]),
///             "required": .array([.string("amount"), .string("currency")])
///         ]
///     }
///
///     static var placeholderValue: Money {
///         Money(amount: 0, currency: "USD")
///     }
///
///     init?(parameterValue value: Value) {
///         guard case .object(let obj) = value,
///               let amountVal = obj["amount"], case .double(let amount) = amountVal,
///               let currencyVal = obj["currency"], case .string(let currency) = currencyVal
///         else { return nil }
///         self.amount = amount
///         self.currency = currency
///     }
/// }
/// ```
///
/// For string enums, use the ``ToolEnum`` protocol instead, which provides
/// automatic conformance for `RawRepresentable` types with `String` raw values.
public protocol ParameterValue: Sendable {
  /// The JSON Schema type name (e.g., "string", "integer", "number", "boolean").
  static var jsonSchemaType: String { get }

  /// Additional schema properties (e.g., format, enum values, items).
  static var jsonSchemaProperties: [String: Value] { get }

  /// Parse from a `Value`.
  /// - Parameter value: The `Value` to parse.
  /// - Returns: The parsed value, or nil if parsing fails.
  init?(parameterValue value: Value)

  /// A placeholder value used during tool initialization.
  /// This value is replaced during parsing from arguments.
  static var placeholderValue: Self { get }
}

public extension ParameterValue {
  /// Default: no additional properties.
  static var jsonSchemaProperties: [String: Value] {
    [:]
  }
}

// MARK: - String Conformance

extension String: ParameterValue {
  public static var jsonSchemaType: String {
    "string"
  }

  public static var placeholderValue: String {
    ""
  }

  /// Parse a string from a Value.
  /// Uses strict mode: only `.string` values are accepted.
  public init?(parameterValue value: Value) {
    self.init(value, strict: true)
  }
}

// MARK: - Int Conformance

extension Int: ParameterValue {
  public static var jsonSchemaType: String {
    "integer"
  }

  public static var placeholderValue: Int {
    0
  }

  /// Parse an integer from a Value.
  /// Uses strict mode: only `.int` values are accepted.
  public init?(parameterValue value: Value) {
    self.init(value, strict: true)
  }
}

// MARK: - Double Conformance

extension Double: ParameterValue {
  public static var jsonSchemaType: String {
    "number"
  }

  public static var placeholderValue: Double {
    0
  }

  /// Parse a double from a Value.
  /// Uses strict mode: only `.double` and `.int` values are accepted.
  public init?(parameterValue value: Value) {
    self.init(value, strict: true)
  }
}

// MARK: - Bool Conformance

extension Bool: ParameterValue {
  public static var jsonSchemaType: String {
    "boolean"
  }

  public static var placeholderValue: Bool {
    false
  }

  /// Parse a boolean from a Value.
  /// Uses strict mode: only `.bool` values are accepted.
  public init?(parameterValue value: Value) {
    self.init(value, strict: true)
  }
}

// MARK: - Date Conformance

extension Date: ParameterValue {
  public static var jsonSchemaType: String {
    "string"
  }

  public static var placeholderValue: Date {
    Date(timeIntervalSince1970: 0)
  }

  public static var jsonSchemaProperties: [String: Value] {
    ["format": .string("date-time")]
  }

  /// Parse a Date from a Value containing an ISO 8601 string.
  public init?(parameterValue value: Value) {
    guard case let .string(str) = value else { return nil }

    // Try ISO 8601 with fractional seconds first
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = formatter.date(from: str) {
      self = date
      return
    }

    // Fall back to without fractional seconds
    formatter.formatOptions = [.withInternetDateTime]
    if let date = formatter.date(from: str) {
      self = date
      return
    }

    return nil
  }
}

// MARK: - Data Conformance

extension Data: ParameterValue {
  public static var jsonSchemaType: String {
    "string"
  }

  public static var placeholderValue: Data {
    Data()
  }

  public static var jsonSchemaProperties: [String: Value] {
    ["contentEncoding": .string("base64")]
  }

  /// Parse Data from a Value containing a base64-encoded string.
  public init?(parameterValue value: Value) {
    guard case let .string(str) = value,
          let data = Data(base64Encoded: str)
    else {
      return nil
    }
    self = data
  }
}

// MARK: - Optional Conformance

extension Optional: ParameterValue where Wrapped: ParameterValue {
  public static var jsonSchemaType: String {
    Wrapped.jsonSchemaType
  }

  public static var placeholderValue: Wrapped? {
    nil
  }

  public static var jsonSchemaProperties: [String: Value] {
    Wrapped.jsonSchemaProperties
  }

  /// Parse an optional value from a Value.
  /// Returns nil (success with nil value) for Value.null, otherwise delegates to wrapped type.
  public init?(parameterValue value: Value) {
    if case .null = value {
      self = .none
      return
    }
    if let wrapped = Wrapped(parameterValue: value) {
      self = .some(wrapped)
    } else {
      return nil
    }
  }
}

// MARK: - Array Conformance

extension Array: ParameterValue where Element: ParameterValue {
  public static var jsonSchemaType: String {
    "array"
  }

  public static var placeholderValue: [Element] {
    []
  }

  public static var jsonSchemaProperties: [String: Value] {
    var props: [String: Value] = [
      "items": .object([
        "type": .string(Element.jsonSchemaType),
      ]),
    ]
    // Merge element's additional properties into items
    let elementProps = Element.jsonSchemaProperties
    if !elementProps.isEmpty {
      var itemsObj: [String: Value] = ["type": .string(Element.jsonSchemaType)]
      for (key, val) in elementProps {
        itemsObj[key] = val
      }
      props["items"] = .object(itemsObj)
    }
    return props
  }

  /// Parse an array from a Value.
  public init?(parameterValue value: Value) {
    guard case let .array(arr) = value else { return nil }

    var result: [Element] = []
    for item in arr {
      guard let element = Element(parameterValue: item) else {
        return nil
      }
      result.append(element)
    }
    self = result
  }
}

// MARK: - Dictionary Conformance

extension Dictionary: ParameterValue where Key == String, Value: ParameterValue {
  public static var jsonSchemaType: String {
    "object"
  }

  public static var placeholderValue: [String: Value] {
    [:]
  }

  public static var jsonSchemaProperties: [String: AI.Value] {
    var additionalProps: [String: AI.Value] = ["type": .string(Value.jsonSchemaType)]
    // Merge value type's additional properties
    let valueProps = Value.jsonSchemaProperties
    for (key, val) in valueProps {
      additionalProps[key] = val
    }
    return ["additionalProperties": .object(additionalProps)]
  }

  /// Parse a dictionary from a Value.
  public init?(parameterValue value: AI.Value) {
    guard case let .object(dict) = value else { return nil }

    var result: [String: Value] = [:]
    for (key, val) in dict {
      guard let parsed = Value(parameterValue: val) else {
        return nil
      }
      result[key] = parsed
    }
    self = result
  }
}
