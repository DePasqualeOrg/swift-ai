// Copyright © Anthony DePasquale

import Foundation
import JSONSchema

/// A type-safe representation of JSON values.
///
/// `Value` provides a Swift-native way to work with JSON data, with separate cases
/// for integers and doubles (matching JSON Schema's `integer` vs `number` distinction),
/// full literal conformances for ergonomic construction, and accessor properties for
/// safe value extraction.
///
/// ## Creating Values
///
/// Use literal syntax for concise value construction:
/// ```swift
/// let schema: [String: Value] = [
///     "type": "object",
///     "properties": [
///         "name": ["type": "string"],
///         "age": ["type": "integer", "minimum": 0]
///     ],
///     "required": ["name"]
/// ]
/// ```
///
/// ## Extracting Values
///
/// Use accessor properties to safely extract typed values:
/// ```swift
/// if let name = value.objectValue?["name"]?.stringValue {
///     print("Name: \(name)")
/// }
/// ```
///
/// For flexible parsing with type coercion, use the standard library extensions:
/// ```swift
/// // Strict mode (default): only exact type matches
/// let count = Int(value)
///
/// // Non-strict mode: allows coercion from compatible types
/// let count = Int(value, strict: false)  // Accepts int, double (if whole), or string
/// ```
public enum Value: Hashable, Sendable {
  case null
  case bool(Bool)
  case int(Int)
  case double(Double)
  case string(String)
  case array([Value])
  case object([String: Value])

  // MARK: - Accessors

  /// Returns whether the value is `null`.
  public var isNull: Bool {
    self == .null
  }

  /// Returns the `Bool` value if this is a `.bool`, otherwise `nil`.
  public var boolValue: Bool? {
    guard case let .bool(value) = self else { return nil }
    return value
  }

  /// Returns the `Int` value if this is an `.int`, otherwise `nil`.
  public var intValue: Int? {
    guard case let .int(value) = self else { return nil }
    return value
  }

  /// Returns the `Double` value if this is a `.double`, otherwise `nil`.
  public var doubleValue: Double? {
    guard case let .double(value) = self else { return nil }
    return value
  }

  /// Returns the `String` value if this is a `.string`, otherwise `nil`.
  public var stringValue: String? {
    guard case let .string(value) = self else { return nil }
    return value
  }

  /// Returns the array if this is an `.array`, otherwise `nil`.
  public var arrayValue: [Value]? {
    guard case let .array(value) = self else { return nil }
    return value
  }

  /// Returns the dictionary if this is an `.object`, otherwise `nil`.
  public var objectValue: [String: Value]? {
    guard case let .object(value) = self else { return nil }
    return value
  }

  // MARK: - String Representation

  /// A string representation of this value, suitable for debugging and display.
  public var stringRepresentation: String {
    switch self {
      case .null:
        return "null"
      case let .bool(value):
        return value ? "true" : "false"
      case let .int(value):
        return String(value)
      case let .double(value):
        return String(value)
      case let .string(value):
        return value
      case let .array(value):
        let elements = value.map { $0.stringRepresentation }.joined(separator: ", ")
        return "[\(elements)]"
      case let .object(value):
        let pairs = value.map { key, val in
          "\"\(key)\": \(val.stringRepresentation)"
        }.joined(separator: ", ")
        return "{\(pairs)}"
    }
  }

  // MARK: - Data Conversion

  /// Creates a Value from raw JSON data.
  public static func fromData(_ data: Data) throws -> Value {
    let object = try JSONSerialization.jsonObject(with: data, options: [])
    return try Value.fromAny(object)
  }

  /// Creates a Value from an arbitrary value.
  public static func fromAny(_ value: Any) throws -> Value {
    switch value {
      case let string as String:
        return .string(string)
      case let number as NSNumber:
        if number.isBool {
          return .bool(number.boolValue)
        } else if number.isInteger {
          return .int(number.intValue)
        } else {
          return .double(number.doubleValue)
        }
      case let array as [Any]:
        return try .array(array.map { try Value.fromAny($0) })
      case let dict as [String: Any]:
        var result = [String: Value]()
        for (key, value) in dict {
          result[key] = try Value.fromAny(value)
        }
        return .object(result)
      case is NSNull:
        return .null
      default:
        throw ValueError.unsupportedType(String(describing: type(of: value)))
    }
  }

  /// Converts this Value to raw JSON data.
  public func toData() throws -> Data {
    let anyValue = toAny()
    return try JSONSerialization.data(withJSONObject: anyValue, options: [])
  }

  /// Converts this Value to a Sendable Any value.
  public func toAny() -> any Sendable {
    switch self {
      case .null:
        return NSNull()
      case let .bool(value):
        return value
      case let .int(value):
        return value
      case let .double(value):
        return value
      case let .string(value):
        return value
      case let .array(value):
        return value.map { $0.toAny() }
      case let .object(value):
        var result = [String: any Sendable]()
        for (key, val) in value {
          result[key] = val.toAny()
        }
        return result
    }
  }

  // MARK: - Errors

  public enum ValueError: LocalizedError {
    case unsupportedType(String)

    public var errorDescription: String? {
      switch self {
        case let .unsupportedType(type):
          "Unsupported value type: \(type)"
      }
    }
  }
}

// MARK: - Codable

extension Value: Codable {
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()

    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Int.self) {
      self = .int(value)
    } else if let value = try? container.decode(Double.self) {
      self = .double(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([Value].self) {
      self = .array(value)
    } else if let value = try? container.decode([String: Value].self) {
      self = .object(value)
    } else {
      throw DecodingError.dataCorruptedError(
        in: container, debugDescription: "Value type not found"
      )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()

    switch self {
      case .null:
        try container.encodeNil()
      case let .bool(value):
        try container.encode(value)
      case let .int(value):
        try container.encode(value)
      case let .double(value):
        try container.encode(value)
      case let .string(value):
        try container.encode(value)
      case let .array(value):
        try container.encode(value)
      case let .object(value):
        try container.encode(value)
    }
  }
}

// MARK: - CustomStringConvertible

extension Value: CustomStringConvertible {
  public var description: String {
    stringRepresentation
  }
}

// MARK: - ExpressibleByNilLiteral

extension Value: ExpressibleByNilLiteral {
  public init(nilLiteral _: ()) {
    self = .null
  }
}

// MARK: - ExpressibleByBooleanLiteral

extension Value: ExpressibleByBooleanLiteral {
  public init(booleanLiteral value: Bool) {
    self = .bool(value)
  }
}

// MARK: - ExpressibleByIntegerLiteral

extension Value: ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Int) {
    self = .int(value)
  }
}

// MARK: - ExpressibleByFloatLiteral

extension Value: ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self = .double(value)
  }
}

// MARK: - ExpressibleByStringLiteral

extension Value: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self = .string(value)
  }
}

// MARK: - ExpressibleByArrayLiteral

extension Value: ExpressibleByArrayLiteral {
  public init(arrayLiteral elements: Value...) {
    self = .array(elements)
  }
}

// MARK: - ExpressibleByDictionaryLiteral

extension Value: ExpressibleByDictionaryLiteral {
  public init(dictionaryLiteral elements: (String, Value)...) {
    var dictionary: [String: Value] = [:]
    for (key, value) in elements {
      dictionary[key] = value
    }
    self = .object(dictionary)
  }
}

// MARK: - ExpressibleByStringInterpolation

extension Value: ExpressibleByStringInterpolation {
  public struct StringInterpolation: StringInterpolationProtocol {
    var stringValue: String

    public init(literalCapacity: Int, interpolationCount: Int) {
      stringValue = ""
      stringValue.reserveCapacity(literalCapacity + interpolationCount)
    }

    public mutating func appendLiteral(_ literal: String) {
      stringValue.append(literal)
    }

    public mutating func appendInterpolation(_ value: some CustomStringConvertible) {
      stringValue.append(value.description)
    }
  }

  public init(stringInterpolation: StringInterpolation) {
    self = .string(stringInterpolation.stringValue)
  }
}

// MARK: - NSNumber Extension

private extension NSNumber {
  /// Determines if this NSNumber represents a Boolean value.
  var isBool: Bool {
    type(of: self) == type(of: NSNumber(value: true))
  }

  /// Determines if this NSNumber represents an integer value.
  var isInteger: Bool {
    let doubleValue = doubleValue
    return doubleValue.truncatingRemainder(dividingBy: 1) == 0
      && doubleValue >= Double(Int.min)
      && doubleValue <= Double(Int.max)
  }
}

// MARK: - Standard Library Type Extensions

public extension Bool {
  /// Creates a boolean value from a `Value` instance.
  ///
  /// In strict mode, only `.bool` values are converted. In non-strict mode,
  /// the following conversions are supported:
  /// - Integers: `1` is `true`, `0` is `false`
  /// - Doubles: `1.0` is `true`, `0.0` is `false`
  /// - Strings (lowercase only):
  ///   - `true`: "true", "t", "yes", "y", "on", "1"
  ///   - `false`: "false", "f", "no", "n", "off", "0"
  ///
  /// - Parameters:
  ///   - value: The `Value` to convert
  ///   - strict: When `true`, only converts from `.bool` values. Defaults to `true`
  /// - Returns: A boolean value if conversion is possible, `nil` otherwise
  init?(_ value: Value, strict: Bool = true) {
    switch value {
      case let .bool(b):
        self = b
      case let .int(i) where !strict:
        switch i {
          case 0: self = false
          case 1: self = true
          default: return nil
        }
      case let .double(d) where !strict:
        switch d {
          case 0.0: self = false
          case 1.0: self = true
          default: return nil
        }
      case let .string(s) where !strict:
        switch s.lowercased() {
          case "true", "t", "yes", "y", "on", "1":
            self = true
          case "false", "f", "no", "n", "off", "0":
            self = false
          default:
            return nil
        }
      default:
        return nil
    }
  }
}

public extension Int {
  /// Creates an integer value from a `Value` instance.
  ///
  /// In strict mode, only `.int` values are converted. In non-strict mode,
  /// the following conversions are supported:
  /// - Doubles: Converted if they can be represented exactly as integers
  /// - Strings: Parsed if they contain a valid integer representation
  ///
  /// - Parameters:
  ///   - value: The `Value` to convert
  ///   - strict: When `true`, only converts from `.int` values. Defaults to `true`
  /// - Returns: An integer value if conversion is possible, `nil` otherwise
  init?(_ value: Value, strict: Bool = true) {
    switch value {
      case let .int(i):
        self = i
      case let .double(d) where !strict:
        guard let intValue = Int(exactly: d) else { return nil }
        self = intValue
      case let .string(s) where !strict:
        guard let intValue = Int(s) else { return nil }
        self = intValue
      default:
        return nil
    }
  }
}

public extension Double {
  /// Creates a double value from a `Value` instance.
  ///
  /// In strict mode, converts from `.double` and `.int` values. In non-strict mode,
  /// also converts from strings.
  ///
  /// - Parameters:
  ///   - value: The `Value` to convert
  ///   - strict: When `true`, only converts from `.double` and `.int` values. Defaults to `true`
  /// - Returns: A double value if conversion is possible, `nil` otherwise
  init?(_ value: Value, strict: Bool = true) {
    switch value {
      case let .double(d):
        self = d
      case let .int(i):
        self = Double(i)
      case let .string(s) where !strict:
        guard let doubleValue = Double(s) else { return nil }
        self = doubleValue
      default:
        return nil
    }
  }
}

public extension String {
  /// Creates a string value from a `Value` instance.
  ///
  /// In strict mode, only `.string` values are converted. In non-strict mode,
  /// also converts from int, double, and bool.
  ///
  /// - Parameters:
  ///   - value: The `Value` to convert
  ///   - strict: When `true`, only converts from `.string` values. Defaults to `true`
  /// - Returns: A string value if conversion is possible, `nil` otherwise
  init?(_ value: Value, strict: Bool = true) {
    switch value {
      case let .string(s):
        self = s
      case let .int(i) where !strict:
        self = String(i)
      case let .double(d) where !strict:
        self = String(d)
      case let .bool(b) where !strict:
        self = String(b)
      default:
        return nil
    }
  }
}

// MARK: - Codable Value Creation

public extension Value {
  /// Create a `Value` from a `Codable` value.
  init(_ value: some Codable) throws {
    if let valueAsValue = value as? Value {
      self = valueAsValue
    } else {
      let data = try JSONEncoder().encode(value)
      self = try JSONDecoder().decode(Value.self, from: data)
    }
  }
}

// MARK: - Internal Constants

extension Value {
  /// Key used internally for buffering partial JSON during streaming.
  static let jsonBufKey = "_jsonBuf"
}

// MARK: - JSON Schema Conversion

public extension Value {
  /// Converts this `Value` to a `JSONValue` for JSON Schema validation.
  func toJSONValue() -> JSONValue {
    switch self {
      case .null:
        .null
      case let .bool(b):
        .boolean(b)
      case let .int(i):
        .integer(i)
      case let .double(d):
        .number(d)
      case let .string(s):
        .string(s)
      case let .array(arr):
        .array(arr.map { $0.toJSONValue() })
      case let .object(obj):
        .object(obj.mapValues { $0.toJSONValue() })
    }
  }
}
