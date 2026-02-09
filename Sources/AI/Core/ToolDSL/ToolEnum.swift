// Copyright © Anthony DePasquale

/// Protocol for enum types that can be used as tool parameters.
///
/// Conforming types must be `RawRepresentable` with a `String` raw value
/// and provide the list of all possible cases for JSON Schema generation.
///
/// Example:
/// ```swift
/// enum Priority: String, ToolEnum, CaseIterable {
///     case low, medium, high
/// }
///
/// @Tool
/// struct SetPriority {
///     static let name = "set_priority"
///     static let description = "Set task priority"
///
///     @Parameter(description: "Priority level")
///     var priority: Priority
/// }
/// ```
public protocol ToolEnum: ParameterValue, RawRepresentable, CaseIterable where RawValue == String {}

public extension ToolEnum {
  static var jsonSchemaType: String {
    "string"
  }

  /// Uses the first case as the placeholder value.
  static var placeholderValue: Self {
    guard let first = allCases.first else {
      fatalError("ToolEnum '\(Self.self)' must have at least one case")
    }
    return first
  }

  static var jsonSchemaProperties: [String: Value] {
    let cases = allCases.map { Value.string($0.rawValue) }
    return ["enum": .array(cases)]
  }

  /// Parse an enum from a Value containing its raw string value.
  init?(parameterValue value: Value) {
    guard case let .string(str) = value else { return nil }
    self.init(rawValue: str)
  }
}
