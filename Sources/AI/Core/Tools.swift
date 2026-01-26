// Copyright © Anthony DePasquale

import Foundation

/// Provides the current tool call ID to downstream code via a task-local value.
///
/// When `Tools.call()` invokes a tool's execute closure, it sets the current tool call ID
/// so that downstream code (e.g., MCP tool providers) can correlate progress notifications
/// with specific tool calls.
public enum ToolCallContext {
  @TaskLocal public static var currentId: String?
}

/// A tool that can be called by an LLM during generation.
///
/// Tools enable models to perform actions like searching the web, executing code,
/// or interacting with external services. Each tool has a name, description, and
/// parameters that define its interface, plus an execute closure that performs the action.
public struct Tool: Sendable {
  /// The data type of a tool parameter.
  public indirect enum ParameterType: Sendable, Hashable {
    /// A string value.
    case string
    /// A floating-point number.
    case float
    /// An integer value.
    case integer
    /// A boolean value.
    case boolean
    /// An array of values with the specified item type.
    case array(items: ParameterType = .string)
    /// A JSON object.
    case object
  }

  /// The model-facing name of the tool (typically in snake_case).
  public let name: String

  /// A description of what the tool does, shown to the model.
  public let description: String

  /// A user-facing display name for the tool.
  public let title: String

  /// The parameters this tool accepts.
  public let parameters: [Parameter]

  /// The closure that executes the tool with the given parameters.
  public let execute: @Sendable ([String: Value]) async throws -> [ToolResult.Content]

  /// The types of values this tool may return.
  ///
  /// For `@Tool` macro tools, this is automatically derived from the `perform()` return type
  /// via `ToolOutput.resultTypes`, ensuring the declaration always matches the implementation.
  ///
  /// For imperative tools, this is an optional declaration used for capability-based filtering
  /// (e.g., `tools.compatible(with: ChatCompletionsClient.self)`). If `nil`, the tool is
  /// assumed to potentially return any type and is always included in filtered results.
  public let resultTypes: Set<ToolResult.ValueType>?

  /// Raw JSON Schema for the tool's input parameters.
  /// This is always populated - either from explicit schema or generated from `parameters`.
  /// Providers should use this instead of building a schema from `parameters`.
  /// This preserves complex schema features like nested objects, anyOf, oneOf, etc.
  public let rawInputSchema: [String: Value]

  /// A parameter definition for a tool.
  public struct Parameter: Sendable {
    /// The model-facing name of the parameter.
    public let name: String

    /// The user-facing display name of the parameter.
    public let title: String

    /// The data type of the parameter.
    public let type: ParameterType

    /// A description of the parameter shown to the model.
    public let description: String

    /// Whether the parameter is required.
    public let required: Bool

    /// Allowed values for string parameters (enum constraint).
    public let enumValues: [String]?

    /// Minimum length for string parameters.
    public let minLength: Int?

    /// Maximum length for string parameters.
    public let maxLength: Int?

    /// Minimum value for numeric parameters.
    public let minimum: Double?

    /// Maximum value for numeric parameters.
    public let maximum: Double?

    /// Creates a new parameter definition.
    ///
    /// - Parameters:
    ///   - name: The model-facing name of the parameter.
    ///   - title: The user-facing display name.
    ///   - type: The data type of the parameter.
    ///   - description: A description shown to the model.
    ///   - required: Whether the parameter is required.
    ///   - enumValues: Allowed values for string parameters.
    ///   - minLength: Minimum length for string parameters.
    ///   - maxLength: Maximum length for string parameters.
    ///   - minimum: Minimum value for numeric parameters.
    ///   - maximum: Maximum value for numeric parameters.
    public init(
      name: String,
      title: String,
      type: ParameterType,
      description: String,
      required: Bool,
      enumValues: [String]? = nil,
      minLength: Int? = nil,
      maxLength: Int? = nil,
      minimum: Double? = nil,
      maximum: Double? = nil
    ) {
      self.name = name
      self.title = title
      self.type = type
      self.description = description
      self.required = required
      self.enumValues = enumValues
      self.minLength = minLength
      self.maxLength = maxLength
      self.minimum = minimum
      self.maximum = maximum
    }

    // MARK: - Factory Methods

    /// Creates a string parameter.
    public static func string(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
      enum enumValues: [String]? = nil,
      minLength: Int? = nil,
      maxLength: Int? = nil
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .string,
        description: description,
        required: required,
        enumValues: enumValues,
        minLength: minLength,
        maxLength: maxLength
      )
    }

    /// Creates an integer parameter.
    public static func integer(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
      minimum: Int? = nil,
      maximum: Int? = nil
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .integer,
        description: description,
        required: required,
        minimum: minimum.map(Double.init),
        maximum: maximum.map(Double.init)
      )
    }

    /// Creates a number (floating point) parameter.
    public static func number(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
      minimum: Double? = nil,
      maximum: Double? = nil
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .float,
        description: description,
        required: required,
        minimum: minimum,
        maximum: maximum
      )
    }

    /// Creates a boolean parameter.
    public static func boolean(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .boolean,
        description: description,
        required: required
      )
    }

    /// Creates an array parameter.
    public static func array(
      _ name: String,
      title: String? = nil,
      description: String,
      items: ParameterType = .string,
      required: Bool = true
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .array(items: items),
        description: description,
        required: required
      )
    }
  }

  /// Creates a new tool with explicit parameters and optional raw schema.
  ///
  /// - Parameters:
  ///   - name: The model-facing name of the tool.
  ///   - description: A description of what the tool does.
  ///   - title: A user-facing display name (defaults to `name`).
  ///   - parameters: The parameter definitions for the tool.
  ///   - resultTypes: The types of values this tool may return.
  ///   - rawInputSchema: A raw JSON Schema (overrides generated schema from parameters).
  ///   - execute: The closure that executes the tool.
  public init(
    name: String,
    description: String,
    title: String? = nil,
    parameters: [Parameter],
    resultTypes: Set<ToolResult.ValueType>? = nil,
    rawInputSchema: [String: Value]? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> [ToolResult.Content]
  ) {
    self.name = name
    self.description = description
    self.title = title ?? name
    self.parameters = parameters
    self.resultTypes = resultTypes
    self.rawInputSchema = rawInputSchema ?? Self.buildSchema(from: parameters)
    self.execute = execute
  }

  /// Creates a new tool with parameters.
  ///
  /// The JSON Schema is automatically generated from the parameters array.
  ///
  /// - Parameters:
  ///   - name: The model-facing name of the tool.
  ///   - description: A description of what the tool does.
  ///   - title: A user-facing display name (defaults to `name`).
  ///   - parameters: The parameter definitions for the tool.
  ///   - resultTypes: The types of values this tool may return.
  ///   - execute: The closure that executes the tool.
  public init(
    name: String,
    description: String,
    title: String? = nil,
    parameters: [Parameter] = [],
    resultTypes: Set<ToolResult.ValueType>? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> [ToolResult.Content]
  ) {
    self.name = name
    self.description = description
    self.title = title ?? name
    self.parameters = parameters
    self.resultTypes = resultTypes
    rawInputSchema = Self.buildSchema(from: parameters)
    self.execute = execute
  }

  /// Initializer for raw JSON Schema (e.g., from MCP or external source).
  /// The inputSchema is used directly as rawInputSchema.
  public init(
    name: String,
    description: String,
    title: String? = nil,
    inputSchema: [String: Value],
    resultTypes: Set<ToolResult.ValueType>? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> [ToolResult.Content]
  ) {
    self.name = name
    self.description = description
    self.title = title ?? name
    parameters = []
    self.resultTypes = resultTypes
    rawInputSchema = inputSchema
    self.execute = execute
  }

  /// Builds a JSON Schema from the parameter definitions.
  private static func buildSchema(from parameters: [Parameter]) -> [String: Value] {
    var properties: [String: Value] = [:]
    var required: [Value] = []

    for param in parameters {
      var property: [String: Value] = [
        "type": .string(param.type.jsonSchemaType),
        "description": .string(param.description),
      ]

      // Add title if different from name
      if param.title != param.name {
        property["title"] = .string(param.title)
      }

      // Add items schema for arrays
      if case let .array(itemType) = param.type {
        property["items"] = ["type": .string(itemType.jsonSchemaType)]
      }

      // Add enum values if present
      if let enumValues = param.enumValues {
        property["enum"] = .array(enumValues.map { .string($0) })
      }

      // Add validation constraints
      if let minLength = param.minLength {
        property["minLength"] = .int(minLength)
      }
      if let maxLength = param.maxLength {
        property["maxLength"] = .int(maxLength)
      }
      if let minimum = param.minimum {
        property["minimum"] = .double(minimum)
      }
      if let maximum = param.maximum {
        property["maximum"] = .double(maximum)
      }

      properties[param.name] = .object(property)

      if param.required {
        required.append(.string(param.name))
      }
    }

    var schema: [String: Value] = [
      "type": "object",
      "properties": .object(properties),
    ]

    if !required.isEmpty {
      schema["required"] = .array(required)
    }

    return schema
  }
}

// MARK: - ParameterType JSON Schema

extension Tool.ParameterType {
  /// The JSON Schema type string for this parameter type.
  var jsonSchemaType: String {
    switch self {
      case .string: "string"
      case .float: "number"
      case .integer: "integer"
      case .boolean: "boolean"
      case .array: "array"
      case .object: "object"
    }
  }
}

// MARK: - Tools Collection

/// A collection of tools with validation and execution support.
///
/// `Tools` provides a unified way to work with tools, handling routing of tool calls
/// and validating input against each tool's JSON Schema before execution.
///
/// ## Thread Safety
///
/// When calling multiple tools via `call(_:)`, tools are executed concurrently using
/// a task group. Tool implementations must be thread-safe:
///
/// - Keep tools stateless when possible
/// - Use actors for shared mutable state
/// - Avoid storing state in tool closures that could be accessed concurrently
///
/// ## Example
///
/// ```swift
/// let tools = Tools([weatherTool, searchTool])
///
/// // Execute tool calls from a model response
/// let results = await tools.call(response.toolCalls)
/// messages.append(results.message)
/// ```
public struct Tools: Collection, Sendable {
  private var items: [Tool]
  private let validator: JSONSchemaValidator

  /// Creates a tools collection from an array of tools.
  public init(_ tools: [Tool], validator: JSONSchemaValidator = DefaultJSONSchemaValidator()) {
    items = tools
    self.validator = validator
  }

  /// Creates a tools collection from variadic tool arguments.
  public init(_ tools: Tool..., validator: JSONSchemaValidator = DefaultJSONSchemaValidator()) {
    items = tools
    self.validator = validator
  }

  /// Creates an empty tools collection.
  public init(validator: JSONSchemaValidator = DefaultJSONSchemaValidator()) {
    items = []
    self.validator = validator
  }

  // MARK: - Tool Lookup

  /// Returns the tool with the given name, or nil if not found.
  public subscript(name: String) -> Tool? {
    items.first { $0.name == name }
  }

  /// Returns whether a tool with the given name exists.
  public func contains(named name: String) -> Bool {
    items.contains { $0.name == name }
  }

  // MARK: - Tool Execution

  /// Calls a single tool and returns the result.
  ///
  /// The tool is looked up by name, input is validated against its schema,
  /// and the embedded executor is invoked. Thrown errors are caught and
  /// returned as results with `isError: true`.
  public func call(_ toolCall: GenerationResponse.ToolCall) async -> ToolResult {
    guard let tool = self[toolCall.name] else {
      return .error("Unknown tool: \(toolCall.name)", name: toolCall.name, id: toolCall.id)
    }

    // Validate input against schema
    do {
      let inputValue = Value.object(toolCall.parameters)
      let schemaValue = Value.object(tool.rawInputSchema)
      try validator.validate(inputValue, against: schemaValue)
    } catch {
      return .error("Input validation error: \(error.localizedDescription)", name: toolCall.name, id: toolCall.id)
    }

    // Execute tool, catching errors
    do {
      let content = try await ToolCallContext.$currentId.withValue(toolCall.id) {
        try await tool.execute(toolCall.parameters)
      }
      return ToolResult(name: toolCall.name, id: toolCall.id, content: content)
    } catch {
      return .error(error.localizedDescription, name: toolCall.name, id: toolCall.id)
    }
  }

  /// Calls multiple tools concurrently and returns results in order.
  ///
  /// Tools are executed in parallel using a task group. Tool implementations
  /// must be thread-safe as they may run concurrently.
  public func call(_ toolCalls: [GenerationResponse.ToolCall]) async -> [ToolResult] {
    await withTaskGroup(of: (Int, ToolResult).self) { group in
      for (index, toolCall) in toolCalls.enumerated() {
        group.addTask {
          let result = await call(toolCall)
          return (index, result)
        }
      }

      var results = [(Int, ToolResult)]()
      for await result in group {
        results.append(result)
      }

      return results.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
  }

  // MARK: - Collection Conformance

  public var startIndex: Int { items.startIndex }
  public var endIndex: Int { items.endIndex }
  public subscript(position: Int) -> Tool { items[position] }
  public func index(after i: Int) -> Int { items.index(after: i) }

  // MARK: - Combining Tools

  /// Returns a new collection combining this collection with additional tools.
  public func adding(_ tools: [Tool]) -> Tools {
    Tools(items + tools, validator: validator)
  }

  /// Returns a new collection combining this collection with another.
  public func adding(_ other: Tools) -> Tools {
    Tools(items + other.items, validator: validator)
  }

  /// Returns a new collection with a tool appended.
  public func adding(_ tool: Tool) -> Tools {
    Tools(items + [tool], validator: validator)
  }
}

/// Concatenates two tool arrays into a Tools collection.
public func + (lhs: [Tool], rhs: [Tool]) -> Tools {
  Tools(lhs + rhs)
}

/// Concatenates a Tools collection with an array.
public func + (lhs: Tools, rhs: [Tool]) -> Tools {
  lhs.adding(rhs)
}

/// Concatenates an array with a Tools collection.
public func + (lhs: [Tool], rhs: Tools) -> Tools {
  Tools(lhs).adding(rhs)
}

/// Concatenates two Tools collections.
public func + (lhs: Tools, rhs: Tools) -> Tools {
  lhs.adding(rhs)
}

// MARK: - ExpressibleByArrayLiteral

extension Tools: ExpressibleByArrayLiteral {
  /// Creates a tools collection from an array literal.
  /// Uses the default JSON Schema validator.
  ///
  /// Example:
  /// ```swift
  /// let tools: Tools = [weatherTool, searchTool]
  /// ```
  public init(arrayLiteral elements: Tool...) {
    items = elements
    validator = DefaultJSONSchemaValidator()
  }
}

// MARK: - Tool Filtering

public extension [Tool] {
  /// Returns tools compatible with the given client type.
  /// Tools without declared resultTypes are always included.
  func compatible<C: APIClient>(with _: C.Type) -> [Tool] {
    filter { tool in
      guard let types = tool.resultTypes else {
        return true // No declaration means any result is possible
      }
      return types.isSubset(of: C.supportedResultTypes)
    }
  }

  /// Returns tools that may produce results incompatible with the given client type.
  func incompatible<C: APIClient>(with _: C.Type) -> [Tool] {
    filter { tool in
      guard let types = tool.resultTypes else {
        return false // No declaration means we can't know
      }
      return !types.isSubset(of: C.supportedResultTypes)
    }
  }
}

/// The result of executing a tool.
///
/// Contains the content returned by the tool (text, images, audio, or files)
/// and indicates whether the execution resulted in an error.
public struct ToolResult: Hashable, Sendable {
  /// The name of the tool that was called.
  public let name: String

  /// The unique identifier of the tool call this result corresponds to.
  public let id: String

  /// Represents the category of a tool result, without the associated data.
  ///
  /// Used for capability declarations and filtering:
  /// - `ToolOutput.resultTypes` declares what types a `perform()` method returns
  /// - `Tool.resultTypes` declares what types a tool may produce
  /// - `APIClient.supportedResultTypes` declares what types a client supports
  /// - `tools.compatible(with:)` filters tools based on client capabilities
  public enum ValueType: String, Sendable, Hashable, CaseIterable {
    case text
    case image
    case audio
    case file
  }

  /// The actual content returned by a tool, with associated data.
  ///
  /// While `ValueType` is used for capability declarations (metadata),
  /// `Content` holds the actual values returned at runtime.
  /// Use `content.type` to get the corresponding `ValueType`.
  public enum Content: Sendable, Hashable {
    case text(String)
    case image(Data, mimeType: String? = nil)
    case audio(Data, mimeType: String)
    case file(Data, mimeType: String, filename: String? = nil)

    /// The `ValueType` of this content.
    public var type: ValueType {
      switch self {
        case .text: .text
        case .image: .image
        case .audio: .audio
        case .file: .file
      }
    }

    /// A concise description of the content for fallback messages.
    public var fallbackDescription: String {
      switch self {
        case let .text(content):
          return content
        case let .image(data, mimeType):
          let type = mimeType ?? "image"
          let size = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
          return "[Unsupported result: \(type), \(size)]"
        case let .audio(data, mimeType):
          let size = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
          return "[Unsupported result: \(mimeType), \(size)]"
        case let .file(data, mimeType, filename):
          let size = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
          let name = filename.map { "\($0) " } ?? ""
          return "[Unsupported result: \(name)\(mimeType), \(size)]"
      }
    }
  }

  /// The content items returned by the tool.
  public let content: [Content]

  /// Whether the tool call ended in an error.
  /// When true, `content` typically contains error message(s).
  public let isError: Bool?

  /// Creates a tool result with multiple content items.
  ///
  /// - Parameters:
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  ///   - content: The content items returned by the tool.
  ///   - isError: Whether the tool call ended in an error.
  public init(name: String, id: String, content: [Content], isError: Bool? = nil) {
    self.name = name
    self.id = id
    self.content = content
    self.isError = isError
  }

  /// Creates a tool result with a single content item.
  ///
  /// - Parameters:
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  ///   - content: The content item returned by the tool.
  ///   - isError: Whether the tool call ended in an error.
  public init(name: String, id: String, content: Content, isError: Bool? = nil) {
    self.init(name: name, id: id, content: [content], isError: isError)
  }

  /// Creates a text result.
  ///
  /// - Parameters:
  ///   - text: The text content.
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  /// - Returns: A tool result containing text content.
  public static func text(_ text: String, name: String, id: String) -> ToolResult {
    ToolResult(name: name, id: id, content: [.text(text)])
  }

  /// Creates an error result.
  ///
  /// - Parameters:
  ///   - message: The error message.
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  /// - Returns: A tool result marked as an error.
  public static func error(_ message: String, name: String, id: String) -> ToolResult {
    ToolResult(name: name, id: id, content: [.text(message)], isError: true)
  }
}

// MARK: - Tool Results Message

public extension [ToolResult] {
  /// A tool message containing these results, suitable for adding to conversation history.
  var message: Message {
    Message(
      role: Message.Role.tool,
      content: nil,
      toolResults: self
    )
  }
}
