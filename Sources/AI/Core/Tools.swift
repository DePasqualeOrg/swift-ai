// Copyright © Anthony DePasquale

import Foundation
import os.log

private let toolsLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "Tools")

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
  public let execute: @Sendable ([String: Value]) async throws -> ToolOutputResult

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

  /// Captures a schema-build failure so it can be surfaced at request-encoding
  /// time rather than at tool-construction time. Populated by:
  /// - the `@Tool` macro when the raw-schema build fails (e.g. duplicate param names);
  /// - the imperative `Tool.init(parameters:)` paths when the generated schema fails to build;
  /// - imperative callers that explicitly pass `schemaBuildErrorMessage:` to `Tool.init(...)`.
  ///
  /// Strict-mode assertion failures from `@Tool`'s `strictSchema: true` flag trap
  /// before `Tool` is ever constructed, so they never land here.
  let schemaBuildErrorMessage: String?

  /// Optional JSON Schema describing the tool's structured output.
  ///
  /// Populated by the `@Tool` macro from a `StructuredOutput` / `PrimitiveToolOutput` /
  /// `StructuredMetadataCarrier` return type, and copied across the MCP boundary.
  /// Forwarded to Gemini as `functionDeclarations[i].responseJsonSchema`. Anthropic /
  /// OpenAI Chat Completions / Responses don't accept per-tool output schemas and drop
  /// the field silently. When non-nil, `Tools.call()` enforces the contract against
  /// `structuredContent` after the tool returns.
  public let outputSchema: Value?

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
      maximum: Double? = nil,
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
      maxLength: Int? = nil,
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .string,
        description: description,
        required: required,
        enumValues: enumValues,
        minLength: minLength,
        maxLength: maxLength,
      )
    }

    /// Creates an integer parameter.
    public static func integer(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
      minimum: Int? = nil,
      maximum: Int? = nil,
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .integer,
        description: description,
        required: required,
        minimum: minimum.map(Double.init),
        maximum: maximum.map(Double.init),
      )
    }

    /// Creates a number (floating point) parameter.
    public static func number(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
      minimum: Double? = nil,
      maximum: Double? = nil,
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .float,
        description: description,
        required: required,
        minimum: minimum,
        maximum: maximum,
      )
    }

    /// Creates a boolean parameter.
    public static func boolean(
      _ name: String,
      title: String? = nil,
      description: String,
      required: Bool = true,
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .boolean,
        description: description,
        required: required,
      )
    }

    /// Creates an array parameter.
    public static func array(
      _ name: String,
      title: String? = nil,
      description: String,
      items: ParameterType = .string,
      required: Bool = true,
    ) -> Parameter {
      Parameter(
        name: name,
        title: title ?? name,
        type: .array(items: items),
        description: description,
        required: required,
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
    outputSchema: Value? = nil,
    schemaBuildErrorMessage: String? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> ToolOutputResult,
  ) {
    let builtSchema = rawInputSchema.map { BuiltSchema(schema: $0, errorMessage: nil) }
      ?? Self.buildSchema(from: parameters)
    self.name = name
    self.description = description
    self.title = title ?? name
    self.parameters = parameters
    self.resultTypes = resultTypes
    self.rawInputSchema = builtSchema.schema
    self.schemaBuildErrorMessage = schemaBuildErrorMessage ?? builtSchema.errorMessage
    self.outputSchema = outputSchema
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
    outputSchema: Value? = nil,
    schemaBuildErrorMessage: String? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> ToolOutputResult,
  ) {
    let builtSchema = Self.buildSchema(from: parameters)
    self.name = name
    self.description = description
    self.title = title ?? name
    self.parameters = parameters
    self.resultTypes = resultTypes
    rawInputSchema = builtSchema.schema
    self.schemaBuildErrorMessage = schemaBuildErrorMessage ?? builtSchema.errorMessage
    self.outputSchema = outputSchema
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
    outputSchema: Value? = nil,
    schemaBuildErrorMessage: String? = nil,
    execute: @escaping @Sendable ([String: Value]) async throws -> ToolOutputResult,
  ) {
    self.name = name
    self.description = description
    self.title = title ?? name
    parameters = []
    self.resultTypes = resultTypes
    rawInputSchema = inputSchema
    self.schemaBuildErrorMessage = schemaBuildErrorMessage
    self.outputSchema = outputSchema
    self.execute = execute
  }

  /// Returns the tool's input schema after confirming schema generation succeeded.
  ///
  /// Throws when the tool's schema could not be built and `rawInputSchema` only
  /// contains a fallback placeholder schema.
  public func validatedInputSchema() throws -> [String: Value] {
    if let schemaBuildErrorMessage {
      throw AIError.invalidRequest(
        message: "Tool '\(name)' has an invalid input schema: \(schemaBuildErrorMessage)",
      )
    }
    return rawInputSchema
  }

  /// Builds a JSON Schema from the parameter definitions.
  private struct BuiltSchema {
    let schema: [String: Value]
    let errorMessage: String?
  }

  private static func buildSchema(from parameters: [Parameter]) -> BuiltSchema {
    do {
      return try BuiltSchema(
        schema: ToolSchema.buildObjectSchema(
          parameters: parameters,
          name: \.name,
          title: { $0.title },
          description: { $0.description },
          jsonSchemaType: { $0.type.jsonSchemaType },
          jsonSchemaProperties: { $0.type.jsonSchemaProperties(enumValues: $0.enumValues) },
          isOptional: { !$0.required },
          hasDefault: { _ in false },
          defaultValue: { _ in nil },
          minLength: \.minLength,
          maxLength: \.maxLength,
          minimum: \.minimum,
          maximum: \.maximum,
        ),
        errorMessage: nil,
      )
    } catch {
      return BuiltSchema(
        schema: [
          "type": .string("object"),
          "properties": .object([:]),
          "required": .array([]),
        ],
        errorMessage: error.localizedDescription,
      )
    }
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

  func jsonSchemaProperties(enumValues: [String]?) -> [String: Value] {
    var properties: [String: Value] = [:]

    if case let .array(itemType) = self {
      properties["items"] = ["type": .string(itemType.jsonSchemaType)]
    }

    if let enumValues {
      properties["enum"] = .array(enumValues.map(Value.string))
    }

    return properties
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

  /// Calls a single tool and returns the execution result.
  ///
  /// The tool is looked up by name, input is validated against its schema,
  /// and the embedded executor is invoked. Thrown errors are caught and
  /// returned as results with `isError: true`.
  ///
  /// If the current task has been cancelled before execution begins, an
  /// aborted error result is returned immediately without running the tool.
  public func call(_ toolCall: ToolCall) async -> ToolExecutionResult {
    let startedAt = Date()
    if Task.isCancelled {
      return ToolExecutionResult(
        result: errorResult(message: "Tool execution aborted", toolCall: toolCall),
        startedAt: startedAt,
        completedAt: startedAt,
      )
    }

    guard let tool = self[toolCall.name] else {
      let completedAt = Date()
      return ToolExecutionResult(
        result: errorResult(message: "Unknown tool: \(toolCall.name)", toolCall: toolCall),
        startedAt: startedAt,
        completedAt: completedAt,
      )
    }

    // Validate input against schema
    do {
      let inputValue = Value.object(toolCall.parameters)
      let schemaValue = try Value.object(tool.validatedInputSchema())
      try validator.validate(inputValue, against: schemaValue)
    } catch {
      let completedAt = Date()
      return ToolExecutionResult(
        result: errorResult(
          message: "Input validation error: \(error.localizedDescription)",
          toolCall: toolCall,
        ),
        startedAt: startedAt,
        completedAt: completedAt,
      )
    }

    // Execute tool, catching errors
    do {
      let output = try await ToolCallContext.$currentId.withValue(toolCall.id) {
        try await tool.execute(toolCall.parameters)
      }
      let completedAt = Date()
      let validatedResult = validateOutput(
        output: output,
        tool: tool,
        toolCall: toolCall,
      )
      return ToolExecutionResult(
        result: validatedResult,
        startedAt: startedAt,
        completedAt: completedAt,
      )
    } catch is CancellationError {
      let completedAt = Date()
      return ToolExecutionResult(
        result: errorResult(message: "Tool execution aborted", toolCall: toolCall),
        startedAt: startedAt,
        completedAt: completedAt,
      )
    } catch let toolError as ToolError {
      // Author-thrown rich error: preserve content and structured channel.
      let completedAt = Date()
      return ToolExecutionResult(
        result: ToolResult(
          name: toolCall.name,
          id: toolCall.id,
          content: toolError.content,
          structuredContent: toolError.structuredContent,
          isError: true,
        ),
        startedAt: startedAt,
        completedAt: completedAt,
      )
    } catch {
      let completedAt = Date()
      return ToolExecutionResult(
        result: errorResult(message: errorMessage(error), toolCall: toolCall),
        startedAt: startedAt,
        completedAt: completedAt,
      )
    }
  }

  /// Single-`.text` error helper used by every dispatcher-internal failure
  /// path (cancellation, unknown tool, input validation, plain throws). Author-
  /// thrown `ToolError` conformers go through a separate branch in `call(_:)`
  /// that preserves their multi-block content and structured channel.
  private func errorResult(message: String, toolCall: ToolCall) -> ToolResult {
    ToolResult(
      name: toolCall.name,
      id: toolCall.id,
      content: [.text(message)],
      isError: true,
    )
  }

  /// `Error.localizedDescription` returns a locale-dependent NSError stub for
  /// plain Swift errors ("The operation couldn't be completed …"). Falling
  /// back to `String(describing:)` surfaces the type name and stored values,
  /// which is what the agent actually wants. Mirrors swift-mcp's helper.
  private func errorMessage(_ error: Error) -> String {
    if error is LocalizedError {
      return error.localizedDescription
    }
    return String(describing: error)
  }

  /// Calls multiple tools concurrently and returns execution results in order.
  ///
  /// Tools are executed in parallel. On cancellation, tools that have
  /// already completed keep their results. Uncollected tool calls receive
  /// synthesized aborted error results. The method returns promptly on
  /// cancellation without waiting for noncooperative tools to finish.
  public func call(_ toolCalls: [ToolCall]) async -> [ToolExecutionResult] {
    guard !toolCalls.isEmpty else { return [] }

    let (stream, continuation) = AsyncStream<(Int, ToolExecutionResult)>.makeStream()

    // Spawn each tool call as an unstructured task so that the caller
    // is not forced to wait for noncooperative tools on cancellation.
    let tasks = toolCalls.enumerated().map { index, toolCall in
      Task {
        let result = await call(toolCall)
        continuation.yield((index, result))
      }
    }

    var results = [(Int, ToolExecutionResult)]()

    await withTaskCancellationHandler {
      for await indexedResult in stream {
        results.append(indexedResult)
        if results.count == toolCalls.count {
          break
        }
      }
    } onCancel: {
      // Runs immediately when the parent task is cancelled.
      // Cancel child tasks so cooperative tools can observe isCancelled,
      // and finish the stream so the for-await loop exits promptly.
      // Already-buffered values are still drained by the for-await loop
      // so that completed tool results are never discarded.
      for task in tasks {
        task.cancel()
      }
      continuation.finish()
    }

    // Synthesize aborted results for any tool calls we didn't collect.
    let collectedIndices = Set(results.map { $0.0 })
    for (index, toolCall) in toolCalls.enumerated() where !collectedIndices.contains(index) {
      let timestamp = Date()
      results.append((index, ToolExecutionResult(
        result: errorResult(message: "Tool execution aborted", toolCall: toolCall),
        startedAt: timestamp,
        completedAt: timestamp,
      )))
    }

    return results.sorted { $0.0 < $1.0 }.map { $0.1 }
  }

  // MARK: - Output Validation

  /// Validates a tool's `ToolOutputResult` against its declared `outputSchema`,
  /// returning the final `ToolResult`. Producer bugs (missing structured payload
  /// when one is required, or schema mismatch) surface as a synthetic
  /// `isError: true` result with a distinctive `"Output validation error: ..."`
  /// message and a paired `toolsLogger.error` log entry.
  ///
  /// Validation is skipped when `outputSchema == nil` (untyped tools). Errors
  /// thrown from the tool body bypass this path entirely — they land in the
  /// `catch` branch of `Tools.call()`, which constructs an `isError: true`
  /// result that's surfaced unchanged.
  private func validateOutput(
    output: ToolOutputResult,
    tool: Tool,
    toolCall: ToolCall,
  ) -> ToolResult {
    guard let outputSchema = tool.outputSchema else {
      return ToolResult(
        name: toolCall.name,
        id: toolCall.id,
        content: output.content,
        structuredContent: output.structuredContent,
      )
    }

    guard let structuredContent = output.structuredContent else {
      let reason = "has an output schema but no structured content was provided"
      toolsLogger.error("Output validation error: tool '\(toolCall.name, privacy: .public)' \(reason, privacy: .public)")
      return ToolResult(
        name: toolCall.name,
        id: toolCall.id,
        content: [.text("Output validation error: Tool '\(toolCall.name)' \(reason)")],
        isError: true,
      )
    }

    do {
      try validator.validate(structuredContent, against: outputSchema)
    } catch {
      let reason = error.localizedDescription
      toolsLogger.error("Output validation error: tool '\(toolCall.name, privacy: .public)' \(reason, privacy: .public)")
      return ToolResult(
        name: toolCall.name,
        id: toolCall.id,
        content: [.text("Output validation error: Tool '\(toolCall.name)' \(reason)")],
        isError: true,
      )
    }

    return ToolResult(
      name: toolCall.name,
      id: toolCall.id,
      content: output.content,
      structuredContent: structuredContent,
    )
  }

  // MARK: - Collection Conformance

  public var startIndex: Int {
    items.startIndex
  }

  public var endIndex: Int {
    items.endIndex
  }

  public subscript(position: Int) -> Tool {
    items[position]
  }

  public func index(after i: Int) -> Int {
    items.index(after: i)
  }

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
    case json
    case image
    case audio
    case file
    /// Covers all three resource cases: `.embeddedResource`, `.embeddedText`, `.resourceLink`.
    case resource
  }

  /// The actual content returned by a tool, with associated data.
  ///
  /// While `ValueType` is used for capability declarations (metadata),
  /// `Content` holds the actual values returned at runtime.
  /// Use `content.type` to get the corresponding `ValueType`.
  public enum Content: Sendable, Hashable {
    case text(String)
    /// A JSON value to be displayed to the model in the tool result.
    /// Distinct from `ToolResult.structuredContent`, which is the parallel
    /// programmatic channel — `.json` blocks live in the model-facing wire and
    /// reach text-stringifying providers as stringified JSON.
    case json(Value)
    case image(Data, mimeType: String? = nil)
    case audio(Data, mimeType: String)
    case file(Data, mimeType: String, filename: String? = nil)
    /// Inline resource bytes with a URI (e.g., generated PDFs, ZIPs, videos).
    case embeddedResource(Data, uri: String, mimeType: String? = nil)
    /// Inline resource text with a URI (e.g., generated markdown, CSV, code).
    case embeddedText(String, uri: String, mimeType: String? = nil)
    /// A URL reference the client may fetch lazily.
    case resourceLink(
      uri: String,
      name: String,
      title: String? = nil,
      description: String? = nil,
      mimeType: String? = nil,
      size: Int? = nil,
    )

    /// The `ValueType` of this content.
    public var type: ValueType {
      switch self {
        case .text: .text
        case .json: .json
        case .image: .image
        case .audio: .audio
        case .file: .file
        case .embeddedResource, .embeddedText, .resourceLink: .resource
      }
    }

    /// A concise description of the content for fallback messages.
    public var fallbackDescription: String {
      switch self {
        case let .text(content):
          return content
        case let .json(value):
          return value.jsonString
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
        case let .embeddedResource(data, uri, mimeType):
          let type = mimeType ?? "application/octet-stream"
          let size = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
          return "\(uri) (\(type), \(size))"
        case let .embeddedText(text, uri, _):
          return "\(uri):\n\(text)"
        case let .resourceLink(uri, name, _, _, mimeType, size):
          var details: [String] = []
          if let mimeType { details.append(mimeType) }
          if let size {
            details.append(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
          }
          if details.isEmpty {
            return "\(name): \(uri)"
          }
          return "\(name): \(uri) (\(details.joined(separator: ", ")))"
      }
    }
  }

  /// The content items returned by the tool.
  public let content: [Content]

  /// Programmatic structured-data channel, parallel to `content[]`.
  ///
  /// Mirrors swift-mcp's `CallTool.Result.structuredContent` and Gemini's
  /// `functionResponse.response`. Invisible on text-stringifying providers
  /// (Anthropic / Responses / ChatCompletions read `content[]`); becomes the
  /// raw payload of `functionResponse.response` on Gemini; round-trips
  /// losslessly through MCP.
  public let structuredContent: Value?

  /// Whether the tool call ended in an error.
  /// When true, `content` typically contains error message(s).
  public let isError: Bool?

  /// Creates a tool result with multiple content items.
  ///
  /// - Parameters:
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  ///   - content: The content items returned by the tool.
  ///   - structuredContent: Optional structured channel payload.
  ///   - isError: Whether the tool call ended in an error.
  public init(
    name: String,
    id: String,
    content: [Content],
    structuredContent: Value? = nil,
    isError: Bool? = nil,
  ) {
    self.name = name
    self.id = id
    self.content = content
    self.structuredContent = structuredContent
    self.isError = isError
  }

  /// Creates a tool result with a single content item.
  ///
  /// - Parameters:
  ///   - name: The name of the tool that was called.
  ///   - id: The unique identifier of the tool call.
  ///   - content: The content item returned by the tool.
  ///   - structuredContent: Optional structured channel payload.
  ///   - isError: Whether the tool call ended in an error.
  public init(
    name: String,
    id: String,
    content: Content,
    structuredContent: Value? = nil,
    isError: Bool? = nil,
  ) {
    self.init(
      name: name,
      id: id,
      content: [content],
      structuredContent: structuredContent,
      isError: isError,
    )
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
}

/// Metadata about a completed tool execution.
///
/// Wraps the tool's semantic `ToolResult` with exact execution timing so callers can
/// reason about runtime chronology without inventing timestamps after the fact.
public struct ToolExecutionResult: Hashable, Sendable {
  /// The semantic tool result to send back to the model or render in a transcript.
  public let result: ToolResult

  /// When tool execution actually began.
  public let startedAt: Date

  /// When the tool finished and the result became available.
  public let completedAt: Date

  /// Creates a new execution result.
  public init(result: ToolResult, startedAt: Date, completedAt: Date) {
    self.result = result
    self.startedAt = startedAt
    self.completedAt = completedAt
  }

  /// How long the tool ran for.
  public var duration: TimeInterval {
    completedAt.timeIntervalSince(startedAt)
  }

  /// Convenience projection of the tool name.
  public var name: String {
    result.name
  }

  /// Convenience projection of the tool call identifier.
  public var id: String {
    result.id
  }

  /// Convenience projection of the execution payload.
  public var content: [ToolResult.Content] {
    result.content
  }

  /// Convenience projection of the execution error flag.
  public var isError: Bool? {
    result.isError
  }
}

// MARK: - Tool Results Message

public extension [ToolResult] {
  /// A tool message containing these results, suitable for adding to conversation history.
  var message: Message {
    Message(role: .tool, content: map(Message.Content.toolResult))
  }
}

public extension [ToolExecutionResult] {
  /// The semantic tool results extracted from these executions.
  var results: [ToolResult] {
    map(\.result)
  }

  /// A tool message containing these execution results, suitable for adding to conversation history.
  var message: Message {
    results.message
  }
}
