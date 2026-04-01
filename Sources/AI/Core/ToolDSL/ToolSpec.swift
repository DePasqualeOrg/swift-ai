// Copyright © Anthony DePasquale

/// A type that defines an AI tool with typed parameters.
///
/// Conformance is typically added by the `@Tool` macro, which generates:
/// - `tool`: The `Tool` definition including name, description, and JSON Schema
/// - `parse(from:)`: Parsing validated arguments into a typed instance
/// - `init()`: Required empty initializer
///
/// ## Basic Usage
///
/// Most tools are simple. Just define the parameters and implement `perform()`:
///
/// ```swift
/// @Tool
/// struct GetWeather {
///     static let name = "get_weather"
///     static let description = "Get weather for a city"
///
///     @Parameter(description: "City name")
///     var city: String
///
///     func perform() async throws -> String {
///         let weather = await fetchWeather(city: city)
///         return "Weather in \(city): \(weather)"
///     }
/// }
/// ```
///
/// ## Using with AI Providers
///
/// ```swift
/// let tools = Tools([GetWeather.tool])
///
/// let response = try await generateText(
///     model: .anthropic("claude-sonnet-4-20250514"),
///     messages: messages,
///     apiKey: "sk-...",
///     tools: Array(tools)
/// )
///
/// // Execute tool calls
/// let results = await tools.call(response.toolCalls)
/// messages.append(results.message)
/// ```
public protocol ToolSpec: Sendable {
  /// The output type returned by `perform()`.
  associatedtype Output: ToolOutput

  /// The tool definition for use with AI providers.
  static var tool: Tool { get }

  /// Parse arguments into a configured instance.
  /// Called after JSON Schema validation has passed.
  /// - Parameter arguments: The validated arguments dictionary.
  /// - Returns: A configured instance of this tool.
  /// - Throws: `ToolError` if parsing fails.
  static func parse(from arguments: [String: Value]) throws -> Self

  /// Execute the tool's action.
  ///
  /// This method can throw errors for additional validation beyond JSON Schema constraints
  /// (e.g., semantic validation, business rules, or format checks).
  ///
  /// For clear, actionable error messages that help models self-correct, use types
  /// conforming to `LocalizedError`. Without it, the model sees generic messages like
  /// `"The operation couldn't be completed."` which aren't helpful for recovery.
  ///
  /// - Returns: The tool's output, which will be converted to a `ToolResult`.
  /// - Throws: Any error to indicate tool failure. The error's `localizedDescription`
  ///   is returned to the model with `isError: true`.
  func perform() async throws -> Output

  /// Required empty initializer for instance creation during parsing.
  /// Generated automatically by the `@Tool` macro.
  init()
}

// The @Tool macro is provided by the AITool module.
// Import AITool alongside AI to define tools:
//
//     import AI
//     import AITool
//
//     @Tool
//     struct MyTool { ... }
