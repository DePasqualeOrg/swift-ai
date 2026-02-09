// Copyright © Anthony DePasquale

import AI
import Foundation
import MCP

// MARK: - AI.Tool → MCP.Tool Conversion

extension MCP.Tool {
  /// Creates an MCP Tool from an AI Tool.
  ///
  /// This extracts the metadata from a Tool to create an MCP Tool definition.
  /// The Tool's `execute` closure is not carried over since MCP Tools
  /// are metadata-only; execution happens via ToolSpec or registered handlers.
  ///
  /// - Parameter tool: The AI Tool to convert
  public init(from tool: AI.Tool) {
    // Build JSON Schema from Tool parameters
    var properties: [String: MCP.Value] = [:]
    var required: [MCP.Value] = []

    for param in tool.parameters {
      properties[param.name] = Self.mcpPropertySchema(for: param.type, description: param.description)
      if param.required {
        required.append(.string(param.name))
      }
    }

    var inputSchema: [String: MCP.Value] = [
      "type": .string("object"),
      "properties": .object(properties),
    ]
    if !required.isEmpty {
      inputSchema["required"] = .array(required)
    }

    self.init(
      name: tool.name,
      title: tool.title,
      description: tool.description,
      inputSchema: .object(inputSchema),
      annotations: .init()
    )
  }

  /// Converts a Tool.ParameterType to an MCP property schema.
  private static func mcpPropertySchema(for type: AI.Tool.ParameterType, description: String) -> MCP.Value {
    switch type {
      case .string:
        .object([
          "type": .string("string"),
          "description": .string(description),
        ])
      case .float:
        .object([
          "type": .string("number"),
          "description": .string(description),
        ])
      case .integer:
        .object([
          "type": .string("integer"),
          "description": .string(description),
        ])
      case .boolean:
        .object([
          "type": .string("boolean"),
          "description": .string(description),
        ])
      case let .array(itemType):
        .object([
          "type": .string("array"),
          "description": .string(description),
          "items": mcpPropertySchema(for: itemType, description: ""),
        ])
      case .object:
        .object([
          "type": .string("object"),
          "description": .string(description),
        ])
    }
  }
}

// MARK: - MCP.Tool → AI.Tool Conversion

extension AI.Tool {
  /// Error thrown when converting an MCP Tool to an AI Tool.
  public enum MCPConversionError: Error {
    case invalidInputSchema(String)
    case unsupportedParameterType(String)
  }

  /// Creates an AI Tool from an MCP Tool.
  ///
  /// Since MCP Tools don't include execution logic, you must provide
  /// an executor closure that will be called when the tool is invoked.
  ///
  /// - Parameters:
  ///   - tool: The MCP Tool to convert
  ///   - executor: Closure that executes the tool and returns a result
  /// - Throws: `MCPConversionError` if the tool's schema can't be converted
  public init(
    from tool: MCP.Tool,
    executor: @escaping @Sendable ([String: AI.Value]) async throws -> [AI.ToolResult.Content]
  ) throws {
    try self.init(from: tool, name: tool.name, executor: executor)
  }

  /// Creates an AI Tool from an MCP Tool with a custom name.
  ///
  /// Use this when you need to rename tools, such as when namespacing
  /// tools from multiple MCP servers.
  ///
  /// - Parameters:
  ///   - tool: The MCP Tool to convert
  ///   - name: Custom name to use for the tool
  ///   - executor: Closure that executes the tool and returns a result
  /// - Throws: `MCPConversionError` if the tool's schema can't be converted
  public init(
    from tool: MCP.Tool,
    name: String,
    executor: @escaping @Sendable ([String: AI.Value]) async throws -> [AI.ToolResult.Content]
  ) throws {
    let parameters = try Self.extractParameters(from: tool.inputSchema)
    let rawSchema = Self.convertMCPValueToJSONValue(tool.inputSchema).objectValue

    self.init(
      name: name,
      description: tool.description ?? "",
      title: tool.title ?? tool.annotations.title ?? tool.name,
      parameters: parameters,
      rawInputSchema: rawSchema,
      execute: executor
    )
  }

  /// Creates an AI Tool from an MCP Tool that calls an MCP client.
  ///
  /// This is a convenience initializer for the common case where you want
  /// to call tools on a remote MCP server.
  ///
  /// - Parameters:
  ///   - tool: The MCP Tool to convert
  ///   - client: The MCP Client to use for tool execution
  /// - Throws: `MCPConversionError` if the tool's schema can't be converted
  public init(from tool: MCP.Tool, client: MCP.Client) throws {
    try self.init(from: tool) { parameters in
      let result = try await client.callTool(
        name: tool.name,
        arguments: parameters.mcpValues
      )
      return try Self.convertCallToolResult(result)
    }
  }

  /// Extracts Tool.Parameter array from an MCP Tool's inputSchema.
  /// Parameters with unsupported types (array, object) are skipped.
  private static func extractParameters(from inputSchema: MCP.Value) throws -> [AI.Tool.Parameter] {
    guard case let .object(schema) = inputSchema,
          case let .object(properties) = schema["properties"]
    else {
      // No properties means no parameters
      return []
    }

    let requiredNames: Set<String> = {
      guard case let .array(required) = schema["required"] else {
        return []
      }
      return Set(required.compactMap { $0.stringValue })
    }()

    var parameters: [AI.Tool.Parameter] = []

    for (name, value) in properties {
      guard case let .object(propSchema) = value else {
        continue
      }

      let description = propSchema["description"]?.stringValue ?? ""
      let parameterType = parseParameterType(from: propSchema)

      // Parse enum values if present
      var enumValues: [String]?
      if case let .array(enumArray) = propSchema["enum"] {
        enumValues = enumArray.compactMap { $0.stringValue }
      }

      parameters.append(AI.Tool.Parameter(
        name: name,
        title: name,
        type: parameterType,
        description: description,
        required: requiredNames.contains(name),
        enumValues: enumValues
      ))
    }

    return parameters
  }

  /// Parses an MCP property schema to extract the AI.Tool.ParameterType.
  private static func parseParameterType(from propSchema: [String: MCP.Value]) -> AI.Tool.ParameterType {
    let typeString = propSchema["type"]?.stringValue ?? "string"

    switch typeString {
      case "string": return .string
      case "number": return .float
      case "integer": return .integer
      case "boolean": return .boolean
      case "array":
        // Parse the items type if present
        if case let .object(itemsSchema) = propSchema["items"] {
          let itemType = parseParameterType(from: itemsSchema)
          return .array(items: itemType)
        }
        return .array(items: .string) // Default to string items
      case "object": return .object
      default: return .string // Treat unknown types as string
    }
  }

  /// Converts an MCP.Value to an AI.Value for schema passthrough.
  private static func convertMCPValueToJSONValue(_ value: MCP.Value) -> AI.Value {
    switch value {
      case let .string(s):
        .string(s)
      case let .int(i):
        .int(i)
      case let .double(d):
        .double(d)
      case let .bool(b):
        .bool(b)
      case .null:
        .null
      case let .array(arr):
        .array(arr.map { convertMCPValueToJSONValue($0) })
      case let .object(obj):
        .object(obj.mapValues { convertMCPValueToJSONValue($0) })
      case let .data(data, _):
        .string(data ?? "")
    }
  }

  /// Error thrown when an MCP tool returns an error result.
  private struct MCPToolError: Error, LocalizedError {
    let message: String
    var errorDescription: String? {
      message
    }
  }

  /// Converts an MCP CallTool.Result to AI ToolResult.Content array.
  private static func convertCallToolResult(_ result: MCP.CallTool.Result) throws -> [AI.ToolResult.Content] {
    // Check for error
    if result.isError == true {
      let errorText = result.content.compactMap { content -> String? in
        if case let .text(text, _, _) = content {
          return text
        }
        return nil
      }.joined(separator: "\n")
      throw MCPToolError(message: errorText.isEmpty ? "Unknown error" : errorText)
    }

    // Convert all content items
    return result.content.map { content in
      switch content {
        case let .text(text, _, _):
          return .text(text)
        case let .image(data, mimeType, _, _):
          if let imageData = Data(base64Encoded: data) {
            return .image(imageData, mimeType: mimeType)
          }
          return .text("[Invalid image data]")
        case let .audio(data, mimeType, _, _):
          if let audioData = Data(base64Encoded: data) {
            return .audio(audioData, mimeType: mimeType)
          }
          return .text("[Invalid audio data]")
        case let .resource(resource, _, _):
          if let text = resource.text {
            return .text(text)
          } else if let blob = resource.blob, let data = Data(base64Encoded: blob) {
            let mimeType = resource.mimeType ?? "application/octet-stream"
            if mimeType.hasPrefix("image/") {
              return .image(data, mimeType: mimeType)
            } else if mimeType.hasPrefix("audio/") {
              return .audio(data, mimeType: mimeType)
            } else {
              return .file(data, mimeType: mimeType, filename: nil)
            }
          } else {
            return .text("[Resource: \(resource.uri)]")
          }
        case let .resourceLink(link):
          return .text("[Resource link: \(link.uri)]")
      }
    }
  }
}

// MARK: - Batch Conversions

public extension [AI.Tool] {
  /// Converts an array of AI Tools to MCP Tools.
  var mcpTools: [MCP.Tool] {
    map { MCP.Tool(from: $0) }
  }
}

public extension [MCP.Tool] {
  /// Converts an array of MCP Tools to AI Tools using a client.
  ///
  /// - Parameter client: The MCP Client to use for tool execution
  /// - Returns: Array of AI Tools
  /// - Throws: If any tool can't be converted
  func aiTools(client: MCP.Client) throws -> [AI.Tool] {
    try map { try AI.Tool(from: $0, client: client) }
  }
}
