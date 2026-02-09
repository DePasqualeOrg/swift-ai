// Copyright © Anthony DePasquale

import AI
import Foundation
import MCP

/// Errors that can occur when using MCPToolProvider.
public enum MCPToolProviderError: Error, CustomStringConvertible {
  /// Tool name conflict between multiple servers.
  case toolNameConflict(toolName: String, servers: [String])
  /// Tool not found in any connected server.
  case toolNotFound(String)

  public var description: String {
    switch self {
      case let .toolNameConflict(name, servers):
        "Tool '\(name)' is provided by multiple servers: \(servers.joined(separator: ", ")). Use tools(namespaced: true) to disambiguate."
      case let .toolNotFound(name):
        "Tool '\(name)' not found in any connected server."
    }
  }
}

/// Provides AI Tools from one or more MCP servers.
///
/// Use this to integrate MCP tools with AI providers through swift-ai.
///
/// Tool names are prefixed with the server name by default (e.g., "filesystem__read_file").
/// The double underscore separator ensures compatibility with all LLM providers while
/// clearly distinguishing the namespace from the tool name.
///
/// ## Example with MCPClient (recommended)
///
/// When initialized with `MCPClient`, tool calls automatically benefit from
/// transparent reconnection and retry on session expiration:
///
/// ```swift
/// let mcpClient = MCPClient(name: "MyApp", version: "1.0")
/// try await mcpClient.connect {
///     HTTPClientTransport(endpoint: URL(string: "https://example.com/mcp")!)
/// }
///
/// let toolProvider = MCPToolProvider(client: mcpClient)
/// let mcpTools = try await toolProvider.tools()
/// ```
///
/// ## Example with Client (basic)
///
/// For simple use cases without automatic reconnection:
///
/// ```swift
/// let client = Client(name: "MyApp", version: "1.0")
/// try await client.connect(transport: StdioTransport())
///
/// let toolProvider = MCPToolProvider(client: client)
/// let mcpTools = try await toolProvider.tools()
/// ```
///
/// Use `tools(namespaced: false)` to get tools without the server name prefix.
public actor MCPToolProvider {
  /// Bundles a low-level `Client` with its optional managed `MCPClient` wrapper.
  private struct ClientEntry {
    let client: MCP.Client
    let mcpClient: MCPClient?
  }

  private let entries: [ClientEntry]
  private var serverNames: [String] = []
  private var toolToClientIndex: [String: Int] = [:]
  private var cachedTools: [String: [MCP.Tool]] = [:] // serverName -> tools

  // MARK: - Initialization

  /// Creates a tool provider with a single managed MCPClient.
  ///
  /// Tool calls are routed through `MCPClient`, which provides automatic
  /// reconnection and retry on session expiration.
  ///
  /// - Parameter client: The managed MCP client connected to a server
  public init(client: MCPClient) {
    entries = [ClientEntry(client: client.client, mcpClient: client)]
  }

  /// Creates a tool provider with multiple managed MCPClients.
  ///
  /// Tool calls are routed through `MCPClient`, which provides automatic
  /// reconnection and retry on session expiration.
  ///
  /// Server names are automatically derived from each client's `serverInfo.name`.
  /// If multiple servers have the same name, they are disambiguated with numeric
  /// suffixes (e.g., "server", "server-2", "server-3").
  ///
  /// - Parameter clients: Array of managed MCP clients, each connected to a server
  public init(clients: [MCPClient]) {
    entries = clients.map { ClientEntry(client: $0.client, mcpClient: $0) }
  }

  /// Creates a tool provider with a single MCP client.
  ///
  /// For automatic reconnection and retry, use `init(client: MCPClient)` instead.
  ///
  /// - Parameter client: The MCP client connected to a server
  public init(client: MCP.Client) {
    entries = [ClientEntry(client: client, mcpClient: nil)]
  }

  /// Creates a tool provider with multiple MCP clients.
  ///
  /// For automatic reconnection and retry, use `init(clients: [MCPClient])` instead.
  ///
  /// Server names are automatically derived from each client's `serverInfo.name`.
  /// If multiple servers have the same name, they are disambiguated with numeric
  /// suffixes (e.g., "server", "server-2", "server-3").
  ///
  /// - Parameter clients: Array of MCP clients, each connected to a server
  public init(clients: [MCP.Client]) {
    entries = clients.map { ClientEntry(client: $0, mcpClient: nil) }
  }

  // MARK: - Tools

  /// Gets tools for use with AI providers.
  ///
  /// Tool names are prefixed with the server name by default (e.g., "filesystem__read_file",
  /// "github__create_issue"). This provides useful context to the model about tool origin
  /// and avoids conflicts when mixing MCP tools with locally defined AI tools.
  ///
  /// - Parameters:
  ///   - namespaced: Whether to prefix tool names with the server name. Defaults to `true`.
  ///                 Set to `false` to use original tool names without prefixes.
  ///   - forceRefresh: If true, fetches fresh tools even if cached
  /// - Returns: Array of Tools ready for use with AI providers
  /// - Throws: `MCPToolProviderError.toolNameConflict` if multiple servers provide the same
  ///           tool name and namespacing is disabled.
  public func tools(namespaced: Bool = true, forceRefresh: Bool = false) async throws -> [AI.Tool] {
    try await refreshToolsIfNeeded(forceRefresh: forceRefresh)

    let shouldNamespace = namespaced

    var allTools: [AI.Tool] = []
    var seenNames: [String: String] = [:] // toolName -> serverName (for conflict detection)

    for (index, serverName) in serverNames.enumerated() {
      guard let mcpTools = cachedTools[serverName] else { continue }

      for tool in mcpTools {
        let toolName = shouldNamespace ? "\(serverName)__\(tool.name)" : tool.name

        // Check for conflicts (only when not namespaced)
        if !shouldNamespace, let existingServer = seenNames[toolName] {
          throw MCPToolProviderError.toolNameConflict(
            toolName: toolName,
            servers: [existingServer, serverName]
          )
        }
        seenNames[toolName] = serverName

        // Track which client handles this tool
        toolToClientIndex[toolName] = index

        let aiTool = try AI.Tool(from: tool, name: toolName) { [weak self] parameters in
          guard let self else {
            throw MCPToolProviderError.toolNotFound(toolName)
          }
          let result = try await callToolInternal(
            index: index,
            originalName: tool.name,
            displayName: toolName,
            arguments: parameters.mcpValues
          )
          return try Self.convertResult(result)
        }
        allTools.append(aiTool)
      }
    }

    return allTools
  }

  /// Gets a tool by name for use with AI providers.
  ///
  /// - Parameters:
  ///   - name: The tool name (use namespaced name if tools were fetched with `namespaced: true`)
  ///   - forceRefresh: If true, fetches fresh tools even if cached
  /// - Returns: The Tool, or nil if not found
  public func tool(named name: String, forceRefresh: Bool = false) async throws -> AI.Tool? {
    try await refreshToolsIfNeeded(forceRefresh: forceRefresh)

    // Check if it's a namespaced name (server__tool format)
    if let separatorRange = name.range(of: "__") {
      let serverName = String(name[..<separatorRange.lowerBound])
      let toolName = String(name[separatorRange.upperBound...])

      if let serverIndex = serverNames.firstIndex(of: serverName),
         let mcpTools = cachedTools[serverName],
         mcpTools.first(where: { $0.name == toolName }) != nil
      {
        return try AI.Tool(from: mcpTools.first(where: { $0.name == toolName })!, name: name) { [weak self] parameters in
          guard let self else {
            throw MCPToolProviderError.toolNotFound(name)
          }
          let result = try await callToolInternal(
            index: serverIndex,
            originalName: toolName,
            displayName: name,
            arguments: parameters.mcpValues
          )
          return try Self.convertResult(result)
        }
      }
    }

    // Search all servers for the tool
    for (index, serverName) in serverNames.enumerated() {
      guard let mcpTools = cachedTools[serverName],
            let tool = mcpTools.first(where: { $0.name == name })
      else {
        continue
      }
      return try AI.Tool(from: tool) { [weak self] parameters in
        guard let self else {
          throw MCPToolProviderError.toolNotFound(name)
        }
        let result = try await callToolInternal(
          index: index,
          originalName: tool.name,
          displayName: name,
          arguments: parameters.mcpValues
        )
        return try Self.convertResult(result)
      }
    }

    return nil
  }

  // MARK: - Execution

  /// Executes an AI ToolCall through MCP and returns the result.
  ///
  /// The tool call is automatically routed to the correct server based on the tool name.
  /// When using `MCPClient`, the call benefits from transparent reconnection and retry.
  ///
  /// - Parameter toolCall: The tool call from an AI response
  /// - Returns: The tool result
  /// - Throws: `MCPToolProviderError.toolNotFound` if the tool is not provided by any server
  public func execute(_ toolCall: AI.GenerationResponse.ToolCall) async throws -> AI.ToolResult {
    let (index, originalName) = try resolveClientIndex(for: toolCall.name)
    let entry = entries[index]

    let result: MCP.CallTool.Result = if let mcpClient = entry.mcpClient {
      try await mcpClient.callTool(
        name: originalName,
        arguments: toolCall.parameters.mcpValues
      )
    } else {
      try await entry.client.callTool(
        name: originalName,
        arguments: toolCall.parameters.mcpValues
      )
    }
    return AI.ToolResult(result, name: toolCall.name, id: toolCall.id)
  }

  /// Executes multiple AI ToolCalls through MCP concurrently.
  ///
  /// Tool calls are automatically routed to the correct servers based on tool names.
  ///
  /// - Parameter toolCalls: The tool calls from an AI response
  /// - Returns: Array of tool results in the same order as input
  public func execute(_ toolCalls: [AI.GenerationResponse.ToolCall]) async throws -> [AI.ToolResult] {
    try await withThrowingTaskGroup(of: (Int, AI.ToolResult).self) { group in
      for (index, call) in toolCalls.enumerated() {
        group.addTask {
          let result = try await self.execute(call)
          return (index, result)
        }
      }

      var results = [(Int, AI.ToolResult)]()
      for try await result in group {
        results.append(result)
      }

      return results.sorted { $0.0 < $1.0 }.map { $0.1 }
    }
  }

  // MARK: - Cache Management

  /// Clears the cached tools list from all servers.
  ///
  /// Call this to force a refresh of tools from all connected servers
  /// on the next call to `tools()` or `tool(named:)`.
  public func clearCache() {
    cachedTools = [:]
    toolToClientIndex = [:]
    serverNames = []
  }

  /// Returns the names of all connected servers.
  ///
  /// Server names are derived from `serverInfo.name` and disambiguated with
  /// numeric suffixes if multiple servers have the same name.
  ///
  /// - Returns: Array of server names in connection order.
  public func connectedServerNames() async -> [String] {
    if serverNames.isEmpty {
      serverNames = await buildServerNames()
    }
    return serverNames
  }

  // MARK: - Private Helpers

  /// Calls a tool, routing through MCPClient (with reconnection) when available,
  /// falling back to direct Client call.
  private func callToolInternal(
    index: Int,
    originalName: String,
    displayName: String,
    arguments: [String: MCP.Value]?
  ) async throws -> MCP.CallTool.Result {
    let entry = entries[index]

    if let onProgress = ToolCallProgress.onProgress {
      let toolCallId = ToolCallContext.currentId
      let progressCallback: MCP.ProgressCallback = { progress in
        await onProgress(ToolCallProgressUpdate(
          toolCallId: toolCallId,
          toolName: displayName,
          value: progress.value,
          total: progress.total,
          message: progress.message
        ))
      }

      if let mcpClient = entry.mcpClient {
        return try await mcpClient.callTool(
          name: originalName,
          arguments: arguments,
          onProgress: progressCallback
        )
      } else {
        return try await entry.client.callTool(
          name: originalName,
          arguments: arguments,
          onProgress: progressCallback
        )
      }
    } else {
      if let mcpClient = entry.mcpClient {
        return try await mcpClient.callTool(
          name: originalName,
          arguments: arguments
        )
      } else {
        return try await entry.client.callTool(
          name: originalName,
          arguments: arguments
        )
      }
    }
  }

  private func refreshToolsIfNeeded(forceRefresh: Bool) async throws {
    if serverNames.isEmpty {
      serverNames = await buildServerNames()
    }

    for (index, serverName) in serverNames.enumerated() {
      if forceRefresh || cachedTools[serverName] == nil {
        let entry = entries[index]
        if let mcpClient = entry.mcpClient {
          let result = try await mcpClient.listTools()
          cachedTools[serverName] = result.tools
        } else {
          let result = try await entry.client.listTools()
          cachedTools[serverName] = result.tools
        }
      }
    }
  }

  private func buildServerNames() async -> [String] {
    var names: [String] = []
    var nameCounts: [String: Int] = [:]

    for (index, entry) in entries.enumerated() {
      let baseName = await entry.client.serverInfo?.name ?? "server-\(index + 1)"

      let count = nameCounts[baseName, default: 0]
      nameCounts[baseName] = count + 1

      let finalName = count == 0 ? baseName : "\(baseName)-\(count + 1)"
      names.append(finalName)
    }

    // If there were duplicates, we need to go back and rename the first occurrence too
    // e.g., ["server", "server-2"] should become ["server", "server-2"] (first one keeps original)
    // This is already handled by the logic above.

    return names
  }

  private func resolveClientIndex(for toolName: String) throws -> (index: Int, originalName: String) {
    // Check if we have a direct mapping
    if let index = toolToClientIndex[toolName] {
      // Check if it's a namespaced name (server__tool format)
      if let separatorRange = toolName.range(of: "__") {
        let originalName = String(toolName[separatorRange.upperBound...])
        return (index, originalName)
      }
      return (index, toolName)
    }

    // For single-client case without prior tools() call
    if entries.count == 1 {
      return (0, toolName)
    }

    throw MCPToolProviderError.toolNotFound(toolName)
  }

  /// Error thrown when an MCP tool returns an error result.
  struct MCPToolError: Error, LocalizedError {
    let message: String
    var errorDescription: String? {
      message
    }
  }

  static func convertResult(_ result: MCP.CallTool.Result) throws -> [AI.ToolResult.Content] {
    if result.isError == true {
      let errorText = result.content.compactMap { content -> String? in
        if case let .text(text, _, _) = content {
          return text
        }
        return nil
      }.joined(separator: "\n")
      throw MCPToolError(message: errorText.isEmpty ? "Unknown error" : errorText)
    }

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
