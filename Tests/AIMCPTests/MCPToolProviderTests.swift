// Copyright © Anthony DePasquale

import AIMCP
import Foundation
import Testing

struct MCPToolProviderTests {
  /// `MCPToolProvider.execute(_:)` must warm swift-mcp's `toolOutputSchemas`
  /// cache via `listTools()` before dispatching the tool call. Otherwise a
  /// fresh provider — or one whose cache was just cleared — silently drops
  /// the structured-output validation that swift-mcp performs in
  /// `Client+ProtocolMethods.callTool`.
  ///
  /// The setup:
  ///   - Server publishes a tool whose `outputSchema` requires `score:
  ///     integer`.
  ///   - Server returns `structuredContent: [:]` (missing `score`) when called.
  ///   - Validation must reject this with `MCPError.invalidParams`.
  ///
  /// Without the warming fix, `execute(_:)` would silently return the
  /// invalid result; the cache wouldn't be populated, so swift-mcp's
  /// `validateToolOutput` short-circuits.
  @Test
  func `execute warms swift-mcp output schema cache for fresh providers`() async throws {
    let server = MCP.Server(
      name: "test-server",
      version: "1.0.0",
      capabilities: .init(tools: .init()),
    )

    let outputSchema: MCP.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "score": .object(["type": .string("integer")]),
      ]),
      "required": .array([.string("score")]),
    ])

    await server.withRequestHandler(MCP.ListTools.self) { _, _ in
      MCP.ListTools.Result(tools: [
        MCP.Tool(
          name: "rate",
          description: "Returns a rating",
          inputSchema: .object([
            "type": .string("object"),
            "properties": .object([:]),
          ]),
          outputSchema: outputSchema,
        ),
      ])
    }

    await server.withRequestHandler(MCP.CallTool.self) { _, _ in
      MCP.CallTool.Result(
        content: [.text("rated")],
        structuredContent: .object([:]),
      )
    }

    let (clientTransport, serverTransport) = await MCP.InMemoryTransport.createConnectedPair()
    try await server.start(transport: serverTransport)

    let client = MCP.Client(name: "test-client", version: "1.0.0")
    _ = try await client.connect(transport: clientTransport)

    let provider = MCPToolProvider(client: client)

    // Note: we deliberately do NOT call `provider.tools()` first. This is the
    // "fresh provider" path — `execute(_:)` must warm the schema cache itself.
    await #expect(throws: MCP.MCPError.self) {
      _ = try await provider.execute(
        AI.ToolCall(name: "rate", id: "call_1", parameters: [:]),
      )
    }
  }

  /// Counterpart: when the server's structured output satisfies the schema,
  /// `execute(_:)` returns successfully — confirming the warming is itself
  /// non-destructive.
  @Test
  func `execute returns valid structured output without throwing`() async throws {
    let server = MCP.Server(
      name: "test-server",
      version: "1.0.0",
      capabilities: .init(tools: .init()),
    )

    let outputSchema: MCP.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "score": .object(["type": .string("integer")]),
      ]),
      "required": .array([.string("score")]),
    ])

    await server.withRequestHandler(MCP.ListTools.self) { _, _ in
      MCP.ListTools.Result(tools: [
        MCP.Tool(
          name: "rate",
          description: "Returns a rating",
          inputSchema: .object([
            "type": .string("object"),
            "properties": .object([:]),
          ]),
          outputSchema: outputSchema,
        ),
      ])
    }

    await server.withRequestHandler(MCP.CallTool.self) { _, _ in
      MCP.CallTool.Result(
        content: [.text("rated")],
        structuredContent: .object(["score": .int(7)]),
      )
    }

    let (clientTransport, serverTransport) = await MCP.InMemoryTransport.createConnectedPair()
    try await server.start(transport: serverTransport)

    let client = MCP.Client(name: "test-client", version: "1.0.0")
    _ = try await client.connect(transport: clientTransport)

    let provider = MCPToolProvider(client: client)
    let result = try await provider.execute(
      AI.ToolCall(name: "rate", id: "call_1", parameters: [:]),
    )

    #expect(result.isError != true)
    #expect(result.structuredContent == .object(["score": .int(7)]))
  }
}
