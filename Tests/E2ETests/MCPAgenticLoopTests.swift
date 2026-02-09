// Copyright © Anthony DePasquale

import AI
import AIMCP
import AITool
import Foundation
import MCP
import Testing

// MARK: - Local AI Tool for Mixed Testing

@Tool
struct LocalCalculator {
  static let name = "local_calculator"
  static let description = "A local calculator tool that performs basic arithmetic. Use this for mathematical calculations."

  @Parameter(description: "First operand")
  var x: Double

  @Parameter(description: "Operator: 'add', 'subtract', 'multiply', or 'divide'")
  var operation: String

  @Parameter(description: "Second operand")
  var y: Double

  func perform() async throws -> String {
    let result: Double = switch operation.lowercased() {
      case "add", "+": x + y
      case "subtract", "-": x - y
      case "multiply", "*": x * y
      case "divide", "/": y != 0 ? x / y : .nan
      default: .nan
    }

    if result.isNaN {
      return "Error: Invalid operation or division by zero"
    }

    if result.truncatingRemainder(dividingBy: 1) == 0 {
      return "Calculation result: \(Int(result))"
    } else {
      return "Calculation result: \(String(format: "%.4f", result))"
    }
  }
}

/// End-to-end tests for agentic loops with MCP tools across different providers.
@Suite("MCP Agentic Loop Tests", .serialized)
struct MCPAgenticLoopTests {
  static let mcpServerURL = URL(string: "http://localhost:52274/mcp")!

  // MARK: - Provider Tests

  @Test("Gemini: Weather in Paris with MCP tools")
  func geminiWeatherInParis() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      Issue.record("GEMINI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather like in Paris?",
      provider: .gemini(apiKey: apiKey, modelId: "gemini-3-pro-preview")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.iterations > 0, "Should have at least one iteration")
    print("Gemini result: \(result.finalResponse ?? "nil")")
  }

  @Test("Anthropic: Weather in Paris with MCP tools")
  func anthropicWeatherInParis() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather like in Paris?",
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.iterations > 0, "Should have at least one iteration")
    print("Anthropic result: \(result.finalResponse ?? "nil")")
  }

  @Test("OpenAI Responses: Weather in Paris with MCP tools")
  func openAIResponsesWeatherInParis() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["OPENAI_API_KEY"] else {
      Issue.record("OPENAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather like in Paris?",
      provider: .responses(apiKey: apiKey, modelId: "gpt-5.2")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.iterations > 0, "Should have at least one iteration")
    print("OpenAI Responses result: \(result.finalResponse ?? "nil")")
  }

  @Test("xAI: Weather in Paris with MCP tools")
  func xaiWeatherInParis() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["XAI_API_KEY"] else {
      Issue.record("XAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather like in Paris?",
      provider: .chatCompletions(
        apiKey: apiKey,
        modelId: "grok-4-1-fast-reasoning",
        endpoint: #require(URL(string: "https://api.x.ai/v1/chat/completions"))
      )
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.iterations > 0, "Should have at least one iteration")
    print("xAI result: \(result.finalResponse ?? "nil")")
  }

  // MARK: - Multi-Tool Tests

  @Test("Gemini: Chained reasoning with multiple tools")
  func geminiChainedReasoning() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      Issue.record("GEMINI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather in San Francisco? Based on that, use another model to suggest what I should wear today.",
      provider: .gemini(apiKey: apiKey, modelId: "gemini-3-pro-preview")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (weather + ask_model)")
    print("Gemini chained result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("Anthropic: Chained reasoning with multiple tools")
  func anthropicChainedReasoning() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather in San Francisco? Based on that, use another model to suggest what I should wear today.",
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (weather + ask_model)")
    print("Anthropic chained result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("OpenAI Responses: Chained reasoning with multiple tools")
  func openAIResponsesChainedReasoning() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["OPENAI_API_KEY"] else {
      Issue.record("OPENAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather in San Francisco? Based on that, use another model to suggest what I should wear today.",
      provider: .responses(apiKey: apiKey, modelId: "gpt-5.2")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (weather + ask_model)")
    print("OpenAI Responses chained result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("xAI: Chained reasoning with multiple tools")
  func xaiChainedReasoning() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["XAI_API_KEY"] else {
      Issue.record("XAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather in San Francisco? Based on that, use another model to suggest what I should wear today.",
      provider: .chatCompletions(
        apiKey: apiKey,
        modelId: "grok-4-1-fast-reasoning",
        endpoint: #require(URL(string: "https://api.x.ai/v1/chat/completions"))
      )
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (weather + ask_model)")
    print("xAI chained result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  // MARK: - Mixed MCP + AI Tools Tests

  @Test("Gemini: Mixed MCP and AI tools")
  func geminiMixedTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      Issue.record("GEMINI_API_KEY not found in .env file")
      return
    }

    let result = try await runMixedToolsLoop(
      prompt: "What's the weather in Tokyo? Also, use the calculator to compute 42 multiplied by 17.",
      localTools: [LocalCalculator.tool],
      provider: .gemini(apiKey: apiKey, modelId: "gemini-3-pro-preview")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (weather from MCP + calculator)")
    print("Gemini mixed tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("Anthropic: Mixed MCP and AI tools")
  func anthropicMixedTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runMixedToolsLoop(
      prompt: "Calculate 99 divided by 3 using the local calculator. Then get the weather in Paris using the weather tool.",
      localTools: [LocalCalculator.tool],
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools (calculator + weather from MCP)")
    print("Anthropic mixed tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("OpenAI: Mixed MCP and AI tools")
  func openAIMixedTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["OPENAI_API_KEY"] else {
      Issue.record("OPENAI_API_KEY not found in .env file")
      return
    }

    let result = try await runMixedToolsLoop(
      prompt: "What is 1000 subtract 371? Also, what's the current weather in London?",
      localTools: [LocalCalculator.tool],
      provider: .responses(apiKey: apiKey, modelId: "gpt-5.2")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least 2 tools")
    print("OpenAI mixed tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  // MARK: - Provider Configuration

  enum Provider {
    case gemini(apiKey: String, modelId: String)
    case anthropic(apiKey: String, modelId: String)
    case responses(apiKey: String, modelId: String)
    case chatCompletions(apiKey: String, modelId: String, endpoint: URL)

    var name: String {
      switch self {
        case .gemini: "Gemini"
        case .anthropic: "Anthropic"
        case .responses: "Responses"
        case .chatCompletions: "ChatCompletions"
      }
    }
  }

  // MARK: - Agentic Loop Result

  struct AgenticLoopResult {
    let finalResponse: String?
    let iterations: Int
    let toolCallsExecuted: Int
  }

  // MARK: - Shared Agentic Loop

  func runAgenticLoop(
    prompt: String,
    provider: Provider,
    maxIterations: Int = 10
  ) async throws -> AgenticLoopResult {
    // Connect to MCP server with long-lived SSE configuration
    let mcpClient = MCP.Client(name: "E2ETest-\(provider.name)", version: "1.0.0")
    let configuration = URLSessionConfiguration.default
    configuration.timeoutIntervalForRequest = 300 // 5 minutes for long tool calls
    configuration.timeoutIntervalForResource = 3600 // 1 hour for the SSE connection
    let transport = HTTPClientTransport(
      endpoint: Self.mcpServerURL,
      configuration: configuration
    )
    try await mcpClient.connect(transport: transport)
    defer { Task { await mcpClient.disconnect() } }

    // Get tools from MCP
    let toolProvider = MCPToolProvider(client: mcpClient)
    let tools = try await toolProvider.tools()
    print("[\(provider.name)] Available tools: \(tools.count)")

    // Initialize messages
    var messages: [AI.Message] = [
      AI.Message(role: .user, content: prompt),
    ]

    var iterations = 0
    var totalToolCalls = 0

    // Run agentic loop
    while iterations < maxIterations {
      iterations += 1

      let response = try await generateResponse(
        messages: messages,
        tools: tools,
        provider: provider
      )

      print("[\(provider.name)] Iteration \(iterations): \(response.toolCalls.count) tool calls")

      if !response.toolCalls.isEmpty {
        // Log tool calls with arguments
        for call in response.toolCalls {
          let argsJSON = formatParameters(call.parameters)
          print("[\(provider.name)] Calling tool: \(call.name)")
          print("[\(provider.name)]   Arguments: \(argsJSON)")
        }

        // Add assistant message with tool calls
        messages.append(response.message)

        // Execute tool calls
        let results = try await toolProvider.execute(response.toolCalls)
        totalToolCalls += response.toolCalls.count

        for (call, result) in zip(response.toolCalls, results) {
          let resultPreview = String(describing: result.content).prefix(200)
          print("[\(provider.name)]   \(call.name) returned: \(resultPreview)...")
        }

        // Add tool results
        messages.append(results.message)
        continue
      }

      // No tool calls - we have the final response
      let finalResponse = response.texts.response
      print("[\(provider.name)] Final response: \(finalResponse ?? "nil")")

      return AgenticLoopResult(
        finalResponse: finalResponse,
        iterations: iterations,
        toolCallsExecuted: totalToolCalls
      )
    }

    // Max iterations reached
    return AgenticLoopResult(
      finalResponse: nil,
      iterations: iterations,
      toolCallsExecuted: totalToolCalls
    )
  }

  // MARK: - Mixed Tools Agentic Loop

  /// Runs an agentic loop with both MCP tools and local AI tools combined.
  func runMixedToolsLoop(
    prompt: String,
    localTools: [AI.Tool],
    provider: Provider,
    maxIterations: Int = 10
  ) async throws -> AgenticLoopResult {
    // Connect to MCP server
    let mcpClient = MCP.Client(name: "E2ETest-Mixed-\(provider.name)", version: "1.0.0")
    let configuration = URLSessionConfiguration.default
    configuration.timeoutIntervalForRequest = 300
    configuration.timeoutIntervalForResource = 3600
    let transport = HTTPClientTransport(
      endpoint: Self.mcpServerURL,
      configuration: configuration
    )
    try await mcpClient.connect(transport: transport)
    defer { Task { await mcpClient.disconnect() } }

    // Get MCP tools
    let mcpToolProvider = MCPToolProvider(client: mcpClient)
    let mcpTools = try await mcpToolProvider.tools()
    print("[\(provider.name)] MCP tools: \(mcpTools.map { $0.name })")
    print("[\(provider.name)] Local AI tools: \(localTools.map { $0.name })")

    // Combine MCP tools with local tools
    var allTools: [AI.Tool] = []
    allTools.append(contentsOf: mcpTools)
    allTools.append(contentsOf: localTools)
    let localToolsCollection = Tools(localTools)

    var messages: [AI.Message] = [
      AI.Message(role: .user, content: prompt),
    ]

    var iterations = 0
    var totalToolCalls = 0

    while iterations < maxIterations {
      iterations += 1

      let response = try await generateResponse(
        messages: messages,
        tools: allTools,
        provider: provider
      )

      print("[\(provider.name)] Iteration \(iterations): \(response.toolCalls.count) tool calls")

      if !response.toolCalls.isEmpty {
        for call in response.toolCalls {
          let argsJSON = formatParameters(call.parameters)
          print("[\(provider.name)] Calling tool: \(call.name)")
          print("[\(provider.name)]   Arguments: \(argsJSON)")
        }

        messages.append(response.message)

        // Route tool calls to the appropriate executor
        var results: [ToolResult] = []
        var mcpCalls: [GenerationResponse.ToolCall] = []
        var localCalls: [GenerationResponse.ToolCall] = []

        for call in response.toolCalls {
          if localToolsCollection.contains(named: call.name) {
            localCalls.append(call)
          } else {
            mcpCalls.append(call)
          }
        }

        // Execute MCP tool calls
        if !mcpCalls.isEmpty {
          let mcpResults = try await mcpToolProvider.execute(mcpCalls)
          results.append(contentsOf: mcpResults)
        }

        // Execute local tool calls
        if !localCalls.isEmpty {
          let localResults = await localToolsCollection.call(localCalls)
          results.append(contentsOf: localResults)
        }

        // Sort results back to original order
        let orderedResults = response.toolCalls.compactMap { call in
          results.first { $0.id == call.id }
        }

        totalToolCalls += response.toolCalls.count

        for (call, result) in zip(response.toolCalls, orderedResults) {
          let resultPreview = String(describing: result.content).prefix(200)
          print("[\(provider.name)]   \(call.name) returned: \(resultPreview)...")
        }

        messages.append(orderedResults.message)
        continue
      }

      let finalResponse = response.texts.response
      print("[\(provider.name)] Final response: \(finalResponse ?? "nil")")

      return AgenticLoopResult(
        finalResponse: finalResponse,
        iterations: iterations,
        toolCallsExecuted: totalToolCalls
      )
    }

    return AgenticLoopResult(
      finalResponse: nil,
      iterations: iterations,
      toolCallsExecuted: totalToolCalls
    )
  }

  // MARK: - Provider-Specific Generation

  private func generateResponse(
    messages: [AI.Message],
    tools: [AI.Tool],
    provider: Provider
  ) async throws -> GenerationResponse {
    switch provider {
      case let .gemini(apiKey, modelId):
        let client = GeminiClient()
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .anthropic(apiKey, modelId):
        let client = AnthropicClient()
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .responses(apiKey, modelId):
        let client = ResponsesClient()
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .chatCompletions(apiKey, modelId, endpoint):
        let client = ChatCompletionsClient(endpoint: endpoint)
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )
    }
  }

  // MARK: - Helpers

  private func formatParameters(_ parameters: [String: AI.Value]) -> String {
    guard !parameters.isEmpty else { return "{}" }
    let pairs = parameters.map { key, value in
      "\"\(key)\": \(formatValue(value))"
    }
    return "{ \(pairs.joined(separator: ", ")) }"
  }

  private func formatValue(_ value: AI.Value) -> String {
    switch value {
      case let .string(s): "\"\(s)\""
      case let .int(i): "\(i)"
      case let .double(d): "\(d)"
      case let .bool(b): "\(b)"
      case .null: "null"
      case let .array(arr): "[\(arr.map { formatValue($0) }.joined(separator: ", "))]"
      case let .object(obj): formatParameters(obj)
    }
  }
}
