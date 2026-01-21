// Copyright © Anthony DePasquale

import AI
import AITool
import Foundation
import Testing

// MARK: - Test Tools

@Tool
struct GetCurrentWeather {
  static let name = "get_current_weather"
  static let description = "Get the current weather for a city. Returns temperature, conditions, and humidity."

  @Parameter(description: "The city name, e.g. 'San Francisco' or 'Tokyo'")
  var city: String

  @Parameter(description: "Temperature unit: 'celsius' or 'fahrenheit'")
  var unit: String = "celsius"

  func perform() async throws -> String {
    // Simulated weather data for testing
    let weatherData: [String: (temp: Int, conditions: String, humidity: Int)] = [
      "paris": (18, "Partly cloudy", 65),
      "london": (14, "Rainy", 80),
      "tokyo": (24, "Sunny", 55),
      "new york": (20, "Clear", 60),
      "san francisco": (16, "Foggy", 75),
    ]

    let cityLower = city.lowercased()
    let data = weatherData[cityLower] ?? (22, "Clear", 50)

    let temp = unit == "fahrenheit" ? (data.temp * 9 / 5 + 32) : data.temp
    let unitSymbol = unit == "fahrenheit" ? "F" : "C"

    return """
    Weather in \(city):
    Temperature: \(temp)°\(unitSymbol)
    Conditions: \(data.conditions)
    Humidity: \(data.humidity)%
    """
  }
}

@Tool
struct CalculateExpression {
  static let name = "calculate"
  static let description = "Evaluate a simple mathematical expression. Supports +, -, *, / operations."

  @Parameter(description: "First number")
  var a: Double

  @Parameter(description: "Mathematical operator: '+', '-', '*', or '/'")
  var op: String

  @Parameter(description: "Second number")
  var b: Double

  func perform() async throws -> String {
    let result: Double = switch op {
      case "+": a + b
      case "-": a - b
      case "*": a * b
      case "/": b != 0 ? a / b : .nan
      default: .nan
    }

    if result.isNaN {
      return "Error: Invalid operation"
    }

    // Format without unnecessary decimal places
    if result.truncatingRemainder(dividingBy: 1) == 0 {
      return "Result: \(Int(result))"
    } else {
      return "Result: \(String(format: "%.2f", result))"
    }
  }
}

@Tool
struct GetCurrentTime {
  static let name = "get_current_time"
  static let description = "Get the current time in a specified timezone."

  @Parameter(description: "Timezone identifier, e.g. 'America/New_York', 'Europe/London', 'Asia/Tokyo'. Use 'UTC' for Coordinated Universal Time.")
  var timezone: String = "UTC"

  func perform() async throws -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"

    if let tz = TimeZone(identifier: timezone) {
      formatter.timeZone = tz
    } else {
      formatter.timeZone = TimeZone(identifier: "UTC")!
    }

    return "Current time in \(timezone): \(formatter.string(from: Date()))"
  }
}

// MARK: - E2E Tests

/// End-to-end tests for AI tools (@Tool macro) with real model providers.
@Suite("AI Tools E2E Tests", .serialized)
struct AIToolsE2ETests {
  // MARK: - Single Tool Tests

  @Test("Gemini: Single tool call with AI tool")
  func geminiSingleTool() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      Issue.record("GEMINI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the current weather in Tokyo? Use celsius.",
      tools: [GetCurrentWeather.tool],
      provider: .gemini(apiKey: apiKey, modelId: "gemini-3-pro-preview")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 1, "Should call at least one tool")
    print("Gemini single tool result: \(result.finalResponse ?? "nil")")
  }

  @Test("Anthropic: Single tool call with AI tool")
  func anthropicSingleTool() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the current weather in London?",
      tools: [GetCurrentWeather.tool],
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 1, "Should call at least one tool")
    print("Anthropic single tool result: \(result.finalResponse ?? "nil")")
  }

  @Test("OpenAI: Single tool call with AI tool")
  func openAISingleTool() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["OPENAI_API_KEY"] else {
      Issue.record("OPENAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's the weather like in San Francisco right now?",
      tools: [GetCurrentWeather.tool],
      provider: .responses(apiKey: apiKey, modelId: "gpt-5.2")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 1, "Should call at least one tool")
    print("OpenAI single tool result: \(result.finalResponse ?? "nil")")
  }

  // MARK: - Multiple AI Tools Tests

  @Test("Gemini: Multiple AI tools with chained reasoning")
  func geminiMultipleTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      Issue.record("GEMINI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What is 15 multiplied by 7? Also, what's the weather in Paris?",
      tools: [CalculateExpression.tool, GetCurrentWeather.tool],
      provider: .gemini(apiKey: apiKey, modelId: "gemini-3-pro-preview")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least two tools")
    print("Gemini multiple tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("Anthropic: Multiple AI tools with chained reasoning")
  func anthropicMultipleTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "Calculate 100 divided by 4, then tell me the weather in New York.",
      tools: [CalculateExpression.tool, GetCurrentWeather.tool],
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least two tools")
    print("Anthropic multiple tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  @Test("OpenAI: Multiple AI tools with chained reasoning")
  func openAIMultipleTools() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["OPENAI_API_KEY"] else {
      Issue.record("OPENAI_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What's 256 plus 128? And what time is it in Tokyo?",
      tools: [CalculateExpression.tool, GetCurrentTime.tool],
      provider: .responses(apiKey: apiKey, modelId: "gpt-5.2")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 2, "Should call at least two tools")
    print("OpenAI multiple tools result: \(result.finalResponse ?? "nil")")
    print("Tool calls executed: \(result.toolCallsExecuted)")
  }

  // MARK: - Tool with Default Parameters

  @Test("Anthropic: Tool uses default parameter value")
  func anthropicToolWithDefaults() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      Issue.record("ANTHROPIC_API_KEY not found in .env file")
      return
    }

    let result = try await runAgenticLoop(
      prompt: "What time is it? (Don't specify a timezone, just get the current time)",
      tools: [GetCurrentTime.tool],
      provider: .anthropic(apiKey: apiKey, modelId: "claude-opus-4-5-20251101")
    )

    #expect(result.finalResponse != nil, "Should have a final response")
    #expect(result.toolCallsExecuted >= 1, "Should call at least one tool")
    print("Anthropic default params result: \(result.finalResponse ?? "nil")")
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
    tools: [Tool],
    provider: Provider,
    maxIterations: Int = 10
  ) async throws -> AgenticLoopResult {
    let toolsCollection = Tools(tools)

    print("[\(provider.name)] Available AI tools: \(tools.map { $0.name })")

    var messages: [AI.Message] = [
      AI.Message(role: .user, content: prompt),
    ]

    var iterations = 0
    var totalToolCalls = 0

    while iterations < maxIterations {
      iterations += 1

      let response = try await generateResponse(
        messages: messages,
        tools: tools,
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

        let results = await toolsCollection.call(response.toolCalls)
        totalToolCalls += response.toolCalls.count

        for (call, result) in zip(response.toolCalls, results) {
          let resultPreview = String(describing: result.content).prefix(200)
          print("[\(provider.name)]   \(call.name) returned: \(resultPreview)...")
        }

        messages.append(results.message)
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
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .anthropic(apiKey, modelId):
        let client = AnthropicClient()
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .responses(apiKey, modelId):
        let client = ResponsesClient()
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
          messages: messages,
          maxTokens: 4096,
          apiKey: apiKey
        )

      case let .chatCompletions(apiKey, modelId, endpoint):
        let client = ChatCompletionsClient(endpoint: endpoint)
        return try await client.generateText(
          modelId: modelId,
          tools: tools,
          systemPrompt: "You are a helpful assistant with access to tools. Use them when needed to answer questions accurately.",
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
