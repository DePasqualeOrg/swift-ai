# Swift AI

AI API clients with tool use and MCP integration

**This package is in early development. Expect breaking changes.**

## Documentation

- [Quick Start](#quick-start)
- [Top-Level API](#top-level-api)
- [Provider-Specific Features](#provider-specific-features)
- [Media Attachments](#media-attachments)
- [Tools](#tools)
- [Imperative Tool Usage](#imperative-tool-usage)
- [MCP Integration](#mcp-integration)
- [Error Handling](#error-handling)
- [Custom Endpoints and Sessions](#custom-endpoints-and-sessions)
- [Cancellation](#cancellation)

## Quick Start

> **Note:** Never store API keys in your app bundle. Users should provide their own keys at runtime.

### Non-Streaming

```swift
import AI

let response = try await generateText(
    model: .anthropic("claude-opus-4-5"),
    prompt: "Recommend a book similar to The Magic Mountain.",
    apiKey: "sk-ant-..."
)

print(response.texts.response ?? "No response")
```

For multi-turn conversations, use `messages:` instead of `prompt:`:

```swift
let response = try await generateText(
    model: .anthropic("claude-opus-4-5"),
    messages: [
        Message(role: .user, content: "I'm planning to hike the Tour du Mont Blanc."),
        Message(role: .assistant, content: "Great choice! The TMB is a 170km trek through France, Italy, and Switzerland. When are you planning to go?"),
        Message(role: .user, content: "Late August. What should I pack?")
    ],
    apiKey: "sk-ant-..."
)
```

### Streaming

Use `streamText()` to receive responses as they're generated:

```swift
import AI

for try await partial in streamText(
    model: .anthropic("claude-opus-4-5"),
    prompt: "Plan a 3-day hiking trip in the Swiss Alps.",
    apiKey: "sk-ant-..."
) {
    print(partial.texts.response ?? "", terminator: "")
}
```

### Using Clients Directly

For full control over provider-specific configuration, use the provider clients directly:

```swift
import AI

let client = AnthropicClient()

// Non-streaming
let response = try await client.generateText(
    modelId: "claude-opus-4-5",
    systemPrompt: "You are a helpful mountain guide assistant.",
    messages: [Message(role: .user, content: "What's the best season to climb Mont Blanc?")],
    maxTokens: 1024,
    apiKey: "sk-ant-..."
)

// Streaming
for try await partial in client.streamText(
    modelId: "claude-opus-4-5",
    systemPrompt: "You are a helpful mountain guide assistant.",
    messages: [Message(role: .user, content: "Describe the Matterhorn.")],
    maxTokens: 1024,
    apiKey: "sk-ant-..."
) {
    print(partial.texts.response ?? "")
}
```

## Top-Level API

The `Model` enum specifies both the provider and model ID. The Chat Completions and Responses clients default to OpenAI but can be configured for any compatible provider by specifying an endpoint.

```swift
public enum Model {
    case anthropic(String)
    case gemini(String)
    case chatCompletions(String, endpoint: URL = ChatCompletionsClient.Endpoint.openAI.url)
    case responses(String, endpoint: URL = ResponsesClient.Endpoint.openAI.url)
}
```

The top-level `generateText` and `streamText` functions provide options that map to provider-specific configurations:

```swift
let response = try await generateText(
    model: .gemini("gemini-3-pro-preview"),
    messages: messages,
    apiKey: apiKey,
    webSearch: true,  // Defaults to false
    reasoning: false  // Defaults to true
)
```

## Provider-Specific Features

### Anthropic

```swift
let client = AnthropicClient()

// With extended thinking and web search
let config = AnthropicClient.Configuration(
    maxThinkingTokens: 10000,  // Enable extended thinking (minimum: 1024)
    webSearch: true,           // Enable web search tool
    webContent: false,         // Enable web content fetching
    codeExecution: false       // Enable code execution
)

// Use the model-specific maximum thinking budget
let modelId = "claude-opus-4-5"
let config = AnthropicClient.Configuration(
    maxThinkingTokens: AnthropicClient.maxThinkingBudget(for: modelId)
)

for try await partial in client.streamText(
    modelId: modelId,
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 4096,
    apiKey: apiKey,
    configuration: config
) {
    // Access reasoning content
    if let reasoning = partial.texts.reasoning {
        print("Reasoning: \(reasoning)")
    }
}
```

### OpenAI (Chat Completions)

```swift
let client = ChatCompletionsClient()

// With extra parameters for provider-specific options
let config = ChatCompletionsClient.Configuration(
    extraParameters: [
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2
    ]
)

for try await partial in client.streamText(
    modelId: "gpt-5.2",
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 4096,
    temperature: 0.7,
    apiKey: apiKey,
    configuration: config
) {
    print(partial.texts.response ?? "")
}
```

### OpenAI (Responses)

For models with reasoning support:

```swift
let client = ResponsesClient()

let config = ResponsesClient.Configuration(
    reasoningEffortLevel: .medium,     // .minimal, .low, .medium, .high
    verbosityLevel: nil,               // .low, .medium, .high (optional)
    serverSideTools: [                 // Provider-specific server-side tools
        .OpenAI.webSearch(contextSize: .medium),
        .OpenAI.codeInterpreter()
    ],
    backgroundMode: false              // Enable background mode for long responses
)

for try await partial in client.streamText(
    modelId: "gpt-5.2",
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 4096,
    apiKey: apiKey,
    configuration: config
) {
    print(partial.texts.response ?? "")
}
```

#### Server-Side Tools

The `ResponsesClient` supports provider-specific server-side tools that run on the provider's infrastructure:

```swift
// OpenAI server-side tools
let openAIConfig = ResponsesClient.Configuration(
    serverSideTools: [
        .OpenAI.webSearch(contextSize: .medium),  // .low, .medium, .high
        .OpenAI.codeInterpreter()
    ]
)

// xAI server-side tools
let xAIClient = ResponsesClient(endpoint: .xAI)
let xAIConfig = ResponsesClient.Configuration(
    serverSideTools: [
        .xAI.webSearch(),      // Web search
        .xAI.xSearch(),        // X search
        .xAI.codeExecution()   // Code execution
    ]
)

// Custom server-side tools for other providers
let customConfig = ResponsesClient.Configuration(
    serverSideTools: [
        ResponsesClient.ServerSideTool(["type": "custom_tool", "option": "value"])
    ]
)
```

#### Background Mode

For long-running responses, enable background mode and manage them:

```swift
let config = ResponsesClient.Configuration(backgroundMode: true)

// Start a background response
let response = try await client.generateText(
    modelId: "gpt-5.2",
    messages: messages,
    apiKey: apiKey,
    configuration: config
)

// The client tracks the active background response ID
if let responseId = await client.activeBackgroundResponseId {
    // Check status of a background response
    let status = try await client.getBackgroundResponseStatus(responseId: responseId, apiKey: apiKey)
    // status.status: .queued, .in_progress, .completed, .failed, .cancelled
    // status.response: GenerationResponse? (when completed)

    // Cancel a background response
    try await client.cancelBackgroundResponse(responseId: responseId, apiKey: apiKey)

    // Delete a response permanently
    try await client.deleteResponse(responseId: responseId, apiKey: apiKey)

    // Resume streaming from a specific sequence number
    let resumed = try await client.resumeBackgroundStream(
        responseId: responseId,
        apiKey: apiKey,
        startingAfter: lastSequenceNumber
    ) { partial in
        print(partial.texts.response ?? "")
    }
}
```

### Google Gemini

```swift
let client = GeminiClient()

let config = GeminiClient.Configuration(
    safetyThreshold: .none,    // Safety filter threshold (.none, .high, .medium, .low)
    searchGrounding: true,     // Enable search grounding
    webContent: false,         // Enable web content fetching
    codeExecution: false,      // Enable code execution
    thinkingBudget: nil,       // Token budget for thinking (Gemini 2.5)
    thinkingLevel: .high       // .minimal (Flash), .low, .medium (Flash), .high
)

for try await partial in client.streamText(
    modelId: "gemini-3-pro-preview",
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 4096,
    apiKey: apiKey,
    configuration: config
) {
    // Access grounding/citations
    if let notes = partial.texts.notes {
        print("Sources: \(notes)")
    }
}
```

## Media Attachments

Send images, documents, video, or audio with your messages:

```swift
let imageData = try Data(contentsOf: imageURL)
let attachment = Attachment(
    kind: .image(data: imageData, mimeType: "image/jpeg"),
    filename: "photo.jpg"
)

let message = Message(
    role: .user,
    content: "What's in this image?",
    attachments: [attachment]
)

let response = try await generateText(
    model: .anthropic("claude-opus-4-5"),
    messages: [message],
    apiKey: apiKey
)
```

Supported attachment types:

```swift
// Image
.image(data: Data, mimeType: String)  // "image/jpeg", "image/png", etc.

// Document
.document(data: Data, mimeType: String)  // "application/pdf", etc.

// Video (Gemini)
.video(data: Data, mimeType: String)

// Audio (Gemini)
.audio(data: Data, mimeType: String)
```

## Tools

### Declarative Tools with `@Tool`

The simplest way to define tools is with the `@Tool` macro. Import `AITool` to use it:

```swift
import AI
import AITool

@Tool
struct GetWeather {
    static let name = "get_weather"
    static let title = "Get Weather"
    static let description = "Get the current weather for a location"

    @Parameter(title: "Location", description: "The city and country")
    var location: String

    @Parameter(title: "Units", description: "Temperature units: celsius or fahrenheit")
    var units: String?

    func perform() async throws -> String {
        // Call your weather API here
        return "72°F and sunny in \(location)"
    }
}
```

Use the tool with any provider:

```swift
let response = try await client.generateText(
    modelId: "claude-opus-4-5",
    tools: [GetWeather.tool],
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 1024,
    apiKey: apiKey
)
```

#### Parameter Constraints

Add validation constraints to parameters:

```swift
@Tool
struct SearchDocuments {
    static let name = "search_documents"
    static let title = "Search Documents"
    static let description = "Search documents by query"

    @Parameter(description: "Search query", minLength: 1, maxLength: 500)
    var query: String

    @Parameter(description: "Maximum results", minimum: 1, maximum: 100)
    var limit: Int = 10  // Default value

    func perform() async throws -> String {
        "Found \(limit) results for: \(query)"
    }
}
```

#### Supported Parameter Types

The `@Parameter` property wrapper supports these types:

- **Basic types**: `String`, `Int`, `Double`, `Bool`
- **Collections**: `Array<T>`, `Dictionary<String, T>` where T is a supported type
- **Temporal**: `Date` (parsed as ISO 8601 strings)
- **Binary**: `Data` (base64-encoded strings)
- **Optional**: `T?` for any supported type T
- **Enums**: Types conforming to `ToolEnum`

#### Enum Parameters

Use `ToolEnum` for parameters with a fixed set of values:

```swift
enum Priority: String, ToolEnum, CaseIterable {
    case low, medium, high
}

@Tool
struct SetPriority {
    static let name = "set_priority"
    static let title = "Set Priority"
    static let description = "Set task priority"

    @Parameter(description: "Priority level")
    var priority: Priority

    func perform() async throws -> String {
        "Priority set to \(priority.rawValue)"
    }
}
```

#### Rich Output Types

Besides `String`, tools can return other content types\. The return type automatically sets `resultTypes` for [capability filtering](#filtering-tools-by-capability).

- `ImageResult` → `resultTypes: [.image]`
- `AudioResult` → `resultTypes: [.audio]`
- `FileResult` → `resultTypes: [.file]`
- `MultiContent` → `resultTypes: nil` (determined at runtime)

```swift
// ImageResult
func perform() async throws -> ImageResult {
    let chartImage = renderChart(from: data)
    return ImageResult(pngData: chartImage)
}

// AudioResult
func perform() async throws -> AudioResult {
    let audioData = synthesizeSpeech(text: text)
    return AudioResult(data: audioData, mimeType: "audio/mpeg")
}

// FileResult
func perform() async throws -> FileResult {
    let pdfData = generateReport()
    return FileResult(data: pdfData, mimeType: "application/pdf", filename: "report.pdf")
}

// MultiContent
func perform() async throws -> MultiContent {
    MultiContent([
        .text("Analysis complete"),
        .image(chartData, mimeType: "image/png")
    ])
}
```

#### Strict Schema Validation

Enable strict schema validation to reject extra properties:

```swift
@Tool
struct StrictTool {
    static let name = "strict_tool"
    static let title = "Strict Tool"
    static let description = "A tool with strict schema validation"
    static let strictSchema = true  // Adds additionalProperties: false

    @Parameter(description: "Input value")
    var input: String

    func perform() async throws -> String {
        "Received: \(input)"
    }
}
```

### Executing Tool Calls with Tools Collection

The `Tools` collection provides automatic validation and concurrent execution:

```swift
let tools: Tools = [GetWeather.tool, SearchDocuments.tool]

// Execute all tool calls from a response concurrently
let results = await tools.call(response.toolCalls)

// Add results to conversation
messages.append(response.message)
messages.append(results.message)
```

The collection validates inputs against JSON Schema before execution and catches errors, returning them as error results rather than throwing. For custom validation, use the explicit initializer:

```swift
let tools = Tools([GetWeather.tool, SearchDocuments.tool], validator: customValidator)
```

### Agentic Loop

Run a conversation loop where the model can call tools and process results until it completes:

```swift
import AI

// Make tools (defined elsewhere) available
let tools: Tools = [GetWeather.tool, SearchDocuments.tool]

// Set up client and conversation
let client = AnthropicClient()
var messages: [Message] = [
    Message(role: .user, content: "What's the weather like at Chamonix this week? And find documents about alpine climbing safety.")
]

// Agentic loop
var iterations = 0
let maxIterations = 50

while iterations < maxIterations {
    iterations += 1
    let response = try await client.generateText(
        modelId: "claude-opus-4-5",
        tools: tools.definitions,
        systemPrompt: "You are an expert alpine guide assistant.",
        messages: messages,
        apiKey: apiKey
    )

    // Check if there are tool calls to execute
    if !response.toolCalls.isEmpty {
        // Add the assistant's response to conversation
        messages.append(response.message)

        // Execute all tool calls concurrently
        let results = await tools.call(response.toolCalls)

        // Add tool results to conversation
        messages.append(results.message)

        // Continue for the model to process results
        continue
    }

    // No tool calls – we have the final response
    print(response.texts.response ?? "No response")
    break
}
```

#### Combining Tools

Combine tools from different sources using the `+` operator or `adding()` methods:

```swift
// Using the + operator
let allTools = localTools + mcpTools
let combined = tools + [AnotherTool.tool]

// Using adding() methods
let expanded = tools.adding(AnotherTool.tool)
let merged = tools.adding(otherTools)
```

### Tool Result Types

Tools can return different types of content via `ToolResult.Content`:

```swift
// Text (supported by all providers)
.text("The weather is 72°F and sunny")

// Image (Anthropic, Gemini, Responses)
.image(imageData, mimeType: "image/png")

// Audio (Gemini only)
.audio(audioData, mimeType: "audio/wav")

// File (Gemini, Responses)
.file(fileData, mimeType: "application/pdf", filename: "report.pdf")
```

### Tool Error Handling

Errors thrown from `perform()` are automatically caught and returned as results with `isError: true`, providing feedback that models can use to self-correct and retry.

For clear, actionable error messages, use types conforming to `LocalizedError`:

```swift
@Tool
struct Translate {
    static let name = "translate"
    static let description = "Translate text into a specified language"

    @Parameter(description: "The text to translate")
    var text: String

    @Parameter(description: "The target language")
    var language: String

    func perform() async throws -> String {
        guard let apiKey = ProcessInfo.processInfo.environment["API_KEY"] else {
            throw TranslateError.missingAPIKey
        }
        // ... translation logic
    }

    enum TranslateError: LocalizedError {
        case missingAPIKey
        case unsupportedLanguage(String)

        var errorDescription: String? {
            switch self {
            case .missingAPIKey:
                "API_KEY not found in environment"
            case .unsupportedLanguage(let lang):
                "Unsupported language: \(lang)"
            }
        }
    }
}
```

Without `LocalizedError` conformance, the model sees generic messages like `"The operation couldn't be completed."` which aren't helpful for recovery.

For manual error results, use `ToolResult.error()` or set `isError: true`:

```swift
// Using convenience method
ToolResult.error("Failed to fetch weather data", name: call.name, id: call.id)

// Or manually
ToolResult(name: call.name, id: call.id, content: [.text("Error message")], isError: true)
```

### Client Capabilities

Each client type supports different result types:

| Client Type | Supported Result Types |
|----------|----------------------|
| Anthropic | text, image |
| Gemini | text, image, audio, file |
| Chat Completions | text |
| Responses | text, image, file |

When a tool returns an unsupported type, Swift AI automatically converts it to a model-legible fallback message (e.g., `[Unsupported result: image/png, 45 KB]`) and logs a warning.

### Filtering Tools by Capability

Filter tools based on what result types they produce and what clients support.

**For tools defined with `@Tool`**, `resultTypes` is automatically derived from the `perform()` return type:

```swift
@Tool
struct TakeScreenshot {
    static let name = "take_screenshot"
    static let title = "Take Screenshot"
    static let description = "Captures a screenshot"

    func perform() async throws -> ImageResult {  // resultTypes automatically set to [.image]
        ImageResult(pngData: captureScreen())
    }
}
```

**For imperative tools**, declare `resultTypes` explicitly:

```swift
let screenshotTool = Tool(
    name: "take_screenshot",
    description: "Captures a screenshot",
    parameters: [],
    resultTypes: [.image],
    execute: { _ in
        [.image(captureScreen(), mimeType: "image/png")]
    }
)
```

**Filter tools by client capabilities:**

```swift
let allTools = [TakeScreenshot.tool, weatherTool, calculatorTool]
let compatibleTools = allTools.compatible(with: ChatCompletionsClient.self)
// Excludes TakeScreenshot since ChatCompletions only supports text
```

## Imperative Tool Usage

For tools defined dynamically at runtime, or when you need full control over tool execution, you can work with the lower-level APIs directly.

### Defining Tools Imperatively

```swift
let weatherTool = Tool(
    name: "get_weather",
    description: "Get the current weather for a location",
    title: "Weather",
    parameters: [
        .string("location", title: "Location", description: "The city and country"),
        .string("units", description: "Temperature units: celsius or fahrenheit", required: false)
    ],
    execute: { params in
        let location = params["location"]?.stringValue ?? "Unknown"
        // Call your weather API here
        return [.text("72°F and sunny in \(location)")]
    }
)
```

### Manual Tool Execution

When you need full control over the tool execution loop:

```swift
let client = AnthropicClient()
let response = try await client.generateText(
    modelId: modelId,
    tools: [weatherTool],
    systemPrompt: systemPrompt,
    messages: messages,
    maxTokens: 1024,
    apiKey: apiKey
)

// Check if the model wants to call a tool
if !response.toolCalls.isEmpty {
    for call in response.toolCalls {
        print("Tool: \(call.name), ID: \(call.id)")
        print("Parameters: \(call.parameters)")
    }
}
```

### Constructing Tool Results

After executing a tool manually, construct the result and continue the conversation:

```swift
let toolResult = ToolResult(
    name: "get_weather",
    id: call.id,
    content: .text("72°F and sunny")
)

let assistantMessage = Message(
    role: .assistant,
    content: nil,
    toolCalls: response.toolCalls
)

let toolMessage = Message(
    role: .tool,
    content: nil,
    toolResults: [toolResult]
)

// Continue the conversation with tool results
let followUp = try await client.generateText(
    messages: messages + [assistantMessage, toolMessage],
    // ... other parameters
)
```

## MCP Integration

The `AIMCP` module bridges Swift AI with the [Model Context Protocol](https://github.com/anthropics/swift-mcp) (MCP), allowing you to use MCP tools with any AI provider.

### Setup

Add both dependencies to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-ai", branch: "main"),
    .package(url: "https://github.com/DePasqualeOrg/swift-mcp", branch: "main"),
]
```

### Using MCP Tools

Connect to an MCP server and use its tools with any AI provider:

```swift
import AIMCP  // AI and MCP bridging, also imports those packages

// Create and connect to an MCP server
let mcpClient = MCP.Client(name: "MyApp", version: "1.0.0")
let transport = StdioTransport()  // Or HTTPClientTransport, etc.
try await mcpClient.connect(transport: transport)

// Create interface between MCP tools and AI tools
let toolProvider = MCPToolProvider(client: mcpClient)

// Get tools for AI (namespaced by default: "servername__toolname")
let tools = try await toolProvider.tools()

// Or get tools without namespace prefix:
// let tools = try await toolProvider.tools(namespaced: false)

// Use MCP tools with any provider
let response = try await client.generateText(
    modelId: "claude-opus-4-5",
    tools: tools.definitions,
    systemPrompt: "You are an expert alpine guide assistant.",
    messages: messages,
    apiKey: apiKey
)

// Execute tool calls through MCP
if !response.toolCalls.isEmpty {
    let results = try await toolProvider.execute(response.toolCalls)
    // Continue conversation with results.message
}
```

MCP tools work with the same [agentic loop pattern](#agentic-loop) shown in the Tools section.

### Multiple MCP Servers

MCPToolProvider supports multiple servers with automatic namespacing:

```swift
// Connect to multiple MCP servers
let filesystemClient = MCP.Client(name: "MyApp", version: "1.0.0")
try await filesystemClient.connect(transport: filesystemTransport)

let githubClient = MCP.Client(name: "MyApp", version: "1.0.0")
try await githubClient.connect(transport: githubTransport)

// Create provider with multiple clients
let toolProvider = MCPToolProvider(clients: [filesystemClient, githubClient])

// Tools are namespaced: "filesystem__read_file", "github__create_issue", etc.
let tools = try await toolProvider.tools()

// Get a specific tool by name
if let readFile = try await toolProvider.tool(named: "filesystem__read_file") {
    // Use the tool directly
}

// Clear cached tools to refresh from servers
await toolProvider.clearCache()

// Get connected server names
let serverNames = await toolProvider.connectedServerNames()
```

### Direct Conversions

For more control, use the conversion extensions directly:

```swift
import AIMCP  // AI and MCP bridging, also imports those packages

// Convert AI Tool to MCP Tool
let mcpTool = MCP.Tool(from: aiTool)

// Convert MCP Tool to AI Tool (executes via MCP client)
let aiTool = try AI.Tool(from: mcpTool, client: mcpClient)

// Convert MCP Tool to AI Tool (with custom executor, e.g. for mocking)
let aiTool = try AI.Tool(from: mcpTool) { parameters in
    // Your execution logic
    return [.text("result")]
}

// Batch conversions
let mcpTools = aiTools.mcpTools
let aiTools = try mcpTools.aiTools(client: mcpClient)
```

## Error Handling

Swift AI provides unified error handling across all providers:

```swift
do {
    let response = try await generateText(model: .anthropic("claude-opus-4-5"), ...)
} catch let error as AIError {
    switch error {
        case .authentication(let message):
            print("Auth failed: \(message)")
        case .rateLimit(let retryAfter):
            if let delay = retryAfter {
                print("Rate limited. Retry after \(delay) seconds")
            }
        case .serverError(let statusCode, let message, let context):
            print("Server error \(statusCode): \(message)")
            // Access provider-specific details via context?.providerInfo
        case .invalidRequest(let message):
            print("Invalid request: \(message)")
        case .parsing(let message):
            print("Parsing error: \(message)")
        case .network(let underlying):
            print("Network error: \(underlying.localizedDescription)")
        case .cancelled:
            print("Request was cancelled")
        case .timeout:
            print("Request timed out")
    }

    // Check if error is retryable
    if error.isRetryable {
        // Implement retry logic
    }
}
```

## Custom Endpoints and Sessions

All clients support custom endpoints and URL sessions:

```swift
// Using top-level functions with custom endpoint
let response = try await generateText(
    model: .chatCompletions("llama-3", endpoint: URL(string: "http://localhost:8080/v1/chat/completions")!),
    messages: messages,
    apiKey: nil  // Local endpoints may not need a key
)

// ChatCompletionsClient with custom endpoint and session
let client = ChatCompletionsClient(
    endpoint: URL(string: "https://your-endpoint/v1/chat/completions")!,
    session: customURLSession
)

// ResponsesClient with custom endpoint
let client = ResponsesClient(
    endpoint: URL(string: "https://your-endpoint/v1/responses")!,
    session: customURLSession
)

// GeminiClient with custom models endpoint
// Uses a default session with no timeout for long-running thinking requests
let client = GeminiClient(
    session: GeminiClient.defaultSession,  // URLSession with no timeout
    modelsEndpoint: URL(string: "https://custom-gemini-endpoint/v1beta/models")
)

// AnthropicClient with retry and timeout configuration
let client = AnthropicClient(
    maxRetries: 3,
    timeout: 300,  // 5 minutes
    session: customURLSession,
    messagesEndpoint: URL(string: "https://custom-anthropic-endpoint/v1/messages")
)
```

## Cancellation

Cancel an in-progress generation:

```swift
await client.stop()
```

All clients expose `isGenerating` for UI state:

```swift
struct ChatView: View {
    let client = AnthropicClient()

    var body: some View {
        VStack {
            if client.isGenerating {
                // Progress indicator
            }
        }
    }
}
```
