// Minimal agentic loop example
// Run with: swift run AgenticLoop

import AI
import AITool
import ExamplesShared
import Foundation

// MARK: - Errors

// Throwing a LocalizedError gives the model helpful information on what went wrong so that it can recover.

enum RandomSelectionError: LocalizedError {
  case emptyCollection(String)

  var errorDescription: String? {
    switch self {
      case let .emptyCollection(name): "No \(name) available to pick from"
    }
  }
}

enum EnvironmentError: LocalizedError {
  case missingKey(String)

  var errorDescription: String? {
    switch self {
      case let .missingKey(key): "\(key) not found in environment"
    }
  }
}

enum TranslateError: LocalizedError {
  case emptyResponse

  var errorDescription: String? {
    switch self {
      case .emptyResponse: "Model returned an empty response"
    }
  }
}

// MARK: - Tools

@Tool
struct PickLanguage {
  static let name = "pick_language"
  static let description = "Pick a random language."

  private static let languages = [
    "Spanish",
    "French",
    "Japanese",
    "German",
    "Italian",
    "Portuguese",
    "Mandarin Chinese",
  ]

  func perform() async throws -> String {
    guard let language = Self.languages.randomElement() else {
      throw RandomSelectionError.emptyCollection("languages")
    }
    return language
  }
}

@Tool
struct GetFortune {
  static let name = "get_fortune"
  static let description = "Get a fortune cookie message."

  private static let fortunes = [
    "A journey of a thousand miles begins with a single step.",
    "Your kindness will lead you to unexpected opportunities.",
    "The best time to plant a tree was 20 years ago. The second best time is now.",
    "A smile is your passport into the hearts of others.",
    "Your hard work will pay off sooner than you think.",
    "Adventure awaits you at the turn of the next corner.",
    "The secret to getting ahead is getting started.",
    "Good things come to those who wait, but better things come to those who act.",
  ]

  func perform() async throws -> String {
    guard let fortune = Self.fortunes.randomElement() else {
      throw RandomSelectionError.emptyCollection("fortunes")
    }
    return fortune
  }
}

@Tool
struct Translate {
  static let name = "translate"
  static let description = "Translate text into a specified language."

  @Parameter(description: "The text to translate")
  var text: String

  @Parameter(description: "The target language")
  var language: String

  func perform() async throws -> String {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["ANTHROPIC_API_KEY"] else {
      throw EnvironmentError.missingKey("ANTHROPIC_API_KEY")
    }

    let response = try await AnthropicClient().generateText(
      modelId: "claude-opus-4-5",
      systemPrompt: "Translate the following text to \(language). Return only the translation, nothing else.",
      prompt: text,
      apiKey: apiKey
    )
    guard let translation = response.texts.response else {
      throw TranslateError.emptyResponse
    }
    return translation
  }
}

// MARK: - Main

@main
enum AgenticLoopExample {
  static func main() async throws {
    let env = try EnvLoader.loadFromPackageRoot()
    guard let apiKey = env["GEMINI_API_KEY"] else {
      print("Missing GEMINI_API_KEY")
      return
    }

    // Create client, select tools,
    let client = GeminiClient()
    let tools = [PickLanguage.tool, GetFortune.tool, Translate.tool]

    let prompt = "Get a fortune cookie in a random language."
    print("Prompt: \(prompt)")
    print()

    // Conversation history for multi-turn use
    var messages = [Message(role: .user, content: prompt)]
    var iteration = 0
    let maxIterations = 50

    // Agentic loop: call model, execute tools, repeat until done
    while iteration < maxIterations {
      iteration += 1
      print("--- Iteration \(iteration) ---")
      print()
      print("[Calling model...]")

      // Call model with tools and conversation history
      let response = try await client.generateText(
        modelId: "gemini-3-flash-preview",
        tools: tools,
        messages: messages,
        apiKey: apiKey
      )

      // No tool calls means model is done
      if response.toolCalls.isEmpty {
        print("[Model returned final response]")
        print()
        print(response.texts.response ?? "")
        print()
        break
      }

      print("[Model requested \(response.toolCalls.count) tool \(response.toolCalls.count == 1 ? "call" : "calls")]")
      print()

      // Add assistant message (with tool calls) to history
      messages.append(response.message)

      for call in response.toolCalls {
        print("Tool call: \(call.name)")
        print("Arguments: \(call.parameters)")
        print()
      }

      print("[Calling \(response.toolCalls.count == 1 ? "tool" : "tools in parallel")...]")
      let results = await Tools(tools).call(response.toolCalls)
      print()

      for result in results {
        let text = result.content.map { $0.fallbackDescription }.joined()
        print("Tool result (\(result.name)): \(text)")
      }
      print()

      // Add tool results to history and continue loop
      messages.append(results.message)
    }

    if iteration >= maxIterations {
      print("[Reached maximum iterations (\(maxIterations))]")
    }
  }
}
