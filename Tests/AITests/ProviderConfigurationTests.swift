// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ProviderConfigurationTests {
  // MARK: - Mismatch Error Tests (generateText)

  @Test
  func `Anthropic model rejects Responses configuration`() async {
    await #expect(throws: AIError.self) {
      _ = try await generateText(
        model: .anthropic("claude-haiku-4-5-20251001"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: .responses(.init(reasoningEffortLevel: .medium)),
      )
    }
  }

  @Test
  func `Gemini model rejects Anthropic configuration`() async {
    await #expect(throws: AIError.self) {
      _ = try await generateText(
        model: .gemini("gemini-2.5-flash"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: .anthropic(.init(effort: .high)),
      )
    }
  }

  @Test
  func `Responses model rejects Gemini configuration`() async {
    await #expect(throws: AIError.self) {
      _ = try await generateText(
        model: .responses("gpt-5.1"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: .gemini(.init(thinkingLevel: .high)),
      )
    }
  }

  @Test
  func `Chat Completions model rejects Responses configuration`() async {
    await #expect(throws: AIError.self) {
      _ = try await generateText(
        model: .chatCompletions("gpt-4o-mini"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: .responses(.init(reasoningEffortLevel: .high)),
      )
    }
  }

  // MARK: - Mismatch Error Tests (streamText)

  @Test
  func `streamText with mismatched configuration fails`() async {
    let stream = streamText(
      model: .anthropic("claude-haiku-4-5-20251001"),
      messages: [Message(role: .user, content: "test")],
      apiKey: "test-key",
      configuration: .responses(.init(reasoningEffortLevel: .medium)),
    )
    await #expect(throws: AIError.self) {
      _ = try await consumeStream(stream)
    }
  }

  @Test
  func `streamText prompt overload with mismatched configuration fails`() async {
    let stream = streamText(
      model: .gemini("gemini-2.5-flash"),
      prompt: "test",
      apiKey: "test-key",
      configuration: .anthropic(.init(effort: .high)),
    )
    await #expect(throws: AIError.self) {
      _ = try await consumeStream(stream)
    }
  }

  // MARK: - Nil Configuration (no mismatch)

  @Test
  func `Nil configuration does not throw mismatch error`() async {
    // With no configuration, the function should proceed to the network call
    // (which will fail since we have no mock, but it won't be a mismatch error).
    // We verify the error is NOT an invalidRequest mismatch.
    do {
      _ = try await generateText(
        model: .anthropic("claude-haiku-4-5-20251001"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: nil,
      )
      // If it somehow succeeds, that's fine too
    } catch let error as AIError {
      if case let .invalidRequest(message) = error {
        #expect(!message.contains("Configuration mismatch"), "Should not get a mismatch error with nil configuration")
      }
      // Other AIError types (network, etc.) are expected
    } catch {
      // Non-AIError failures (network, etc.) are expected
    }
  }

  @Test
  func `Matching provider configuration does not throw mismatch error`() async {
    // Responses config with Responses model should not throw a mismatch error.
    // It will fail at the network level since there's no mock, but not with invalidRequest.
    do {
      _ = try await generateText(
        model: .responses("gpt-5.1"),
        messages: [Message(role: .user, content: "test")],
        apiKey: "test-key",
        configuration: .responses(.init(reasoningEffortLevel: .medium)),
      )
    } catch let error as AIError {
      if case let .invalidRequest(message) = error {
        #expect(!message.contains("Configuration mismatch"), "Should not get a mismatch error with matching configuration")
      }
    } catch {
      // Non-AIError failures (network, etc.) are expected
    }
  }

  @Test
  func `Default Responses configuration infers provider from built in endpoints`() throws {
    #expect(
      try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: ResponsesClient.Endpoint.xAI.url,
        provider: nil,
      ).serverSideTools == [.xAI.webSearch()],
    )

    #expect(
      try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: ResponsesClient.Endpoint.openAI.url,
        provider: nil,
      ).serverSideTools == [.OpenAI.webSearch(contextSize: .medium)],
    )
  }

  @Test
  func `Default Responses configuration rejects conflicting provider for built in endpoints`() {
    do {
      _ = try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: ResponsesClient.Endpoint.xAI.url,
        provider: .openAI,
      )
      Issue.record("Expected conflicting provider for built-in Responses endpoint to throw")
    } catch let error as AIError {
      guard case let .invalidRequest(message) = error else {
        Issue.record("Expected invalidRequest but got \(error)")
        return
      }
      #expect(message.contains("conflicts"))
      #expect(message.contains("`.xAI`"))
    } catch {
      Issue.record("Expected AIError.invalidRequest but got \(error)")
    }
  }

  @Test
  func `Default Responses configuration uses explicit provider for custom endpoints`() throws {
    let customEndpoint = try #require(URL(string: "https://proxy.example.test/v1/responses"))

    #expect(
      try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: customEndpoint,
        provider: .xAI,
      ).serverSideTools == [.xAI.webSearch()],
    )

    #expect(
      try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: customEndpoint,
        provider: .openAI,
      ).serverSideTools == [.OpenAI.webSearch(contextSize: .medium)],
    )
  }

  @Test
  func `Default Responses configuration rejects ambiguous custom webSearch endpoints`() throws {
    let customEndpoint = try #require(URL(string: "https://proxy.example.test/v1/responses"))

    do {
      _ = try defaultResponsesConfiguration(
        webSearch: true,
        endpoint: customEndpoint,
        provider: nil,
      )
      Issue.record("Expected custom Responses webSearch endpoint without provider to throw")
    } catch let error as AIError {
      guard case let .invalidRequest(message) = error else {
        Issue.record("Expected invalidRequest but got \(error)")
        return
      }
      #expect(message.contains("responsesProvider"))
    } catch {
      Issue.record("Expected AIError.invalidRequest but got \(error)")
    }
  }
}
