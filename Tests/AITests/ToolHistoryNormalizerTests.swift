// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct ToolHistoryNormalizerTests {
  @Test
  func `Tool history normalizer repairs late tool results with synthetic output plus collapsed text`() throws {
    let messages = ReplayFixtures.lateToolResultHistory()

    let normalized = ToolHistoryNormalizer.normalize(messages)

    #expect(normalized.count == 4)
    #expect(normalized[0] == messages[0])
    #expect(normalized[2] == messages[1])

    let syntheticResult = try #require(normalized[1].content.first)
    guard case let .toolResult(toolResult) = syntheticResult else {
      Issue.record("Expected a synthetic tool result")
      return
    }
    #expect(toolResult.id == "call_1")
    #expect(toolResult.isError == true)
    #expect(toolResult.content == [.text(ToolReplaySupport.syntheticToolResultErrorText)])

    let collapsedText = normalized[3].replayableTextSegments().joined(separator: "\n\n")
    #expect(normalized[3].role == .user)
    #expect(collapsedText.contains(ReplayFixtures.lateToolResultText))
  }

  @Test
  func `Tool history normalizer splits mixed tool turns and preserves unresolved siblings`() throws {
    let messages = ReplayFixtures.mixedToolTurnHistory()

    let normalized = ToolHistoryNormalizer.normalize(messages)

    #expect(normalized.count == 5)
    #expect(normalized[0] == messages[0])
    #expect(normalized[4] == messages[2])

    let matchedResult = try #require(normalized[1].content.first)
    guard case let .toolResult(matchedToolResult) = matchedResult else {
      Issue.record("Expected the matched tool result to remain native")
      return
    }
    #expect(matchedToolResult.id == "call_1")
    #expect(matchedToolResult.content == [.text(ReplayFixtures.matchedToolResultText)])

    let syntheticResult = try #require(normalized[2].content.first)
    guard case let .toolResult(syntheticToolResult) = syntheticResult else {
      Issue.record("Expected a synthetic tool result for the unresolved sibling")
      return
    }
    #expect(syntheticToolResult.id == "call_2")
    #expect(syntheticToolResult.isError == true)

    let collapsedText = normalized[3].replayableTextSegments().joined(separator: "\n\n")
    #expect(normalized[3].role == .user)
    #expect(collapsedText.contains(ReplayFixtures.strayToolResultText))
  }

  @Test
  func `Tool history normalizer synthesizes trailing unresolved tool calls`() throws {
    let messages = ReplayFixtures.trailingUnresolvedToolCallHistory()

    let normalized = ToolHistoryNormalizer.normalize(messages)

    #expect(normalized.count == 2)
    let syntheticResult = try #require(normalized[1].content.first)
    guard case let .toolResult(toolResult) = syntheticResult else {
      Issue.record("Expected a synthetic tool result")
      return
    }
    #expect(toolResult.id == "call_1")
    #expect(toolResult.isError == true)
  }
}
