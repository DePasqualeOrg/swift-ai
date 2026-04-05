// Copyright © Anthony DePasquale

import Foundation

enum AnthropicReplayNormalizer {
  struct Plan {
    let systemTexts: [String]
    let messages: [AnthropicClient.MessageParam]
  }

  static func normalize(_ messages: [Message], thinkingEnabled: Bool) async throws -> Plan {
    let profile = ReplayProviderProfile.forAnthropic(thinkingEnabled: thinkingEnabled)
    let repairedMessages = ReplayNormalizer.normalize(messages, profile: profile).messages
    let normalizedMessages = if profile.reasoning.collapsesToolExchangesWithoutNativeReasoning {
      collapseThinkingHistory(in: repairedMessages)
    } else {
      repairedMessages
    }
    var systemTexts: [String] = []
    var messageParams: [AnthropicClient.MessageParam] = []

    for message in normalizedMessages {
      if message.role == .system || message.role == .developer {
        systemTexts.append(contentsOf: AnthropicClient.systemInstructionTexts(for: message))
        continue
      }

      let contentBlocks = try await AnthropicClient.anthropicContentBlocks(for: message)
      guard !contentBlocks.isEmpty else { continue }
      messageParams.append(AnthropicClient.MessageParam(
        role: AnthropicClient.mapRole(message.role),
        text: nil,
        contentBlocks: contentBlocks,
        attachments: nil,
      ))
    }

    return Plan(systemTexts: systemTexts, messages: messageParams)
  }

  private static func collapseThinkingHistory(in messages: [Message]) -> [Message] {
    var normalizedMessages: [Message] = []
    var skipUntilIndex = 0

    for (index, message) in messages.enumerated() {
      if index < skipUntilIndex {
        continue
      }

      if message.role == .assistant, message.hasToolCalls, !message.hasNativeAnthropicThinkingBlocks {
        normalizedMessages.append(message.collapsingToolCalls())

        var nextIndex = index + 1
        while nextIndex < messages.count, messages[nextIndex].hasToolResults {
          normalizedMessages.append(messages[nextIndex].collapsingToolResults())
          nextIndex += 1
        }
        skipUntilIndex = nextIndex
      } else {
        normalizedMessages.append(message)
      }
    }

    return normalizedMessages
  }
}
