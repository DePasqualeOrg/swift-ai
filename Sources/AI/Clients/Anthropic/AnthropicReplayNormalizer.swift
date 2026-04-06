// Copyright © Anthony DePasquale

import Foundation

enum AnthropicReplayNormalizer {
  struct Plan {
    let systemTexts: [String]
    let messages: [AnthropicClient.MessageParam]
  }

  static func normalize(_ messages: [Message], thinkingEnabled: Bool) async throws -> Plan {
    let profile = ReplayProviderProfile.forAnthropic(thinkingEnabled: thinkingEnabled)
    let normalizedMessages = ReplayNormalizer.normalize(messages, profile: profile).messages
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
}
