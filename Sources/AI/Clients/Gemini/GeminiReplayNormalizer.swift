// Copyright © Anthony DePasquale

import Foundation

enum GeminiReplayNormalizer {
  struct Plan {
    let systemParts: [[String: any Sendable]]
    let contents: [[String: any Sendable]]
  }

  static func normalize(
    _ messages: [Message],
    systemPrompt: String?,
    apiKey: String,
    requestParts: (Message, String) async throws -> [[String: any Sendable]],
  ) async throws -> Plan {
    var systemParts: [[String: any Sendable]] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      systemParts.append(["text": systemPrompt])
    }

    var contents: [[String: any Sendable]] = []
    let replayPlan = ReplayNormalizer.normalize(messages, profile: .gemini)
    for message in replayPlan.messages {
      switch message.role {
        case .system, .developer:
          systemParts.append(contentsOf: GeminiClient.systemInstructionParts(for: message))
        case .assistant, .user, .tool:
          let parts = try await requestParts(message, apiKey)
          guard !parts.isEmpty else { continue }
          let role = switch message.role {
            case .assistant: "model"
            case .tool: "user"
            default: message.role.rawValue
          }
          contents.append([
            "role": role,
            "parts": parts,
          ])
      }
    }

    return Plan(systemParts: systemParts, contents: contents)
  }
}
