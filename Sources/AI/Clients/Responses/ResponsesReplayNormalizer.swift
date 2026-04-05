// Copyright © Anthony DePasquale

import Foundation

enum ResponsesReplayNormalizer {
  struct Plan {
    let inputItems: [[String: any Sendable]]
  }

  static func normalize(_ messages: [Message]) async throws -> Plan {
    var inputItems: [[String: any Sendable]] = []
    let replayPlan = ReplayNormalizer.normalize(messages, profile: .responses)
    for message in replayPlan.messages {
      try await inputItems.append(contentsOf: ResponsesClient.inputItems(for: message))
    }
    return Plan(inputItems: inputItems)
  }
}
