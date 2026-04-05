// Copyright © Anthony DePasquale

import Foundation

enum ChatCompletionsReplayNormalizer {
  struct Plan {
    let messages: [Message]
  }

  static func normalize(_ messages: [Message]) -> Plan {
    Plan(messages: ReplayNormalizer.normalize(messages, profile: .chatCompletions).messages)
  }
}
