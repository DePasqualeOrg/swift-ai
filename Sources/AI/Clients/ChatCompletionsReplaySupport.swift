// Copyright © Anthony DePasquale

import Foundation

enum ChatCompletionsReplaySupport {
  static func patchingOrphanedToolCalls(_ messages: [Message]) -> [Message] {
    let toolCallIDs = Set(messages.flatMap(\.toolCalls).map(\.id))
    let toolResultIDs = Set(messages.flatMap { message in
      message.content.compactMap { item -> String? in
        guard case let .toolResult(toolResult) = item else { return nil }
        return toolResult.id
      }
    })
    let orphanedIDs = toolCallIDs.subtracting(toolResultIDs)
    guard !orphanedIDs.isEmpty else { return messages }

    var patchedMessages: [Message] = []
    for message in messages {
      patchedMessages.append(message)
      let orphanedToolCalls = message.toolCalls.filter { orphanedIDs.contains($0.id) }
      if let syntheticResultMessage = ToolReplaySupport.syntheticToolResultMessage(for: orphanedToolCalls) {
        patchedMessages.append(syntheticResultMessage)
      }
    }

    return patchedMessages
  }
}
