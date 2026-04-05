// Copyright © Anthony DePasquale

import Foundation

extension AnthropicClient {
  static func generationResponse(from message: APIMessage) -> GenerationResponse {
    GenerationResponse(content: blocks(from: message), metadata: metadata(from: message))
  }

  static func systemInstructionTexts(for message: Message) -> [String] {
    message.replayableTextSegmentsWithAttachmentFallback()
  }

  private static func metadata(from message: APIMessage?) -> GenerationResponse.Metadata? {
    guard let message else { return nil }

    let finishReason: GenerationResponse.FinishReason? = switch message.stopReason {
      case "end_turn", "stop_sequence": .stop
      case "max_tokens": .maxTokens
      case "tool_use": .toolUse
      case "refusal": .refusal
      case "pause_turn": .pauseTurn
      case .some: .other
      case .none: nil
    }

    return GenerationResponse.Metadata(
      responseId: message.id,
      finishReason: finishReason,
      inputTokens: message.usage.inputTokens,
      outputTokens: message.usage.outputTokens,
      cacheCreationInputTokens: message.usage.cacheCreationInputTokens,
      cacheReadInputTokens: message.usage.cacheReadInputTokens,
    )
  }

  private static func toolCall(from toolUseBlock: ToolUseBlock) -> ToolCall {
    var parameters: [String: Value] = [:]
    if case let .object(dict) = toolUseBlock.input {
      for (key, value) in dict where key != Value.jsonBufKey {
        parameters[key] = value
      }
    }
    return .init(name: toolUseBlock.name, id: toolUseBlock.id, parameters: parameters)
  }

  private static func textBlock(from serverToolUse: ServerToolUseBlock) -> Message.Content? {
    guard serverToolUse.name == "code_execution",
          case let .object(inputDict) = serverToolUse.input,
          case let .string(code) = inputDict["code"]
    else {
      return nil
    }
    return .text("\n\n```python\n\(code)\n```\n\n")
  }

  private static func textBlocks(from codeExecutionResultBlock: CodeExecutionToolResultBlock) -> [Message.Content] {
    var blocks: [Message.Content] = []

    for item in codeExecutionResultBlock.content {
      let text: String?
      switch item {
        case let .result(resultDetails):
          var resultText = ""
          if !resultDetails.stdout.isEmpty {
            resultText += "\n\n```\n\(resultDetails.stdout)\(resultDetails.stdout.hasSuffix("\n") ? "" : "\n")```\n\n"
          }
          if !resultDetails.stderr.isEmpty {
            resultText += "\n\n**Error:**\n```\n\(resultDetails.stderr)\(resultDetails.stderr.hasSuffix("\n") ? "" : "\n")```\n\n"
          }
          if resultDetails.returnCode != 0 || !resultDetails.stderr.isEmpty {
            if resultText.isEmpty {
              resultText += "\n\n"
            }
            resultText += "```Exit code: \(resultDetails.returnCode)```\n\n"
          }
          text = resultText.isEmpty ? nil : resultText
        case let .encryptedResult(resultDetails):
          var resultText = ""
          if !resultDetails.stderr.isEmpty {
            resultText += "\n\n**Error:**\n```\n\(resultDetails.stderr)\(resultDetails.stderr.hasSuffix("\n") ? "" : "\n")```\n\n"
          }
          if resultDetails.returnCode != 0 || !resultDetails.stderr.isEmpty {
            if resultText.isEmpty {
              resultText += "\n\n"
            }
            resultText += "```Exit code: \(resultDetails.returnCode)```\n\n"
          }
          text = resultText.isEmpty ? nil : resultText
        case let .error(errorDetails):
          text = "\n\n**Code execution error:** \(errorDetails.errorCode)\n\n"
        case let .output(outputBlock):
          text = "\n\n**File Output Generated:** `\(outputBlock.fileId)` (Content not displayed)\n\n"
      }

      if let text, !text.isEmpty {
        blocks.append(.text(text))
      }
    }

    return blocks
  }

  private static func toolResultContents(from content: ToolResultContent) -> [ToolResult.Content] {
    switch content {
      case let .text(text):
        [.text(text)]
      case let .blocks(blocks):
        blocks.flatMap(toolResultContents(from:))
    }
  }

  private static func toolResultContents(from block: ToolResultContentBlock) -> [ToolResult.Content] {
    switch block.type {
      case "text":
        if let text = block.text {
          return [.text(text)]
        }
      case "image":
        if let source = block.source {
          switch source.type {
            case "base64":
              if let data = source.data.flatMap({ Data(base64Encoded: $0) }) {
                return [.image(data, mimeType: source.mediaType)]
              }
            case "url":
              if let url = source.url, !url.isEmpty {
                return [.text("[Image URL: \(url)]")]
              }
            default:
              break
          }
        }
      case "document":
        if let source = block.source {
          switch source.type {
            case "base64":
              if let data = source.data.flatMap({ Data(base64Encoded: $0) }) {
                return [.file(data, mimeType: source.mediaType)]
              }
            case "text":
              if let data = source.data?.data(using: .utf8) {
                return [.file(data, mimeType: source.mediaType)]
              }
            case "url":
              if let url = source.url, !url.isEmpty {
                return [.text("[Document URL: \(url)]")]
              }
            default:
              break
          }
        }
      case "search_result":
        var parts: [String] = []
        if let title = block.title, !title.isEmpty {
          parts.append(title)
        }
        if let source = block.searchResultSource, !source.isEmpty {
          parts.append(source)
        }
        let snippets = block.contentBlocks?
          .compactMap(\.text)
          .filter { !$0.isEmpty }
          .joined(separator: "\n\n")
        if let snippets, !snippets.isEmpty {
          parts.append(snippets)
        }
        if !parts.isEmpty {
          return [.text(parts.joined(separator: "\n"))]
        }
      case "tool_reference":
        if let toolName = block.toolName, !toolName.isEmpty {
          return [.text("[Tool reference: \(toolName)]")]
        }
        return [.text("[Tool reference]")]
      default:
        break
    }

    if let text = block.text, !text.isEmpty {
      return [.text(text)]
    }
    if let jsonString = toolResultBlockJSONString(from: block.raw) {
      return [.text(jsonString)]
    }
    return []
  }

  private static func toolResultBlockJSONString(from raw: [String: Value]) -> String? {
    guard let data = try? JSONSerialization.data(withJSONObject: Value.toSendable(raw)),
          let jsonString = String(data: data, encoding: .utf8)
    else {
      return nil
    }
    return jsonString
  }

  private static func blocks(from message: APIMessage) -> [Message.Content] {
    var blocks: [Message.Content] = []
    var citationURLs: [String] = []
    var seenCitationURLs = Set<String>()

    func appendCitationURL(_ url: String) {
      guard seenCitationURLs.insert(url).inserted else { return }
      citationURLs.append(url)
    }

    for contentBlock in message.content {
      switch contentBlock.type {
        case .text:
          if let text = contentBlock.text, !text.isEmpty {
            blocks.append(.text(text))
          }
          for citation in contentBlock.citations ?? [] {
            if let url = citation.url {
              appendCitationURL(url)
            }
          }
        case .thinking:
          if let thinking = contentBlock.thinking {
            blocks.append(.thinking(text: thinking, signature: contentBlock.signature))
          }
        case .redactedThinking:
          if let data = contentBlock.data {
            blocks.append(.redactedThinking(data: data))
          }
        case .toolUse:
          if let toolUse = contentBlock.toolUse {
            blocks.append(.toolCall(toolCall(from: toolUse)))
          }
        case .toolResult:
          if let toolResult = contentBlock.toolResult {
            blocks.append(.toolResult(.init(
              name: "",
              id: toolResult.toolUseId,
              content: toolResultContents(from: toolResult.content),
              isError: toolResult.isError,
            )))
          }
        case .serverToolUse:
          var displayText: String?
          if let serverToolUse = contentBlock.serverToolUse,
             let textBlock = textBlock(from: serverToolUse),
             case let .text(text) = textBlock
          {
            displayText = text
          }
          if let jsonData = try? JSONEncoder().encode(contentBlock),
             let jsonString = String(data: jsonData, encoding: .utf8)
          {
            blocks.append(.providerOpaque(OpaqueBlock(
              provider: "anthropic",
              type: contentBlock.type.rawValue,
              content: displayText,
              data: jsonString,
              isResponseContent: displayText != nil,
            )))
          }
        case .codeExecutionToolResult:
          let displayText: String? = if let codeExecutionToolResult = contentBlock.codeExecutionToolResult {
            textBlocks(from: codeExecutionToolResult).compactMap { block -> String? in
              if case let .text(text) = block { return text }
              return nil
            }.joined()
          } else {
            nil
          }
          if let jsonData = try? JSONEncoder().encode(contentBlock),
             let jsonString = String(data: jsonData, encoding: .utf8)
          {
            blocks.append(.providerOpaque(OpaqueBlock(
              provider: "anthropic",
              type: contentBlock.type.rawValue,
              content: displayText,
              data: jsonString,
              isResponseContent: displayText != nil,
            )))
          }
        case .webSearchToolResult:
          if let jsonData = try? JSONEncoder().encode(contentBlock),
             let jsonString = String(data: jsonData, encoding: .utf8)
          {
            blocks.append(.providerOpaque(OpaqueBlock(
              provider: "anthropic",
              type: contentBlock.type.rawValue,
              data: jsonString,
            )))
          }
          if let webSearchToolResult = contentBlock.webSearchToolResult,
             case let .results(items) = webSearchToolResult.content
          {
            for item in items {
              appendCitationURL(item.url)
            }
          }
        case .webFetchToolResult:
          var displayText: String?
          if let webFetchToolResult = contentBlock.webFetchToolResult {
            if case let .result(result) = webFetchToolResult.content {
              appendCitationURL(result.url)
              if result.content.source.type == "text", !result.content.source.data.isEmpty {
                displayText = result.content.source.data
              }
            }
          }
          if let jsonData = try? JSONEncoder().encode(contentBlock),
             let jsonString = String(data: jsonData, encoding: .utf8)
          {
            blocks.append(.providerOpaque(OpaqueBlock(
              provider: "anthropic",
              type: contentBlock.type.rawValue,
              content: displayText,
              data: jsonString,
              isResponseContent: displayText != nil,
            )))
          }
        case .image, .document:
          break
      }
    }

    if let endnotes = formatEndnotesList(urlStrings: citationURLs), !endnotes.isEmpty {
      blocks.append(.endnotes(endnotes))
    }

    return blocks
  }

  private static func formatEndnotesList(urlStrings: [String]) -> String? {
    guard !urlStrings.isEmpty else {
      return nil
    }
    var result = ""
    for urlString in urlStrings {
      result += "- [\(urlString)](\(urlString))\n"
    }
    return result
  }
}
