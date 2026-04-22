// Copyright © Anthony DePasquale

import Foundation

extension AnthropicClient {
  static func anthropicContentBlocks(for message: Message) async throws -> [AnthropicClient.ContentBlockParam] {
    var contentBlocks: [AnthropicClient.ContentBlockParam] = []
    let hasNativeCitationBlocks = message.content.contains { block in
      guard case let .providerOpaque(opaque) = block else { return false }
      return opaque.isAnthropicCitationCarrier
    }

    for block in message.content {
      switch block {
        case let .thinking(text, signature) where signature != nil:
          contentBlocks.append(.init(
            type: .thinking,
            thinking: text,
            signature: signature,
          ))
        case let .redactedThinking(data):
          contentBlocks.append(.init(type: .redactedThinking, data: data))
        case let .providerOpaque(opaqueBlock) where opaqueBlock.isAnthropicThinking:
          contentBlocks.append(.init(
            type: .thinking,
            thinking: opaqueBlock.content,
            signature: opaqueBlock.signature,
          ))
        case let .providerOpaque(opaqueBlock) where opaqueBlock.isAnthropicRedactedThinking:
          contentBlocks.append(.init(type: .redactedThinking, data: opaqueBlock.data))
        case let .providerOpaque(opaqueBlock) where opaqueBlock.isAnthropicNativeStructuredBlock:
          if let jsonString = opaqueBlock.data,
             let jsonData = jsonString.data(using: .utf8),
             let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable],
             let rawValue = try? Value.fromAny(jsonObject)
          {
            contentBlocks.append(.init(
              type: ContentBlockType(rawValue: opaqueBlock.type) ?? .serverToolUse,
              rawValue: rawValue,
            ))
          } else if let text = opaqueBlock.replayDowngradeText(for: .anthropic) {
            contentBlocks.append(.init(type: .text, text: text))
          }
        case let .providerOpaque(opaqueBlock):
          if let text = opaqueBlock.replayDowngradeText(for: .anthropic) {
            contentBlocks.append(.init(type: .text, text: text))
          }
        case let .text(text) where !text.isEmpty:
          contentBlocks.append(.init(type: .text, text: text))
        case let .endnotes(text) where !text.isEmpty:
          guard !hasNativeCitationBlocks else { break }
          contentBlocks.append(.init(type: .text, text: text))
        case let .toolCall(toolCall):
          contentBlocks.append(.init(
            type: .toolUse,
            toolUse: .init(
              id: toolCall.id,
              name: toolCall.name,
              input: .object(toolCall.parameters),
            ),
          ))
        case let .toolResult(toolResult):
          var resultContentBlocks: [AnthropicClient.ToolResultContentBlock] = []
          for content in toolResult.content {
            switch content {
              case let .text(text):
                resultContentBlocks.append(.text(text))
              case let .image(data, mimeType):
                let (normalizedData, normalizedMime) = try await MediaProcessor.normalizeImageForAnthropic(data, mimeType: mimeType ?? "image/png")
                resultContentBlocks.append(.image(
                  mediaType: normalizedMime,
                  data: normalizedData.base64EncodedString(),
                ))
              case let .audio(data, mimeType):
                anthropicLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Anthropic. Using fallback text.")
                resultContentBlocks.append(.text(ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription))
              case let .file(data, mimeType, filename):
                if mimeType.hasPrefix("image/") {
                  let (normalizedData, normalizedMime) = try await MediaProcessor.normalizeImageForAnthropic(data, mimeType: mimeType)
                  resultContentBlocks.append(.image(
                    mediaType: normalizedMime,
                    data: normalizedData.base64EncodedString(),
                  ))
                } else if mimeType == "application/pdf" {
                  resultContentBlocks.append(.document(
                    mediaType: mimeType,
                    data: data.base64EncodedString(),
                  ))
                } else if mimeType == "text/plain", let text = String(data: data, encoding: .utf8) {
                  resultContentBlocks.append(.document(
                    mediaType: mimeType,
                    data: text,
                    sourceType: "text",
                  ))
                } else {
                  anthropicLogger.warning("Tool '\(toolResult.name)' returned a file (\(mimeType)), which is not supported by Anthropic. Using fallback text.")
                  resultContentBlocks.append(.text(ToolResult.Content.file(data, mimeType: mimeType, filename: filename).fallbackDescription))
                }
              case let .json(value):
                resultContentBlocks.append(.text(value.jsonString))
              case let .embeddedResource(data, _, mimeType):
                let inlineMime = mimeType ?? "application/octet-stream"
                if inlineMime.hasPrefix("image/") {
                  let (normalizedData, normalizedMime) = try await MediaProcessor.normalizeImageForAnthropic(data, mimeType: inlineMime)
                  resultContentBlocks.append(.image(
                    mediaType: normalizedMime,
                    data: normalizedData.base64EncodedString(),
                  ))
                } else if inlineMime == "application/pdf" {
                  resultContentBlocks.append(.document(
                    mediaType: inlineMime,
                    data: data.base64EncodedString(),
                  ))
                } else {
                  // Audio, arbitrary MIME — bytes lost, URI annotated.
                  resultContentBlocks.append(.text(content.fallbackDescription))
                }
              case .embeddedText, .resourceLink:
                resultContentBlocks.append(.text(content.fallbackDescription))
            }
          }

          let resultContent: AnthropicClient.ToolResultContent = if resultContentBlocks.count == 1, let text = resultContentBlocks[0].text, resultContentBlocks[0].source == nil {
            .text(text)
          } else if resultContentBlocks.isEmpty {
            .text("")
          } else {
            .blocks(resultContentBlocks)
          }

          contentBlocks.append(.init(
            type: .toolResult,
            toolResult: .init(
              toolUseId: toolResult.id,
              content: resultContent,
              isError: toolResult.isError,
            ),
          ))
        case let .attachment(attachment):
          switch attachment.kind {
            case let .image(data, mimeType):
              let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
              let (normalizedData, normalizedMime) = try await MediaProcessor.normalizeImageForAnthropic(processedImageData, mimeType: processedMimeType)
              contentBlocks.append(.init(
                type: .image,
                source: .init(
                  type: "base64",
                  mediaType: normalizedMime,
                  data: normalizedData.base64EncodedString(),
                  url: nil,
                ),
              ))
            case let .document(data, mimeType) where mimeType == "application/pdf":
              if data.count > 32 * 1024 * 1024 {
                throw AIError.invalidRequest(message: "PDF exceeds maximum size of 32MB")
              }
              contentBlocks.append(.init(
                type: .document,
                source: .init(
                  type: "base64",
                  mediaType: mimeType,
                  data: data.base64EncodedString(),
                  url: nil,
                ),
              ))
            case let .document(data, mimeType) where mimeType == "text/plain":
              guard let text = String(data: data, encoding: .utf8) else {
                throw AIError.invalidRequest(message: "Plain text document attachments must be valid UTF-8 for Anthropic")
              }
              contentBlocks.append(.init(
                type: .document,
                source: .init(
                  type: "text",
                  mediaType: mimeType,
                  data: text,
                  url: nil,
                ),
              ))
            case .video, .audio, .document:
              anthropicLogger.warning("Attachment type '\(attachment.kind.mimeType)' is not supported by Anthropic and will be omitted.")
          }
        default:
          break
      }
    }

    return contentBlocks
  }

  func buildMessagesRequest(params: MessageCreateParams, stream: Bool, apiKey: String) async throws -> URLRequest {
    var request = URLRequest(url: messagesEndpoint)
    request.httpMethod = "POST"
    request.timeoutInterval = timeout
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue(version, forHTTPHeaderField: "anthropic-version")
    request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
    request.setValue("AnthropicSwift/1.0", forHTTPHeaderField: "User-Agent")

    var betaHeaders: [String] = []
    if let tools = params.tools, tools.contains(where: { tool in
      if case .codeExecution = tool { return true }
      return false
    }) {
      betaHeaders.append("code-execution-2025-05-22")
    }

    if let tools = params.tools, tools.contains(where: { tool in
      if case .webFetch = tool { return true }
      return false
    }) {
      betaHeaders.append("web-fetch-2025-09-10")
    }

    if !betaHeaders.isEmpty {
      request.setValue(betaHeaders.joined(separator: ","), forHTTPHeaderField: "anthropic-beta")
    }

    var messagesArray: [Value] = []
    for message in params.messages {
      var messageDict: [String: Value] = [
        "role": .string(message.role.rawValue),
      ]
      if let attachments = message.attachments, !attachments.isEmpty {
        var contentArray: [Value] = []
        for attachment in attachments {
          switch attachment.kind {
            case let .image(data, mimeType):
              do {
                let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                let (normalizedData, normalizedMime) = try await MediaProcessor.normalizeImageForAnthropic(processedImageData, mimeType: processedMimeType)
                contentArray.append(.object([
                  "type": .string("image"),
                  "source": .object([
                    "type": .string("base64"),
                    "media_type": .string(normalizedMime),
                    "data": .string(normalizedData.base64EncodedString()),
                  ]),
                ]))
              } catch {
                anthropicLogger.error("Failed to process image: \(error.localizedDescription)")
                throw error
              }
            case .video:
              break
            case .audio:
              break
            case let .document(data, mimeType):
              if mimeType == "application/pdf" {
                do {
                  if data.count > 32 * 1024 * 1024 {
                    throw AIError.invalidRequest(message: "PDF exceeds maximum size of 32MB")
                  }
                  contentArray.append(.object([
                    "type": .string("document"),
                    "source": .object([
                      "type": .string("base64"),
                      "media_type": .string(mimeType),
                      "data": .string(data.base64EncodedString()),
                    ]),
                  ]))
                } catch {
                  anthropicLogger.error("Failed to process PDF: \(error.localizedDescription)")
                  throw error
                }
              }
          }
        }
        if let text = message.text, !text.isEmpty {
          contentArray.append(.object([
            "type": .string("text"),
            "text": .string(text),
          ]))
        }
        messageDict["content"] = .array(contentArray)
      } else if let contentBlocks = message.contentBlocks {
        messageDict["content"] = .array(contentBlocks.map { block -> Value in
          if let rawValue = block.rawValue {
            return rawValue
          }
          var blockDict: [String: Value] = [
            "type": .string(block.type.rawValue),
          ]
          switch block.type {
            case .text:
              if let text = block.text {
                blockDict["text"] = .string(text)
              }
            case .toolUse:
              if let toolUse = block.toolUse {
                blockDict["id"] = .string(toolUse.id)
                blockDict["name"] = .string(toolUse.name)
                blockDict["input"] = toolUse.input
              }
            case .toolResult:
              if let toolResult = block.toolResult {
                blockDict["tool_use_id"] = .string(toolResult.toolUseId)
                switch toolResult.content {
                  case let .text(text):
                    blockDict["content"] = .string(text)
                  case let .blocks(contentBlocks):
                    blockDict["content"] = .array(contentBlocks.map { .object($0.raw) })
                }
                if let isError = toolResult.isError {
                  blockDict["is_error"] = .bool(isError)
                }
              }
            case .thinking:
              if let thinking = block.thinking {
                blockDict["thinking"] = .string(thinking)
              }
              if let signature = block.signature {
                blockDict["signature"] = .string(signature)
              }
            case .redactedThinking:
              if let data = block.data {
                blockDict["data"] = .string(data)
              }
            case .serverToolUse, .webSearchToolResult, .webFetchToolResult:
              break
            case .image, .document:
              if let source = block.source {
                var sourceDict: [String: Value] = [
                  "type": .string(source.type),
                  "media_type": .string(source.mediaType),
                ]
                if let data = source.data {
                  sourceDict["data"] = .string(data)
                }
                if let url = source.url {
                  sourceDict["url"] = .string(url)
                }
                blockDict["source"] = .object(sourceDict)
              }
            case .codeExecutionToolResult:
              if let codeExecutionToolResult = block.codeExecutionToolResult {
                blockDict["tool_use_id"] = .string(codeExecutionToolResult.toolUseId)
                blockDict["content"] = codeExecutionToolResult.content
              }
          }
          return .object(blockDict)
        })
      } else if let text = message.text {
        messageDict["content"] = .array([
          .object(["type": .string("text"), "text": .string(text)]),
        ])
      }
      messagesArray.append(.object(messageDict))
    }

    var requestBody: [String: Value] = [
      "model": .string(params.model),
      "messages": .array(messagesArray),
      "stream": .bool(stream),
    ]
    let effectiveMaxTokens = params.maxTokens ?? Self.defaultMaxTokens(for: params.model)
    requestBody["max_tokens"] = .int(effectiveMaxTokens)
    // Prompt caching: use a counter to stay within the 4-breakpoint limit.
    // Priority order: system prompt, last tool, then top-level (conversation).
    let maxCacheBreakpoints = 4
    var cacheBreakpointsUsed = 0
    func nextCacheControl() -> Value? {
      guard let cacheControl = params.cacheControl, cacheBreakpointsUsed < maxCacheBreakpoints else { return nil }
      cacheBreakpointsUsed += 1
      return cacheControl.asValue
    }
    if let systemPrompt = params.system, !systemPrompt.isEmpty {
      var systemBlock: [String: Value] = [
        "type": .string("text"),
        "text": .string(systemPrompt),
      ]
      if let cc = nextCacheControl() {
        systemBlock["cache_control"] = cc
      }
      requestBody["system"] = .array([.object(systemBlock)])
    }
    if let temperature = params.temperature {
      requestBody["temperature"] = .double(Double(temperature))
    }
    if let topP = params.topP {
      requestBody["top_p"] = .double(Double(topP))
    }
    if let topK = params.topK {
      requestBody["top_k"] = .int(topK)
    }
    if let thinking = params.thinking {
      var thinkingDict: [String: Value] = [
        "type": .string(thinking.type.rawValue),
      ]
      if case .enabled = thinking.type, let budgetTokens = thinking.budgetTokens {
        thinkingDict["budget_tokens"] = .int(budgetTokens)
      }
      if let display = thinking.display {
        thinkingDict["display"] = .string(display.rawValue)
      }
      requestBody["thinking"] = .object(thinkingDict)
    }
    if let effort = params.effort {
      requestBody["output_config"] = .object(["effort": .string(effort.rawValue)])
    }
    if let tools = params.tools, !tools.isEmpty {
      var toolsArray = try tools.compactMap { tool -> Value? in
        let toolData = try JSONEncoder().encode(tool)
        if let toolJson = try? JSONSerialization.jsonObject(with: toolData, options: []) {
          return try Value.fromAny(toolJson)
        } else {
          anthropicLogger.error("Failed to convert encoded tool to Value: \(String(describing: tool))")
          return nil
        }
      }
      if !toolsArray.isEmpty, case var .object(lastToolDict) = toolsArray[toolsArray.count - 1],
         let cc = nextCacheControl()
      {
        lastToolDict["cache_control"] = cc
        toolsArray[toolsArray.count - 1] = .object(lastToolDict)
      }
      requestBody["tools"] = .array(toolsArray)
    }
    if let toolChoice = params.toolChoice {
      var toolChoiceDict: [String: Value] = switch toolChoice {
        case .auto: ["type": .string("auto")]
        case .any: ["type": .string("any")]
        case .none: ["type": .string("none")]
        case let .tool(name): ["type": .string("tool"), "name": .string(name)]
      }
      if case .none = toolChoice {} else if let disableParallelToolUse = params.disableParallelToolUse {
        toolChoiceDict["disable_parallel_tool_use"] = .bool(disableParallelToolUse)
      }
      requestBody["tool_choice"] = .object(toolChoiceDict)
    }
    if let metadata = params.metadata {
      var metadataDict: [String: Value] = [:]
      for (key, value) in metadata {
        metadataDict[key] = .string(value)
      }
      requestBody["metadata"] = .object(metadataDict)
    }
    // Top-level cache_control is a separate server-side mechanism that automatically
    // applies to the last cacheable block. It does not count against the per-block limit.
    if let cacheControl = params.cacheControl {
      requestBody["cache_control"] = cacheControl.asValue
    }
    request.httpBody = try Value.object(requestBody).toData()
    return request
  }

  func createMessageStream(params: MessageCreateParams, apiKey: String) async -> MessageStream {
    await MessageStream.createMessage(client: self, params: params, apiKey: apiKey, session: session)
  }

  static func mapRole(_ role: Message.Role) -> AnthropicClient.Role {
    switch role {
      case .user, .tool:
        return .user
      case .assistant:
        return .assistant
      case .system, .developer:
        anthropicLogger.error("Message role \(role.rawValue) not supported in Anthropic API client")
        return .user
    }
  }
}
