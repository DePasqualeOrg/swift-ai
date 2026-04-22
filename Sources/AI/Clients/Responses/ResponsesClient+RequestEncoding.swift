// Copyright © Anthony DePasquale

import Foundation

extension ResponsesClient {
  enum ContentType {
    static let inputText = "input_text"
    static let inputImage = "input_image"
    static let inputAudio = "input_audio"
    static let inputFile = "input_file"
    static let outputText = "output_text"
    static let message = "message"
    static let functionCall = "function_call"
    static let functionCallOutput = "function_call_output"
  }

  static func inputItems(for message: Message) async throws -> [[String: any Sendable]] {
    enum SerializedInput {
      case contentItem([String: any Sendable])
      case topLevelItem([String: any Sendable])
    }

    func serializedAttachmentInput(for attachment: Attachment) async throws -> SerializedInput? {
      switch attachment.kind {
        case let .image(data, mimeType):
          let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
          return .contentItem([
            "type": ContentType.inputImage,
            "detail": "auto",
            "image_url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: processedMimeType),
          ])
        case let .document(data, _):
          var contentItem: [String: any Sendable] = [
            "type": ContentType.inputFile,
            "file_data": data.base64EncodedString(),
          ]
          if let fileName = attachment.filename {
            contentItem["filename"] = fileName
          }
          return .contentItem(contentItem)
        case let .audio(data, mimeType):
          let format: String? = switch mimeType {
            case "audio/wav", "audio/x-wav", "audio/wave": "wav"
            case "audio/mpeg", "audio/mp3": "mp3"
            default: nil
          }
          guard let format else {
            openAIResponsesLogger.warning("Audio format '\(mimeType)' is not supported by Responses (only wav and mp3). Attachment will be omitted.")
            return nil
          }
          return .topLevelItem([
            "type": ContentType.inputAudio,
            "input_audio": [
              "data": data.base64EncodedString(),
              "format": format,
            ] as [String: any Sendable],
          ])
        case .video:
          openAIResponsesLogger.warning("Attachment type '\(attachment.kind.mimeType)' is not supported in Responses message content and will be omitted.")
          return nil
      }
    }

    func downgradedTextContentItem(for block: Message.Content) -> [String: any Sendable]? {
      let text = block.portableReplayText()

      guard let text, !text.isEmpty else { return nil }
      return [
        "type": ContentType.inputText,
        "text": text,
      ]
    }

    switch message.role {
      case .user:
        var items: [[String: any Sendable]] = []
        var contentItems: [[String: any Sendable]] = []

        func flushUserMessage() {
          guard !contentItems.isEmpty else { return }
          items.append([
            "type": ContentType.message,
            "role": "user",
            "content": contentItems,
          ])
          contentItems.removeAll(keepingCapacity: true)
        }

        for block in message.content {
          switch block {
            case let .text(text) where !text.isEmpty:
              contentItems.append([
                "type": ContentType.inputText,
                "text": text,
              ])
            case let .endnotes(text) where !text.isEmpty:
              contentItems.append([
                "type": ContentType.inputText,
                "text": text,
              ])
            case let .attachment(attachment):
              if let serializedAttachment = try await serializedAttachmentInput(for: attachment) {
                switch serializedAttachment {
                  case let .contentItem(contentItem):
                    contentItems.append(contentItem)
                  case let .topLevelItem(item):
                    flushUserMessage()
                    items.append(item)
                }
              }
            default:
              break
          }
        }
        flushUserMessage()
        return items

      case .assistant:
        var items: [[String: any Sendable]] = []
        var contentItems: [[String: any Sendable]] = []
        let hasNativeAnnotatedOutputText = message.content.contains { block in
          guard case let .providerOpaque(opaque) = block else { return false }
          return opaque.provider == OpaqueBlock.ProviderID.openAIResponses
            && opaque.type == OpaqueBlock.OpenAIResponsesType.annotatedOutputText
        }
        var currentMetadata: [String: String]?
        var currentPhase: String?

        var isReplayingOutputMessage: Bool {
          guard let currentMetadata else { return false }
          return currentMetadata["id"] != nil && currentMetadata["status"] != nil
        }

        var textContentType: String {
          ContentType.outputText
        }

        var currentMessageRole: String {
          "assistant"
        }

        func appendAssistantTextContent(_ text: String) {
          guard !text.isEmpty else { return }
          var item: [String: any Sendable] = [
            "type": textContentType,
            "text": text,
          ]
          if isReplayingOutputMessage {
            item["annotations"] = [[String: any Sendable]]()
          }
          contentItems.append(item)
        }

        func clearCurrentMetadata() {
          currentMetadata = nil
          currentPhase = nil
        }

        func flushContentItems() {
          guard !contentItems.isEmpty else { return }
          var messageItem: [String: any Sendable] = [
            "type": ContentType.message,
            "role": currentMessageRole,
            "content": contentItems,
          ]
          if isReplayingOutputMessage, let metadata = currentMetadata {
            if let id = metadata["id"] { messageItem["id"] = id }
            if let status = metadata["status"] { messageItem["status"] = status }
            if let currentPhase {
              messageItem["phase"] = currentPhase
            }
          }
          items.append(messageItem)
          contentItems.removeAll(keepingCapacity: true)
        }

        for block in message.content {
          switch block {
            case let .providerOpaque(opaque) where opaque.isOpenAIResponsesMessageMetadata:
              flushContentItems()
              if let jsonString = opaque.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: String]
              {
                currentMetadata = parsed
                currentPhase = parsed["phase"]
              } else {
                currentMetadata = nil
                currentPhase = nil
              }
            case let .thinking(text, _) where !text.isEmpty:
              appendAssistantTextContent(text)
            case let .text(text) where !text.isEmpty:
              appendAssistantTextContent(text)
            case let .endnotes(text) where !text.isEmpty:
              guard !hasNativeAnnotatedOutputText else { break }
              appendAssistantTextContent(text)
            case let .providerOpaque(block) where block.isOpenAIResponsesAnnotatedOutputText:
              if let text = block.content {
                var item: [String: any Sendable] = [
                  "type": textContentType,
                  "text": text,
                ]
                if isReplayingOutputMessage {
                  let annotations: [[String: any Sendable]] = if let jsonString = block.data,
                                                                 let jsonData = jsonString.data(using: .utf8),
                                                                 let parsedAnnotations = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: any Sendable]]
                  {
                    parsedAnnotations
                  } else {
                    []
                  }
                  item["annotations"] = annotations
                }
                contentItems.append(item)
              }
            case let .providerOpaque(block) where block.isOpenAIResponsesRefusal:
              if let refusal = block.content {
                contentItems.append([
                  "type": OutputItemType.refusal,
                  "refusal": refusal,
                ])
              }
            case let .providerOpaque(block) where block.isOpenAIChatCompletionsRefusal:
              if let refusal = block.content {
                contentItems.append([
                  "type": OutputItemType.refusal,
                  "refusal": refusal,
                ])
              }
            case let .providerOpaque(block) where block.isOpenAIResponsesReasoning:
              flushContentItems()
              if let jsonString = block.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable]
              {
                items.append(parsed)
              } else {
                var reasoningItem: [String: any Sendable] = [
                  "type": OutputItemType.reasoning,
                ]
                if let id = block.signature {
                  reasoningItem["id"] = id
                }
                if let summaryText = block.content {
                  reasoningItem["summary"] = [["type": "summary_text", "text": summaryText]]
                }
                if let encryptedContent = block.data {
                  reasoningItem["encrypted_content"] = encryptedContent
                }
                items.append(reasoningItem)
              }
            case let .providerOpaque(block) where block.isAnthropicThinking || block.isGeminiThinking:
              if let text = block.content, !text.isEmpty {
                appendAssistantTextContent(text)
              }
            case let .providerOpaque(block):
              if let text = block.replayDowngradeText(for: .responses) {
                appendAssistantTextContent(text)
              }
            case let .attachment(attachment):
              if let serializedAttachment = try await serializedAttachmentInput(for: attachment) {
                switch serializedAttachment {
                  case let .contentItem(contentItem):
                    if currentMetadata != nil {
                      flushContentItems()
                      clearCurrentMetadata()
                    }
                    contentItems.append(contentItem)
                  case let .topLevelItem(item):
                    flushContentItems()
                    clearCurrentMetadata()
                    items.append(item)
                }
              }
            case let .toolCall(toolCall):
              flushContentItems()
              let foundationParams = Value.toSendable(toolCall.parameters)
              let argumentsData = try JSONSerialization.data(withJSONObject: foundationParams, options: [])
              guard let argumentsString = String(data: argumentsData, encoding: .utf8) else {
                throw AIError.invalidRequest(message: "Failed to serialize function call arguments to JSON string")
              }
              items.append([
                "type": ContentType.functionCall,
                "call_id": toolCall.id,
                "name": toolCall.name,
                "arguments": argumentsString,
              ])
            case let .providerOpaque(block) where block.provider == "openai-responses":
              flushContentItems()
              if let jsonString = block.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable]
              {
                items.append(parsed)
              } else if block.isResponseContent, let text = block.content, !text.isEmpty {
                clearCurrentMetadata()
                contentItems.append([
                  "type": ContentType.outputText,
                  "text": text,
                ])
              }
            default:
              break
          }
        }

        flushContentItems()
        return items

      case .tool:
        return try message.content.compactMap { block -> [String: any Sendable]? in
          guard case let .toolResult(toolResult) = block else { return nil }

          let isError = toolResult.isError == true
          var outputItems: [[String: any Sendable]] = []
          var hasNonTextContent = false

          for content in toolResult.content {
            switch content {
              case let .text(text):
                outputItems.append([
                  "type": ContentType.inputText,
                  "text": text,
                ])
              case let .image(data, mimeType):
                hasNonTextContent = true
                let mediaType = mimeType ?? "image/png"
                let dataURL = "data:\(mediaType);base64,\(data.base64EncodedString())"
                outputItems.append([
                  "type": ContentType.inputImage,
                  "detail": "auto",
                  "image_url": dataURL,
                ])
              case let .audio(data, mimeType):
                openAIResponsesLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Responses API. Using fallback text.")
                outputItems.append([
                  "type": ContentType.inputText,
                  "text": ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription,
                ])
              case let .file(data, mimeType, filename):
                hasNonTextContent = true
                if mimeType.hasPrefix("image/") {
                  let dataURL = "data:\(mimeType);base64,\(data.base64EncodedString())"
                  outputItems.append([
                    "type": ContentType.inputImage,
                    "detail": "auto",
                    "image_url": dataURL,
                  ])
                } else {
                  var fileItem: [String: any Sendable] = [
                    "type": ContentType.inputFile,
                    "file_data": data.base64EncodedString(),
                  ]
                  if let filename {
                    fileItem["filename"] = filename
                  }
                  outputItems.append(fileItem)
                }
              case let .json(value):
                outputItems.append([
                  "type": ContentType.inputText,
                  "text": value.jsonString,
                ])
              case let .embeddedResource(data, _, mimeType):
                let inlineMime = mimeType ?? "application/octet-stream"
                if inlineMime.hasPrefix("image/") {
                  hasNonTextContent = true
                  let dataURL = "data:\(inlineMime);base64,\(data.base64EncodedString())"
                  outputItems.append([
                    "type": ContentType.inputImage,
                    "detail": "auto",
                    "image_url": dataURL,
                  ])
                } else if inlineMime == "application/pdf" {
                  hasNonTextContent = true
                  outputItems.append([
                    "type": ContentType.inputFile,
                    "file_data": data.base64EncodedString(),
                  ])
                } else {
                  // Audio, arbitrary MIME — text fallback (bytes lost; URI annotated).
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": content.fallbackDescription,
                  ])
                }
              case let .resourceLink(uri, _, _, _, _, _):
                if isResponsesFetchableScheme(uri) {
                  hasNonTextContent = true
                  outputItems.append([
                    "type": ContentType.inputFile,
                    "file_url": uri,
                  ])
                } else {
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": content.fallbackDescription,
                  ])
                }
              case .embeddedText:
                outputItems.append([
                  "type": ContentType.inputText,
                  "text": content.fallbackDescription,
                ])
            }
          }

          // Responses has no out-of-band `isError` field on `function_call_output`,
          // so for errors with rich (non-text) content we prepend a `"Tool call failed:"`
          // sentinel part to keep the failure signal distinguishable from a successful
          // multi-part return. Pure-text errors keep today's `{"error": text}` JSON
          // wrap so OpenAI's models recognize the failure.
          let resultOutput: any Sendable
          if isError, !hasNonTextContent {
            let errorText = outputItems.compactMap { $0["text"] as? String }.joined(separator: "\n")
            let errorPayload: [String: String] = ["error": errorText.isEmpty ? "Unknown error" : errorText]
            let errorData = try JSONSerialization.data(withJSONObject: errorPayload, options: [])
            resultOutput = String(data: errorData, encoding: .utf8) ?? "{\"error\":\"Unknown error\"}"
          } else if hasNonTextContent {
            if isError {
              outputItems.insert([
                "type": ContentType.inputText,
                "text": "Tool call failed:",
              ], at: 0)
            }
            resultOutput = outputItems
          } else {
            let texts = outputItems.compactMap { $0["text"] as? String }
            resultOutput = texts.joined(separator: "\n")
          }

          return [
            "type": ContentType.functionCallOutput,
            "call_id": toolResult.id,
            "output": resultOutput,
          ]
        }

      case .system, .developer:
        var items: [[String: any Sendable]] = []
        var contentItems: [[String: any Sendable]] = []

        func flushInstructionMessage() {
          guard !contentItems.isEmpty else { return }
          items.append([
            "type": ContentType.message,
            "role": message.role.rawValue,
            "content": contentItems,
          ])
          contentItems.removeAll(keepingCapacity: true)
        }

        for block in message.content {
          if let textContentItem = downgradedTextContentItem(for: block) {
            contentItems.append(textContentItem)
            continue
          }
          if case let .attachment(attachment) = block,
             let serializedAttachment = try await serializedAttachmentInput(for: attachment)
          {
            switch serializedAttachment {
              case let .contentItem(contentItem):
                contentItems.append(contentItem)
              case let .topLevelItem(item):
                flushInstructionMessage()
                items.append(item)
            }
          }
        }
        flushInstructionMessage()
        return items
    }
  }
}
