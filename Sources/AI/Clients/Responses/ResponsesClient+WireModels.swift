// Copyright © Anthony DePasquale

import Foundation
import os.log

extension ResponsesClient {
  struct StreamEvent: Decodable {
    let type: String?
    let sequenceNumber: Int?
    let delta: String?
    let text: String?
    let refusal: String?
    let name: String?
    let annotationIndex: Int?
    let itemId: String?
    let outputIndex: Int?
    let contentIndex: Int?
    let summaryIndex: Int?
    let arguments: String?
    let annotation: AnnotationItem?
    let item: ResponseOutputItem?
    let part: ContentItem?
    let response: ResponseObject?
    let error: ErrorObject?

    enum CodingKeys: String, CodingKey {
      case type
      case sequenceNumber = "sequence_number"
      case delta
      case text
      case refusal
      case name
      case annotationIndex = "annotation_index"
      case itemId = "item_id"
      case outputIndex = "output_index"
      case contentIndex = "content_index"
      case summaryIndex = "summary_index"
      case arguments
      case annotation
      case item
      case part
      case response
      case error
    }
  }

  struct SummaryItem: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var text: String? {
      raw["text"]?.stringValue
    }
  }

  struct Usage: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var inputTokens: Int? {
      raw["input_tokens"]?.intValue
    }

    var outputTokens: Int? {
      raw["output_tokens"]?.intValue
    }

    var totalTokens: Int? {
      raw["total_tokens"]?.intValue
    }

    var inputTokensDetails: InputTokensDetails? {
      raw["input_tokens_details"]?.objectValue.map(InputTokensDetails.init(raw:))
    }

    var outputTokensDetails: OutputTokensDetails? {
      raw["output_tokens_details"]?.objectValue.map(OutputTokensDetails.init(raw:))
    }
  }

  struct InputTokensDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var cachedTokens: Int? {
      raw["cached_tokens"]?.intValue
    }
  }

  struct OutputTokensDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var reasoningTokens: Int? {
      raw["reasoning_tokens"]?.intValue
    }
  }

  struct IncompleteDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var reason: String? {
      raw["reason"]?.stringValue
    }
  }

  struct ResponseObject: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var id: String? {
      raw["id"]?.stringValue
    }

    var status: String? {
      raw["status"]?.stringValue
    }

    var model: String? {
      raw["model"]?.stringValue
    }

    var createdAt: Int? {
      raw["created_at"]?.intValue
    }

    var output: [ResponseOutputItem]? {
      raw["output"]?.arrayValue?.compactMap(\.objectValue).map(ResponseOutputItem.init(raw:))
    }

    var outputText: String? {
      raw["output_text"]?.stringValue
    }

    var usage: Usage? {
      raw["usage"]?.objectValue.map(Usage.init(raw:))
    }

    var error: ErrorObject? {
      raw["error"]?.objectValue.map(ErrorObject.init(raw:))
    }

    var incompleteDetails: IncompleteDetails? {
      raw["incomplete_details"]?.objectValue.map(IncompleteDetails.init(raw:))
    }

    func toGenerationResponse() -> GenerationResponse {
      var content: [Message.Content] = []
      var hasRefusal = false
      var citations: [(label: String, url: String?, fileId: String?)] = []

      if let outputArray = output, !outputArray.isEmpty {
        for item in outputArray {
          guard let itemType = item.type else { continue }

          switch itemType {
            case OutputItemType.message:
              var metadata: [String: String] = [:]
              if let messageId = item.id { metadata["id"] = messageId }
              if let status = item.status { metadata["status"] = status }
              if let phase = item.phase { metadata["phase"] = phase }
              if !metadata.isEmpty,
                 let jsonData = try? JSONSerialization.data(withJSONObject: metadata),
                 let jsonString = String(data: jsonData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "message_metadata",
                  data: jsonString,
                )))
              }
              if let contentArray = item.content {
                for contentItem in contentArray {
                  if contentItem.type == OutputItemType.outputText, let text = contentItem.text, !text.isEmpty {
                    if let annotations = contentItem.annotations, !annotations.isEmpty {
                      for annotation in annotations {
                        switch annotation.type {
                          case "url_citation":
                            if let url = annotation.url {
                              citations.append((label: annotation.title ?? url, url: url, fileId: nil))
                            }
                          case "file_citation", "container_file_citation":
                            if let filename = annotation.filename {
                              citations.append((label: filename, url: nil, fileId: annotation.fileId))
                            }
                          default:
                            break
                        }
                      }
                      let annotationsRaw = annotations.map { Value.toSendable($0.raw) }
                      let annotationsJson = (try? JSONSerialization.data(withJSONObject: annotationsRaw))
                        .flatMap { String(data: $0, encoding: .utf8) }
                      content.append(.providerOpaque(OpaqueBlock(
                        provider: "openai-responses",
                        type: "annotated_output_text",
                        content: text,
                        data: annotationsJson,
                        isResponseContent: true,
                      )))
                    } else {
                      content.append(.text(text))
                    }
                  } else if contentItem.type == OutputItemType.refusal,
                            let refusal = contentItem.refusal,
                            !refusal.isEmpty
                  {
                    content.append(.providerOpaque(OpaqueBlock(
                      provider: "openai-responses",
                      type: "refusal",
                      content: refusal,
                      isResponseContent: true,
                    )))
                    hasRefusal = true
                  }
                }
              }
            case OutputItemType.reasoning:
              let reasoningContentText = item.content?
                .filter { $0.type == "reasoning_text" }
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let summaryText = item.summary?
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let reasoningText = (reasoningContentText?.isEmpty == false) ? reasoningContentText : summaryText
              if let reasoningText, !reasoningText.isEmpty {
                content.append(.thinking(text: reasoningText, signature: nil))
              }
              if let itemId = item.id,
                 let rawItemData = try? JSONEncoder().encode(item.raw),
                 let rawItemJson = String(data: rawItemData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "reasoning",
                  content: summaryText,
                  signature: itemId,
                  data: rawItemJson,
                )))
              }
            case OutputItemType.functionCall:
              if let name = item.name,
                 let callId = item.callId ?? item.id,
                 let argumentsString = item.arguments
              {
                var parameters: [String: Value] = [:]
                if let argumentsData = argumentsString.data(using: .utf8),
                   let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
                {
                  parameters = parsedArgs
                } else if !argumentsString.isEmpty {
                  parameters = [
                    "_parseError": .string("Failed to parse arguments JSON"),
                    "_rawArguments": .string(argumentsString),
                  ]
                }
                content.append(.toolCall(ToolCall(
                  name: name,
                  id: callId,
                  parameters: parameters,
                )))
              }
            default:
              let sendable = Value.toSendable(item.raw)
              if let jsonData = try? JSONSerialization.data(withJSONObject: sendable),
                 let jsonString = String(data: jsonData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: itemType,
                  data: jsonString,
                )))
              }
          }
        }
      } else if let outputText, !outputText.isEmpty {
        content.append(.text(outputText))
      }

      if !citations.isEmpty {
        let uniqueCitations = citations.reduce(into: [(label: String, url: String?, fileId: String?)]()) { result, citation in
          let key = citation.url ?? citation.fileId ?? citation.label
          if !result.contains(where: { ($0.url ?? $0.fileId ?? $0.label) == key }) {
            result.append(citation)
          }
        }
        let endnotes = uniqueCitations.map { citation in
          if let url = citation.url {
            "- [\(citation.label)](\(url))"
          } else {
            "- \(citation.label)"
          }
        }.joined(separator: "\n") + "\n"
        content.append(.endnotes(endnotes))
      }

      let toolCallCount = content.reduce(into: 0) { count, item in
        if case .toolCall = item {
          count += 1
        }
      }
      let finishReason: GenerationResponse.FinishReason? = if let status {
        switch status {
          case "completed":
            if hasRefusal { .refusal }
            else if toolCallCount > 0 { .toolUse }
            else { .stop }
          case "incomplete":
            switch incompleteDetails?.reason {
              case "max_output_tokens": .maxTokens
              case "content_filter": .contentFilter
              default: .other
            }
          case "failed", "cancelled": .other
          default: nil
        }
      } else {
        nil
      }

      var createdAtDate: Date?
      if let createdAt {
        createdAtDate = Date(timeIntervalSince1970: TimeInterval(createdAt))
      }

      let metadata = GenerationResponse.Metadata(
        responseId: id,
        model: model,
        createdAt: createdAtDate,
        finishReason: finishReason,
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
        totalTokens: usage?.totalTokens,
        cacheReadInputTokens: usage?.inputTokensDetails?.cachedTokens,
        reasoningTokens: usage?.outputTokensDetails?.reasoningTokens,
      )

      return GenerationResponse(content: content, metadata: metadata)
    }
  }

  struct ResponseOutputItem: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var id: String? {
      raw["id"]?.stringValue
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var status: String? {
      raw["status"]?.stringValue
    }

    var name: String? {
      raw["name"]?.stringValue
    }

    var callId: String? {
      raw["call_id"]?.stringValue
    }

    var arguments: String? {
      raw["arguments"]?.stringValue
    }

    var encryptedContent: String? {
      raw["encrypted_content"]?.stringValue
    }

    var phase: String? {
      raw["phase"]?.stringValue
    }

    var content: [ContentItem]? {
      raw["content"]?.arrayValue?.compactMap(\.objectValue).map(ContentItem.init(raw:))
    }

    var summary: [SummaryItem]? {
      raw["summary"]?.arrayValue?.compactMap(\.objectValue).map(SummaryItem.init(raw:))
    }
  }

  struct ContentItem: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var text: String? {
      raw["text"]?.stringValue
    }

    var refusal: String? {
      raw["refusal"]?.stringValue
    }

    var annotations: [AnnotationItem]? {
      raw["annotations"]?.arrayValue?.compactMap(\.objectValue).map(AnnotationItem.init(raw:))
    }
  }

  struct AnnotationItem: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var url: String? {
      raw["url"]?.stringValue
    }

    var title: String? {
      raw["title"]?.stringValue
    }

    var filename: String? {
      raw["filename"]?.stringValue
    }

    var fileId: String? {
      raw["file_id"]?.stringValue
    }
  }

  struct ErrorObject: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var message: String? {
      raw["message"]?.stringValue
    }

    var code: String? {
      raw["code"]?.stringValue
    }
  }

  static func supportsReasoning(_ modelId: String) -> Bool {
    if modelId.hasPrefix("gpt-3") || modelId.hasPrefix("gpt-4") { return false }
    if modelId.hasPrefix("chatgpt-") { return false }
    if modelId == "o1-mini" || modelId.hasPrefix("o1-mini-") { return false }
    if modelId == "o1-preview" || modelId.hasPrefix("o1-preview-") { return false }
    if modelId.hasPrefix("grok-") {
      return modelId.hasPrefix("grok-3-mini")
    }
    return true
  }
}

enum ResponseFormat {
  case text
  case jsonObject
  case jsonSchema(schema: [String: any Sendable], name: String? = nil, description: String? = nil)
}

let openAIResponsesLogger = Logger(
  subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence",
  category: "ResponsesClient",
)

/// True when `uri` uses a scheme OpenAI's Responses API will fetch
/// (`https://`, `http://`). Other schemes — including `gs://` — stringify.
func isResponsesFetchableScheme(_ uri: String) -> Bool {
  let lower = uri.lowercased()
  return lower.hasPrefix("https://") || lower.hasPrefix("http://")
}
