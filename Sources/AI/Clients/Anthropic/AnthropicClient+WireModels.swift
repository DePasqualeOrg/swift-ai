// Copyright © Anthony DePasquale

import Foundation

extension AnthropicClient {
  enum Role: String, Codable {
    case user
    case assistant
  }

  struct APIMessage: Codable {
    let id: String
    let role: Role
    var content: [ContentBlock]
    var stopReason: String?
    var stopSequence: String?
    var usage: Usage

    struct Usage: Codable {
      var inputTokens: Int?
      var outputTokens: Int?
      var cacheCreationInputTokens: Int?
      var cacheReadInputTokens: Int?

      enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case cacheCreationInputTokens = "cache_creation_input_tokens"
        case cacheReadInputTokens = "cache_read_input_tokens"
      }
    }
  }

  enum ContentBlockType: String, Codable {
    case text
    case thinking
    case redactedThinking = "redacted_thinking"
    case toolUse = "tool_use"
    case toolResult = "tool_result"
    case serverToolUse = "server_tool_use"
    case webSearchToolResult = "web_search_tool_result"
    case webFetchToolResult = "web_fetch_tool_result"
    case codeExecutionToolResult = "code_execution_tool_result"
    case image
    case document
  }

  struct ContentBlock: Codable {
    let type: ContentBlockType
    var text: String?
    var thinking: String?
    var citations: [Citation]?
    var toolUse: ToolUseBlock?
    var toolResult: ToolResultBlock?
    var serverToolUse: ServerToolUseBlock?
    var webSearchToolResult: WebSearchToolResultBlock?
    var webFetchToolResult: WebFetchToolResultBlock?
    var codeExecutionToolResult: CodeExecutionToolResultBlock?
    var source: ContentBlockSource?
    var signature: String?
    var data: String?

    private enum CodingKeys: String, CodingKey {
      case type, text, thinking, citations, toolUse, toolResult, serverToolUse, webSearchToolResult, webFetchToolResult, codeExecutionToolResult, source, signature, data
      case id, name, input
      case toolUseId = "tool_use_id"
      case content, isError = "is_error"
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      type = try container.decode(ContentBlockType.self, forKey: .type)
      switch type {
        case .text:
          text = try container.decodeIfPresent(String.self, forKey: .text)
          citations = try container.decodeIfPresent([Citation].self, forKey: .citations)
        case .thinking:
          thinking = try container.decodeIfPresent(String.self, forKey: .thinking)
          signature = try container.decodeIfPresent(String.self, forKey: .signature)
        case .redactedThinking:
          data = try container.decodeIfPresent(String.self, forKey: .data)
        case .toolUse:
          let id = try container.decode(String.self, forKey: .id)
          let name = try container.decode(String.self, forKey: .name)
          let inputDecoder = try container.superDecoder(forKey: .input)
          var input: Value
          do {
            let singleValueContainer = try inputDecoder.singleValueContainer()
            if singleValueContainer.decodeNil() {
              input = .null
            } else {
              input = try Value(from: inputDecoder)
            }
          } catch {
            anthropicLogger.warning("Failed to decode input for toolUse, name: \(name)): \(error.localizedDescription)")
            input = .object([:])
          }
          toolUse = ToolUseBlock(id: id, name: name, input: input)
        case .toolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          let content = try container.decode(ToolResultContent.self, forKey: .content)
          let isError = try container.decodeIfPresent(Bool.self, forKey: .isError)
          toolResult = ToolResultBlock(toolUseId: toolUseId, content: content, isError: isError)
        case .image, .document:
          source = try container.decodeIfPresent(ContentBlockSource.self, forKey: .source)
        case .serverToolUse:
          let id = try container.decode(String.self, forKey: .id)
          let name = try container.decode(String.self, forKey: .name)
          var finalInputValue: Value = .object([:])
          if name == "code_execution" {
            let inputFieldDecoder = try container.superDecoder(forKey: .input)
            do {
              struct TempServerInput: Decodable { let code: String? }
              let tempInput = try TempServerInput(from: inputFieldDecoder)

              if let codeString = tempInput.code {
                finalInputValue = .object(["code": .string(codeString)])
              } else {
                finalInputValue = .object([:])
              }
            } catch {
              anthropicLogger.warning("Failed to decode 'input' for serverToolUse (name: code_execution) as TempServerInput: \(error.localizedDescription). Raw input might be: \(String(describing: try? container.decodeIfPresent(String.self, forKey: .input)))")
              do {
                let genericInputDecoder = try container.superDecoder(forKey: .input)
                finalInputValue = try Value(from: genericInputDecoder)
              } catch {
                anthropicLogger.error("Fallback to generic Value also failed for serverToolUse (name: code_execution): \(error.localizedDescription)")
                finalInputValue = .object([:])
              }
            }
          } else if name == "web_search" {
            do {
              let inputFieldDecoder = try container.superDecoder(forKey: .input)
              finalInputValue = try Value(from: inputFieldDecoder)
            } catch {
              anthropicLogger.warning("Failed to decode 'input' for serverToolUse (name: web_search) using Value(from: Decoder): \(error.localizedDescription). Raw input might be: \(String(describing: try? container.decodeIfPresent(String.self, forKey: .input)))")
              finalInputValue = .object([:])
            }
          } else {
            do {
              let inputFieldDecoder = try container.superDecoder(forKey: .input)
              finalInputValue = try Value(from: inputFieldDecoder)
            } catch {
              anthropicLogger.warning("Failed to decode 'input' for serverToolUse (name: \(name)) using Value(from: Decoder): \(error.localizedDescription). Raw input might be: \(String(describing: try? container.decodeIfPresent(String.self, forKey: .input)))")
              finalInputValue = .object([:])
            }
          }
          serverToolUse = ServerToolUseBlock(id: id, name: name, input: finalInputValue)
        case .webSearchToolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          let content = try container.decode(WebSearchToolResultBlockContent.self, forKey: .content)
          webSearchToolResult = WebSearchToolResultBlock(toolUseId: toolUseId, content: content)
        case .webFetchToolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          let content = try container.decode(WebFetchToolResultBlockContent.self, forKey: .content)
          webFetchToolResult = WebFetchToolResultBlock(toolUseId: toolUseId, content: content)
        case .codeExecutionToolResult:
          codeExecutionToolResult = try CodeExecutionToolResultBlock(from: decoder)
      }
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(type, forKey: .type)
      switch type {
        case .text:
          try container.encodeIfPresent(text, forKey: .text)
          try container.encodeIfPresent(citations, forKey: .citations)
        case .thinking:
          try container.encodeIfPresent(thinking, forKey: .thinking)
          try container.encodeIfPresent(signature, forKey: .signature)
        case .redactedThinking:
          try container.encodeIfPresent(data, forKey: .data)
        case .toolUse:
          if let toolUse {
            try container.encode(toolUse.id, forKey: .id)
            try container.encode(toolUse.name, forKey: .name)
            try container.encode(toolUse.input, forKey: .input)
          }
        case .toolResult:
          if let toolResult {
            try container.encode(toolResult.toolUseId, forKey: .toolUseId)
            try container.encode(toolResult.content, forKey: .content)
            try container.encodeIfPresent(toolResult.isError, forKey: .isError)
          }
        case .image, .document:
          try container.encodeIfPresent(source, forKey: .source)
        case .serverToolUse:
          if let serverToolUse {
            try container.encode(serverToolUse.id, forKey: .id)
            try container.encode(serverToolUse.name, forKey: .name)
            try container.encode(serverToolUse.input, forKey: .input)
          }
        case .webSearchToolResult:
          if let webSearchToolResult {
            try container.encode(webSearchToolResult.toolUseId, forKey: .toolUseId)
            try container.encode(webSearchToolResult.content, forKey: .content)
          }
        case .webFetchToolResult:
          if let webFetchToolResult {
            try container.encode(webFetchToolResult.toolUseId, forKey: .toolUseId)
            try container.encode(webFetchToolResult.content, forKey: .content)
          }
        case .codeExecutionToolResult:
          if let codeExecutionToolResult {
            try container.encode(codeExecutionToolResult.toolUseId, forKey: .toolUseId)
            try container.encode(codeExecutionToolResult.content, forKey: .content)
          }
      }
    }
  }

  enum Citation: Codable {
    case text(TextCitation)
    case webSearch(WebSearchCitation)

    private enum CodingKeys: String, CodingKey {
      case type
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      let type = try container.decode(String.self, forKey: .type)

      switch type {
        case "web_search_result_location":
          self = try .webSearch(WebSearchCitation(from: decoder))
        default:
          self = try .text(TextCitation(from: decoder))
      }
    }

    func encode(to encoder: Encoder) throws {
      switch self {
        case let .text(citation):
          try citation.encode(to: encoder)
        case let .webSearch(citation):
          try citation.encode(to: encoder)
      }
    }

    var displayText: String {
      switch self {
        case let .text(citation): citation.text
        case let .webSearch(citation): citation.citedText
      }
    }

    var url: String? {
      switch self {
        case .text: nil
        case let .webSearch(citation): citation.url
      }
    }

    var title: String? {
      switch self {
        case .text: nil
        case let .webSearch(citation): citation.title
      }
    }
  }

  struct TextCitation: Codable {
    let type: String
    let text: String
    let startIndex: Int
    let endIndex: Int
    let source: String?

    enum CodingKeys: String, CodingKey {
      case type, text
      case startIndex = "start_index"
      case endIndex = "end_index"
      case source
    }
  }

  struct WebSearchCitation: Codable {
    let type: String
    let citedText: String
    let url: String
    let title: String
    let encryptedIndex: String

    enum CodingKeys: String, CodingKey {
      case type
      case citedText = "cited_text"
      case url, title
      case encryptedIndex = "encrypted_index"
    }
  }

  struct ContentBlockSource: Codable {
    let type: String
    let mediaType: String
    let data: String?
    let url: String?

    enum CodingKeys: String, CodingKey {
      case type
      case mediaType = "media_type"
      case data
      case url
    }

    init(type: String, mediaType: String, data: String?, url: String?) {
      self.type = type
      self.mediaType = mediaType
      self.data = data
      self.url = url
    }

    init(raw: [String: Value]) {
      type = raw["type"]?.stringValue ?? "base64"
      mediaType = raw["media_type"]?.stringValue ?? "application/octet-stream"
      data = raw["data"]?.stringValue
      url = raw["url"]?.stringValue
    }
  }

  enum MessageStreamEventType: String, Codable {
    case messageStart = "message_start"
    case messageDelta = "message_delta"
    case messageStop = "message_stop"
    case contentBlockStart = "content_block_start"
    case contentBlockDelta = "content_block_delta"
    case contentBlockStop = "content_block_stop"
    case error
    case ping
  }

  struct MessageStreamEvent: Decodable {
    let type: MessageStreamEventType
    let message: APIMessage?
    let index: Int?
    let contentBlock: ContentBlock?
    let delta: MessageDelta?
    let usage: APIMessage.Usage?
    let error: ErrorInfo?

    private enum CodingKeys: String, CodingKey {
      case type, message, index, delta, usage, error
      case contentBlock = "content_block"
    }

    struct ErrorInfo: Decodable {
      let type: String?
      let message: String?
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      type = try container.decode(MessageStreamEventType.self, forKey: .type)
      message = try container.decodeIfPresent(APIMessage.self, forKey: .message)
      index = try container.decodeIfPresent(Int.self, forKey: .index)
      contentBlock = try container.decodeIfPresent(ContentBlock.self, forKey: .contentBlock)
      delta = try container.decodeIfPresent(MessageDelta.self, forKey: .delta)
      usage = try container.decodeIfPresent(APIMessage.Usage.self, forKey: .usage)
      error = try container.decodeIfPresent(ErrorInfo.self, forKey: .error)
    }
  }

  enum DeltaType: String, Codable {
    case textDelta = "text_delta"
    case thinkingDelta = "thinking_delta"
    case citationsDelta = "citations_delta"
    case inputJsonDelta = "input_json_delta"
    case signatureDelta = "signature_delta"
  }

  struct MessageDelta: Codable {
    let type: DeltaType?
    let text: String?
    let thinking: String?
    let citation: Citation?
    let partialJson: String?
    let signature: String?
    let stopReason: String?
    let stopSequence: String?

    enum CodingKeys: String, CodingKey {
      case type, text, thinking, citation, signature
      case partialJson = "partial_json"
      case stopReason = "stop_reason"
      case stopSequence = "stop_sequence"
    }
  }
}
