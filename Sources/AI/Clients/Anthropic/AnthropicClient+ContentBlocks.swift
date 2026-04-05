// Copyright © Anthony DePasquale

import Foundation

extension AnthropicClient {
  struct ToolUseBlock: Codable {
    let id: String
    let name: String
    var input: Value

    enum CodingKeys: String, CodingKey {
      case id, name, input
    }
  }

  struct ToolResultBlock: Codable {
    let toolUseId: String
    let content: ToolResultContent
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
      case isError = "is_error"
    }
  }

  struct ServerToolUseBlock: Codable {
    let id: String
    let name: String
    var input: Value
  }

  struct WebSearchToolResultBlock: Codable {
    let toolUseId: String
    let content: WebSearchToolResultBlockContent

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  struct CodeExecutionToolResultBlock: Codable {
    let toolUseId: String
    let content: [CodeExecutionContentItem]

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      toolUseId = try container.decode(String.self, forKey: .toolUseId)
      do {
        content = try container.decode([CodeExecutionContentItem].self, forKey: .content)
      } catch {
        let singleItem = try container.decode(CodeExecutionContentItem.self, forKey: .content)
        content = [singleItem]
      }
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(toolUseId, forKey: .toolUseId)
      try container.encode(content, forKey: .content)
    }
  }

  struct CodeExecutionResultContent: Codable {
    let type: String
    let stdout: String
    let stderr: String
    let returnCode: Int
    let content: [CodeExecutionOutputBlock]

    enum CodingKeys: String, CodingKey {
      case type, stdout, stderr, content
      case returnCode = "return_code"
    }
  }

  struct CodeExecutionToolResultErrorContent: Codable {
    let type: String
    let errorCode: String

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  struct EncryptedCodeExecutionResultContent: Codable {
    let type: String
    let encryptedStdout: String
    let stderr: String
    let returnCode: Int
    let content: [CodeExecutionOutputBlock]

    enum CodingKeys: String, CodingKey {
      case type, stderr, content
      case encryptedStdout = "encrypted_stdout"
      case returnCode = "return_code"
    }
  }

  enum CodeExecutionContentItem: Codable {
    case result(CodeExecutionResultContent)
    case encryptedResult(EncryptedCodeExecutionResultContent)
    case error(CodeExecutionToolResultErrorContent)
    case output(CodeExecutionOutputBlock)

    private enum TypeCodingKey: String, CodingKey {
      case type
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: TypeCodingKey.self)
      let type = try container.decode(String.self, forKey: .type)

      switch type {
        case "code_execution_result":
          self = try .result(CodeExecutionResultContent(from: decoder))
        case "encrypted_code_execution_result":
          self = try .encryptedResult(EncryptedCodeExecutionResultContent(from: decoder))
        case "code_execution_tool_result_error":
          self = try .error(CodeExecutionToolResultErrorContent(from: decoder))
        case "code_execution_output":
          self = try .output(CodeExecutionOutputBlock(from: decoder))
        default:
          let context = DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unknown code execution content item type: \(type)")
          throw DecodingError.dataCorrupted(context)
      }
    }

    func encode(to encoder: Encoder) throws {
      switch self {
        case let .result(content):
          try content.encode(to: encoder)
        case let .encryptedResult(content):
          try content.encode(to: encoder)
        case let .error(content):
          try content.encode(to: encoder)
        case let .output(content):
          try content.encode(to: encoder)
      }
    }
  }

  struct CodeExecutionOutputBlock: Codable {
    let fileId: String
    let type: String

    enum CodingKeys: String, CodingKey {
      case fileId = "file_id"
      case type
    }
  }

  enum WebSearchErrorCode: String, Codable {
    case invalidToolInput = "invalid_tool_input"
    case unavailable
    case maxUsesExceeded = "max_uses_exceeded"
    case tooManyRequests = "too_many_requests"
    case queryTooLong = "query_too_long"
  }

  struct WebSearchErrorDetails: Codable {
    let type: String
    let errorCode: WebSearchErrorCode

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  struct WebSearchResultItem: Codable {
    let encryptedContent: String
    let pageAge: String?
    let title: String
    let type: String
    let url: String

    enum CodingKeys: String, CodingKey {
      case encryptedContent = "encrypted_content"
      case pageAge = "page_age"
      case title, type, url
    }
  }

  enum WebSearchToolResultBlockContent: Codable {
    case results(items: [WebSearchResultItem])
    case error(details: WebSearchErrorDetails)

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      if let items = try? container.decode([WebSearchResultItem].self) {
        self = .results(items: items)
        return
      }
      if let errorDetails = try? container.decode(WebSearchErrorDetails.self),
         errorDetails.type == "web_search_tool_result_error"
      {
        self = .error(details: errorDetails)
        return
      }
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "WebSearchToolResultBlockContent could not be decoded as either [WebSearchResultItem] or WebSearchErrorDetails with type 'web_search_tool_result_error'.")
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.singleValueContainer()
      switch self {
        case let .results(items):
          try container.encode(items)
        case let .error(details):
          try container.encode(details)
      }
    }
  }

  struct WebFetchToolResultBlock: Codable {
    let toolUseId: String
    let content: WebFetchToolResultBlockContent

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  enum WebFetchToolResultBlockContent: Codable {
    case result(WebFetchResult)
    case error(WebFetchErrorDetails)

    init(from decoder: Decoder) throws {
      if let errorDetails = try? WebFetchErrorDetails(from: decoder),
         errorDetails.type.contains("error")
      {
        self = .error(errorDetails)
        return
      }

      let result = try WebFetchResult(from: decoder)
      self = .result(result)
    }

    func encode(to encoder: Encoder) throws {
      switch self {
        case let .result(result):
          try result.encode(to: encoder)
        case let .error(error):
          try error.encode(to: encoder)
      }
    }
  }

  struct WebFetchResult: Codable {
    let type: String
    let url: String
    let retrievedAt: String
    let content: WebFetchContent

    enum CodingKeys: String, CodingKey {
      case type, url, content
      case retrievedAt = "retrieved_at"
    }
  }

  struct WebFetchErrorDetails: Codable {
    let type: String
    let errorCode: String

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  struct WebFetchContent: Codable {
    let type: String
    let source: WebFetchDocumentSource
    let title: String?
  }

  struct WebFetchDocumentSource: Codable {
    let type: String
    let mediaType: String
    let data: String

    enum CodingKeys: String, CodingKey {
      case type, data
      case mediaType = "media_type"
    }
  }
}
