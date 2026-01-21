// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

extension AnthropicClient {
  enum Role: String, Codable {
    case user
    case assistant
  }

  struct APIMessage: Codable, Sendable {
    let id: String
    let role: Role
    var content: [ContentBlock]
    var stopReason: String?
    var stopSequence: String?
    var usage: Usage

    struct Usage: Codable, Sendable {
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
    case toolUse = "tool_use"
    case toolResult = "tool_result"
    case serverToolUse = "server_tool_use"
    case webSearchToolResult = "web_search_tool_result"
    case webFetchToolResult = "web_fetch_tool_result"
    case codeExecutionToolResult = "code_execution_tool_result"
    case image
    case document
  }

  struct ContentBlock: Codable, Sendable {
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

    private enum CodingKeys: String, CodingKey {
      case type, text, thinking, citations, toolUse, toolResult, serverToolUse, webSearchToolResult, webFetchToolResult, codeExecutionToolResult, source, signature
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
        case .toolUse:
          let id = try container.decode(String.self, forKey: .id)
          let name = try container.decode(String.self, forKey: .name)
          let inputDecoder = try container.superDecoder(forKey: .input)
          var input: Value
          do {
            // First, check if the input is explicitly null
            let singleValueContainer = try inputDecoder.singleValueContainer()
            if singleValueContainer.decodeNil() {
              input = .null // Represent JSON null as Value.null
            } else {
              // If not null, attempt to decode it as Value using the superDecoder directly.
              // The inputDecoder obtained from superDecoder is the correct decoder to use here.
              input = try Value(from: inputDecoder)
            }
          } catch {
            anthropicLogger.warning("Failed to decode input for toolUse, name: \(name)): \(error.localizedDescription)")
            input = .object([:])
          }
          toolUse = ToolUseBlock(id: id, name: name, input: input)
        case .toolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          let content = try container.decode(String.self, forKey: .content)
          let isError = try container.decodeIfPresent(Bool.self, forKey: .isError)
          toolResult = ToolResultBlock(toolUseId: toolUseId, content: content, isError: isError)
        case .image, .document:
          source = try container.decodeIfPresent(ContentBlockSource.self, forKey: .source)
        case .serverToolUse:
          let id = try container.decode(String.self, forKey: .id)
          let name = try container.decode(String.self, forKey: .name)
          var finalInputValue: Value = .object([:]) // Default
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
              // Fallback: try to decode as generic Value if specific struct fails
              do {
                let genericInputDecoder = try container.superDecoder(forKey: .input)
                finalInputValue = try Value(from: genericInputDecoder)
              } catch {
                anthropicLogger.error("Fallback to generic Value also failed for serverToolUse (name: code_execution): \(error.localizedDescription)")
                finalInputValue = .object([:])
              }
            }
          } else if name == "web_search" {
            // Specific handling for web_search input
            do {
              let inputFieldDecoder = try container.superDecoder(forKey: .input)
              // The input for web_search is expected to be a JSON object,
              // e.g., {"query": "some search query", "other_params": ...}
              // Value(from: inputFieldDecoder) should handle this.
              finalInputValue = try Value(from: inputFieldDecoder)
            } catch {
              anthropicLogger.warning("Failed to decode 'input' for serverToolUse (name: web_search) using Value(from: Decoder): \(error.localizedDescription). Raw input might be: \(String(describing: try? container.decodeIfPresent(String.self, forKey: .input)))")
              // If direct Value decoding fails, it implies a more complex issue
              // or a structure that Value's current implementation cannot handle.
              // For web_search, the input is usually an object. If it's failing,
              // it's crucial to understand what that structure is.
              // As a last resort, you could try to decode it as a more specific known struct if one exists.
              // For now, we'll stick to the generic Value and log the error.
              finalInputValue = .object([:]) // Fallback to empty object
            }
          } else {
            // Generic handler for other server tool uses
            do {
              let inputFieldDecoder = try container.superDecoder(forKey: .input)
              finalInputValue = try Value(from: inputFieldDecoder)
            } catch {
              anthropicLogger.warning("Failed to decode 'input' for serverToolUse (name: \(name)) using Value(from: Decoder): \(error.localizedDescription). Raw input might be: \(String(describing: try? container.decodeIfPresent(String.self, forKey: .input)))")
              finalInputValue = .object([:]) // Fallback
            }
          }
          serverToolUse = ServerToolUseBlock(id: id, name: name, input: finalInputValue)
        case .webSearchToolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          // Decode the 'content' field directly as WebSearchToolResultBlockContent
          let content = try container.decode(WebSearchToolResultBlockContent.self, forKey: .content)
          webSearchToolResult = WebSearchToolResultBlock(toolUseId: toolUseId, content: content)
        case .webFetchToolResult:
          let toolUseId = try container.decode(String.self, forKey: .toolUseId)
          // Decode the 'content' field directly as WebFetchToolResultBlockContent
          let content = try container.decode(WebFetchToolResultBlockContent.self, forKey: .content)
          webFetchToolResult = WebFetchToolResultBlock(toolUseId: toolUseId, content: content)
        case .codeExecutionToolResult:
          // Decode as CodeExecutionToolResultBlock directly
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
        case .toolUse:
          if let toolUse {
            try container.encode(toolUse.id, forKey: .id)
            try container.encode(toolUse.name, forKey: .name)
            try container.encode(toolUse.input.toData(), forKey: .input)
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
            try container.encode(serverToolUse.input.toData(), forKey: .input)
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

  enum Citation: Codable, Sendable {
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

    // Helper to get display text regardless of type
    var displayText: String {
      switch self {
        case let .text(citation): citation.text
        case let .webSearch(citation): citation.citedText
      }
    }

    // Other convenience properties
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

  // Used for retrieval-augmented generation (when documents are included with the request)
  struct TextCitation: Codable, Sendable {
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

  struct WebSearchCitation: Codable, Sendable {
    let type: String // Always "web_search_result_location"
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
  }

  // Stream event types
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

  struct MessageStreamEvent: Decodable, Sendable {
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

    struct ErrorInfo: Decodable, Sendable {
      let type: String?
      let message: String?
      // Add other error fields as needed
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

  struct MessageDelta: Codable, Sendable {
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

extension AnthropicClient {
  actor MessageStream {
    enum Event: Sendable {
      case connect
      case streamEvent(event: MessageStreamEvent, snapshot: APIMessage)
      case text(delta: String, snapshot: String)
      case thinking(delta: String, snapshot: String)
      case citation(citation: Citation, snapshot: [Citation])
      case inputJson(partialJson: String, snapshot: Value)
      case signature(signature: String)
      case message(message: APIMessage)
      case contentBlock(content: ContentBlock)
      case finalMessage(message: APIMessage)
      case toolUse(toolUse: ToolUseBlock)
      case toolResult(toolResult: ToolResultBlock)
      case serverToolUse(serverToolUse: ServerToolUseBlock)
      case webSearchResult(webSearchResult: WebSearchToolResultBlock)
      case webFetchResult(webFetchResult: WebFetchToolResultBlock)
      case codeExecutionResult(codeExecutionResult: CodeExecutionToolResultBlock)
      case error(error: AIError)
      case abort(error: AIError)
      case end
    }

    private var listeners: [UUID: (Event) -> Void] = [:]
    private var messages: [MessageParam] = []
    private var receivedMessages: [APIMessage] = []
    private var currentMessageSnapshot: APIMessage?

    private var ended = false
    private var errored = false
    private var aborted = false

    init() {}

    // Create an AsyncStream for events
    func events() -> AsyncStream<Event> {
      AsyncStream { continuation in
        let id = UUID()
        listeners[id] = { event in
          continuation.yield(event)
          if case .end = event {
            continuation.finish()
          }
        }

        continuation.onTermination = { _ in
          Task { await self.off(id: id) }
        }
      }
    }

    func off(id: UUID) {
      listeners.removeValue(forKey: id)
    }

    func abort() {
      aborted = true
      Task {
        emit(.abort(error: AIError.cancelled))
        emit(.end)
      }
    }

    // MARK: - Stream Processing

    static func createMessage(
      client: AnthropicClient,
      params: MessageCreateParams,
      apiKey: String,
      session: URLSession
    ) -> MessageStream {
      let stream = MessageStream()

      // Start the stream processing in a detached task
      Task.detached {
        do {
          for message in params.messages {
            await stream.addMessageParam(message)
          }
          try await stream.createMessage(client: client, params: params, apiKey: apiKey, session: session)
          await stream.emitFinal()
          await stream.emit(.end)
        } catch {
          await stream.handleError(error)
        }
      }
      return stream
    }

    func addMessageParam(_ message: MessageParam) {
      messages.append(message)
    }

    func addMessage(_ message: APIMessage, emit: Bool = true) {
      receivedMessages.append(message)
      if emit {
        Task { self.emit(.message(message: message)) }
      }
    }

    func createMessage(
      client: AnthropicClient,
      params: MessageCreateParams,
      apiKey: String,
      session: URLSession
    ) async throws {
      beginRequest()
      do {
        let request = try await client.buildMessagesRequest(params: params, stream: true, apiKey: apiKey)
        let (stream, response) = try await session.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
          throw AIError.network(underlying: URLError(.badServerResponse))
        }
        connected()
        if !(200 ... 299).contains(httpResponse.statusCode) {
          var errorData = Data()
          for try await byte in stream {
            try Task.checkCancellation()
            errorData.append(byte)
          }
          throw AnthropicError.aiErrorFromHTTPResponse(status: httpResponse.statusCode, data: errorData)
        }
        for try await jsonString in SSEParser.dataPayloads(from: stream, terminateOnDone: false) {
          // Check for abort
          if aborted {
            throw AIError.cancelled
          }
          guard let jsonData = jsonString.data(using: .utf8) else {
            throw AIError.parsing(message: "Failed to convert line to data: \(jsonString)")
          }
          do {
            let event = try JSONDecoder().decode(MessageStreamEvent.self, from: jsonData)
            await addStreamEvent(event)
          } catch {
            anthropicLogger.error("Failed to decode stream event: \(error). JSON string: \(jsonString)")
            throw AIError.parsing(message: "Failed to decode stream event: \(error.localizedDescription)")
          }
        }
        // Use the result of endRequest
        _ = endRequest()
      } catch let urlError as URLError {
        // Handle specific URL errors
        switch urlError.code {
          case .timedOut:
            throw AIError.timeout
          case .notConnectedToInternet, .networkConnectionLost:
            throw AIError.network(underlying: urlError)
          default:
            throw AIError.network(underlying: urlError)
        }
      } catch {
        // Re-throw if it's already an LLMError
        if let aiError = error as? AIError {
          throw aiError
        } else {
          throw AIError.network(underlying: error)
        }
      }
    }

    func addStreamEvent(_ event: MessageStreamEvent) async {
      if ended { return }
      // Handle error events immediately
      if event.type == .error {
        let errorMessage = event.error?.message ?? "Unknown error"
        //    let errorType = event.error?.type ?? "unknown"
        let error = AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
        handleError(error)
        return
      }
      // Ignore ping events
      if event.type == .ping {
        return
      }
      do {
        let messageSnapshot = try await accumulateMessage(event)
        emit(.streamEvent(event: event, snapshot: messageSnapshot))
        switch event.type {
          case .contentBlockDelta:
            guard let delta = event.delta, let index = event.index, index < messageSnapshot.content.count else { break }
            let content = messageSnapshot.content[index]
            if let deltaType = delta.type {
              switch deltaType {
                case .textDelta:
                  if content.type == .text, let textDelta = delta.text {
                    emit(.text(delta: textDelta, snapshot: content.text ?? ""))
                  }
                case .thinkingDelta:
                  if content.type == .thinking, let thinkingDelta = delta.thinking {
                    emit(.thinking(delta: thinkingDelta, snapshot: content.thinking ?? ""))
                  }
                case .citationsDelta:
                  if content.type == .text, let citation = delta.citation {
                    emit(.citation(citation: citation, snapshot: content.citations ?? []))
                  }
                case .inputJsonDelta:
                  if let partialJson = delta.partialJson {
                    if content.type == .toolUse, let toolUse = content.toolUse {
                      emit(.inputJson(partialJson: partialJson, snapshot: toolUse.input))
                    } else if content.type == .serverToolUse, let serverToolUse = content.serverToolUse {
                      emit(.inputJson(partialJson: partialJson, snapshot: serverToolUse.input))
                    } else {
                      anthropicLogger.warning("Not handling partial JSON delta for unknown content type: \(content.type.rawValue)")
                    }
                  }
                case .signatureDelta:
                  if content.type == .thinking, let signature = delta.signature {
                    emit(.signature(signature: signature))
                  }
              }
            }
          case .messageStop:
            // Convert Message to MessageParam
            let messageParam = MessageParam(
              role: messageSnapshot.role,
              text: messageSnapshot.content.first(where: { $0.type == .text })?.text,
              contentBlocks: nil
            )
            addMessageParam(messageParam)
            addMessage(messageSnapshot, emit: true)
          case .contentBlockStop:
            if let index = event.index, index < messageSnapshot.content.count {
              let content = messageSnapshot.content[index]
              emit(.contentBlock(content: content))
              if content.type == .toolUse, let toolUse = content.toolUse {
                emit(.toolUse(toolUse: toolUse))
              } else if content.type == .serverToolUse, let serverToolUse = content.serverToolUse {
                emit(.serverToolUse(serverToolUse: serverToolUse))
              } else if content.type == .webSearchToolResult, let webSearchResult = content.webSearchToolResult {
                emit(.webSearchResult(webSearchResult: webSearchResult))
              } else if content.type == .webFetchToolResult, let webFetchResult = content.webFetchToolResult {
                emit(.webFetchResult(webFetchResult: webFetchResult))
              } else if content.type == .codeExecutionToolResult, let codeExecutionResult = content.codeExecutionToolResult {
                emit(.codeExecutionResult(codeExecutionResult: codeExecutionResult))
              }
            }
          case .messageStart:
            currentMessageSnapshot = messageSnapshot
          case .messageDelta, .contentBlockStart, .ping, .error:
            break
        }
      } catch {
        if let aiError = error as? AIError {
          handleError(aiError)
        } else {
          let wrappedError = AIError.parsing(message: error.localizedDescription)
          handleError(wrappedError)
        }
      }
    }

    func accumulateMessage(_ event: MessageStreamEvent) async throws -> APIMessage {
      if event.type == .messageStart {
        guard let message = event.message else {
          throw AIError.parsing(message: "Message start event without message")
        }
        currentMessageSnapshot = message
        return message
      }
      // For error events, we don't need to accumulate anything
      if event.type == .error {
        // Create a default empty message to use as a placeholder if needed
        let emptyMessage = APIMessage(
          id: "error_\(UUID().uuidString)",
          role: .assistant,
          content: [],
          usage: APIMessage.Usage()
        )
        return currentMessageSnapshot ?? emptyMessage
      }
      guard var snapshot = currentMessageSnapshot else {
        // Create a more detailed error message
        var errorDetails = "Unexpected event order: got \(event.type) before message_start"

        // Include relevant event properties in the error message
        if let index = event.index {
          errorDetails += ", index=\(index)"
        }
        if let contentBlock = event.contentBlock {
          errorDetails += ", contentBlockType=\(contentBlock.type)"
        }
        if let delta = event.delta {
          errorDetails += ", deltaType=\(delta.type?.rawValue ?? "unknown")"
        }
        anthropicLogger.warning("\(errorDetails)")
        throw AIError.parsing(message: "Unexpected event order: received \(event.type.rawValue) before message_start")
      }
      switch event.type {
        case .messageStop:
          return snapshot
        case .messageDelta:
          if let delta = event.delta {
            snapshot.stopReason = delta.stopReason
            snapshot.stopSequence = delta.stopSequence
          }
          if let usage = event.usage {
            snapshot.usage.outputTokens = usage.outputTokens
          }
        case .contentBlockStart:
          if let contentBlock = event.contentBlock {
            snapshot.content.append(contentBlock)
          }
        case .contentBlockDelta:
          guard let delta = event.delta, let index = event.index, index < snapshot.content.count else { break }
          if let deltaType = delta.type {
            switch deltaType {
              case .textDelta:
                if snapshot.content[index].type == .text, let textDelta = delta.text {
                  snapshot.content[index].text = (snapshot.content[index].text ?? "") + textDelta
                }
              case .thinkingDelta:
                if snapshot.content[index].type == .thinking, let thinkingDelta = delta.thinking {
                  snapshot.content[index].thinking = (snapshot.content[index].thinking ?? "") + thinkingDelta
                }
              case .citationsDelta:
                if snapshot.content[index].type == .text, let citation = delta.citation {
                  var citations = snapshot.content[index].citations ?? []
                  citations.append(citation)
                  snapshot.content[index].citations = citations
                }
              case .inputJsonDelta:
                guard let partialJson = delta.partialJson, let index = event.index, index < snapshot.content.count else { break }
                if snapshot.content[index].type == .toolUse, var toolUse = snapshot.content[index].toolUse {
                  let currentInput = toolUse.input
                  let existingJsonBuf: String = if case let .object(dict) = currentInput,
                                                   case let .string(buf) = dict[Value.jsonBufKey]
                  {
                    buf
                  } else {
                    ""
                  }
                  let jsonBuf = existingJsonBuf + partialJson
                  if let jsonData = jsonBuf.data(using: .utf8) {
                    do {
                      if let parsedJson = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable] {
                        let jsonValue = try Value.fromAny(parsedJson)
                        if case var .object(dict) = jsonValue {
                          dict[Value.jsonBufKey] = .string(jsonBuf)
                          toolUse.input = .object(dict)
                        } else {
                          var dict: [String: Value] = [:]
                          dict["value"] = jsonValue
                          dict[Value.jsonBufKey] = .string(jsonBuf)
                          toolUse.input = .object(dict)
                        }
                      } else {
                        var dict: [String: Value] = [:]
                        dict[Value.jsonBufKey] = .string(jsonBuf)
                        toolUse.input = .object(dict)
                      }
                      snapshot.content[index].toolUse = toolUse
                    } catch {
                      var dict: [String: Value] = [:]
                      dict[Value.jsonBufKey] = .string(jsonBuf)
                      toolUse.input = .object(dict)
                      snapshot.content[index].toolUse = toolUse
                    }
                  }
                } else if snapshot.content[index].type == .serverToolUse, var serverToolUse = snapshot.content[index].serverToolUse, serverToolUse.name == "code_execution" {
                  let existingJsonBuf: String = if case let .object(currentInputDict) = serverToolUse.input,
                                                   case let .string(buf) = currentInputDict[Value.jsonBufKey]
                  {
                    buf
                  } else {
                    ""
                  }
                  let jsonBuf = existingJsonBuf + partialJson
                  if let jsonData = jsonBuf.data(using: .utf8) {
                    struct TempServerToolInput: Decodable { let code: String }
                    // Try to parse the complete {"code": "..."} structure
                    let newInputValue: Value = if let tempInput = try? JSONDecoder().decode(TempServerToolInput.self, from: jsonData) {
                      // Successfully parsed the full JSON for code input
                      .object([
                        "code": .string(tempInput.code),
                        Value.jsonBufKey: .string(jsonBuf), // Keep buffer for potential further (though unlikely) deltas
                      ])
                    } else {
                      // Parsing failed (likely incomplete JSON string), just store the buffer
                      .object([Value.jsonBufKey: .string(jsonBuf)])
                    }
                    serverToolUse.input = newInputValue
                    snapshot.content[index].serverToolUse = serverToolUse
                  }
                }
              case .signatureDelta:
                if snapshot.content[index].type == .thinking, let signature = delta.signature {
                  snapshot.content[index].signature = signature
                }
            }
          }
        case .contentBlockStop:
          break
        case .messageStart, .ping, .error:
          break
      }
      currentMessageSnapshot = snapshot
      return snapshot
    }

    func connected() {
      if ended { return }
      Task { emit(.connect) }
    }

    func beginRequest() {
      if ended { return }
      currentMessageSnapshot = nil
    }

    func endRequest() -> APIMessage? {
      if ended { return nil }
      let snapshot = currentMessageSnapshot
      currentMessageSnapshot = nil
      return snapshot
    }

    func emit(_ event: Event) {
      if ended { return }
      if case .end = event {
        ended = true
      }
      for listener in listeners.values {
        listener(event)
      }
      if case let .abort(error) = event {
        if listeners.isEmpty {
          anthropicLogger.warning("Unhandled abort: \(error)")
        }
        return
      }
      if case let .error(error) = event {
        if listeners.isEmpty {
          anthropicLogger.error("Unhandled error: \(error)")
        }
        ended = true
      }
    }

    func emitFinal() {
      if let finalMessage = receivedMessages.last {
        Task { emit(.finalMessage(message: finalMessage)) }
      }
    }

    func handleError(_ error: Error) {
      errored = true
      if let error = error as? URLError, error.code == .cancelled {
        aborted = true
        let abortError = AIError.cancelled
        Task { emit(.abort(error: abortError)) }
        return
      }
      if let aiError = error as? AIError {
        Task {
          emit(.error(error: aiError))
          emit(.end)
        }
        return
      }
      let wrappedError = AIError.network(underlying: error)
      Task {
        emit(.error(error: wrappedError))
        emit(.end)
      }
    }
  }
}

/// A client for the Anthropic Messages API.
///
/// Supports Claude models with features like tool use, streaming, prompt caching,
/// extended thinking, and web search (via computer use beta).
///
/// ## Example
///
/// ```swift
/// let client = AnthropicClient()
/// let response = try await client.generateText(
///   modelId: "claude-sonnet-4-20250514",
///   prompt: "Hello, Claude!",
///   apiKey: "your-api-key"
/// )
/// print(response.texts.response ?? "")
/// ```
@Observable
public final class AnthropicClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image]

  private let baseURL = URL(string: "https://api.anthropic.com/v1")!
  private let messagesEndpoint: URL

  private let version = "2023-06-01"
  private let maxRetries: Int
  private let timeout: TimeInterval
  private let session: URLSession

  @MainActor public private(set) var isGenerating: Bool = false
  @MainActor private var currentTask: Task<(GenerationResponse, Bool), Error>?

  struct ThinkingConfig: Encodable {
    enum EnabledSetting: String, Encodable {
      case enabled, disabled
    }

    let type: EnabledSetting
    let budgetTokens: Int?

    enum CodingKeys: String, CodingKey {
      case type
      case budgetTokens = "budget_tokens"
    }
  }

  /// Configuration options for Anthropic API requests.
  public struct Configuration: Sendable {
    /// The default thinking budget in tokens (10,000).
    public static let defaultThinkingBudget = 10000

    /// Maximum tokens for extended thinking. Set to enable thinking mode.
    /// The minimum value supported by Anthropic is 1024.
    public var maxThinkingTokens: Int?

    /// Enables web search tool for retrieving information from the internet.
    public var webSearch: Bool

    /// Enables web content fetching for retrieving full page content.
    public var webContent: Bool

    /// Enables code execution in a sandboxed environment.
    public var codeExecution: Bool

    /// A configuration with all features disabled.
    public static let disabled = Configuration()

    var thinkingConfig: ThinkingConfig {
      // The minimum thinking budget for Anthropic is 1024.
      // Gemini uses thinking budget of 0 to turn off thinking.
      // Anthropic client doesn't include a thinking config to turn off thinking.
      // If thinking budget is 0, don't include the thinking config.
      if let maxThinkingTokens, maxThinkingTokens > 0 {
        ThinkingConfig(type: .enabled, budgetTokens: maxThinkingTokens)
      } else {
        ThinkingConfig(type: .disabled, budgetTokens: nil)
      }
    }

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - maxThinkingTokens: Maximum tokens for extended thinking. Minimum is 1024.
    ///   - webSearch: Enable web search tool.
    ///   - webContent: Enable web content fetching.
    ///   - codeExecution: Enable sandboxed code execution.
    public init(maxThinkingTokens: Int? = nil, webSearch: Bool = false, webContent: Bool = false, codeExecution: Bool = false) {
      self.maxThinkingTokens = maxThinkingTokens
      self.webSearch = webSearch
      self.webContent = webContent
      self.codeExecution = codeExecution
    }
  }

  /// Creates a new Anthropic client.
  ///
  /// - Parameters:
  ///   - maxRetries: Maximum number of retry attempts for failed requests.
  ///   - timeout: Request timeout in seconds.
  ///   - session: URLSession to use for requests.
  ///   - messagesEndpoint: Custom endpoint URL for the messages API.
  public init(maxRetries: Int = 2, timeout: TimeInterval = 600, session: URLSession = .shared, messagesEndpoint: URL? = nil) {
    self.maxRetries = maxRetries
    self.timeout = timeout
    self.session = session
    self.messagesEndpoint = messagesEndpoint ?? baseURL.appendingPathComponent("messages")
  }

  func buildMessagesRequest(params: MessageCreateParams, stream: Bool, apiKey: String) async throws -> URLRequest {
    var request = URLRequest(url: messagesEndpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue(version, forHTTPHeaderField: "anthropic-version")
    request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
    request.setValue("AnthropicSwift/1.0", forHTTPHeaderField: "User-Agent")

    // Add beta headers
    var betaHeaders: [String] = []
    // Check if code execution tool is present
    if let tools = params.tools, tools.contains(where: { tool in
      if case .codeExecution = tool { return true }
      return false
    }) {
      betaHeaders.append("code-execution-2025-05-22")
    }

    // Check if web fetch tool is present
    if let tools = params.tools, tools.contains(where: { tool in
      if case .webFetch = tool { return true }
      return false
    }) {
      betaHeaders.append("web-fetch-2025-09-10")
    }

    if !betaHeaders.isEmpty {
      request.setValue(betaHeaders.joined(separator: ","), forHTTPHeaderField: "anthropic-beta")
    }
    // Convert messages to Value array (needs to be done separately due to async image processing)
    var messagesArray: [Value] = []
    for message in params.messages {
      var messageDict: [String: Value] = [
        "role": .string(message.role.rawValue),
      ]
      // Process content with attachments if present
      if let attachments = message.attachments, !attachments.isEmpty {
        var contentArray: [Value] = []
        // Process attachments first
        for attachment in attachments {
          switch attachment.kind {
            case let .image(data, mimeType):
              do {
                // Resize image if necessary before encoding
                let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                contentArray.append(.object([
                  "type": .string("image"),
                  "source": .object([
                    "type": .string("base64"),
                    "media_type": .string(mimeType),
                    "data": .string(processedImageData.base64EncodedString()),
                  ]),
                ]))
              } catch {
                anthropicLogger.error("Failed to process image: \(error.localizedDescription)")
                throw error
              }
            case .video:
              // Not supported
              break
            case .audio:
              // Not supported
              break
            case let .document(data, mimeType):
              if mimeType == "application/pdf" {
                do {
                  // Check PDF size
                  if data.count > 32 * 1024 * 1024 { // 32 MB limit
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
        // Add text content if present
        if let text = message.text, !text.isEmpty {
          contentArray.append(.object([
            "type": .string("text"),
            "text": .string(text),
          ]))
        }
        messageDict["content"] = .array(contentArray)
      } else if let contentBlocks = message.contentBlocks {
        // Handle content blocks
        messageDict["content"] = .array(contentBlocks.map { block -> Value in
          var blockDict: [String: Value] = [
            "type": .string(block.type.rawValue),
          ]
          // Handle different block types
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
                // Handle ToolResultContent which can be text or array of blocks
                switch toolResult.content {
                  case let .text(text):
                    blockDict["content"] = .string(text)
                  case let .blocks(contentBlocks):
                    blockDict["content"] = .array(contentBlocks.map { contentBlock -> Value in
                      var contentBlockDict: [String: Value] = ["type": .string(contentBlock.type)]
                      if let text = contentBlock.text {
                        contentBlockDict["text"] = .string(text)
                      }
                      if let source = contentBlock.source {
                        var sourceDict: [String: Value] = [
                          "type": .string(source.type),
                          "media_type": .string(source.mediaType),
                        ]
                        if let data = source.data {
                          sourceDict["data"] = .string(data)
                        }
                        contentBlockDict["source"] = .object(sourceDict)
                      }
                      return .object(contentBlockDict)
                    })
                }
                if let isError = toolResult.isError {
                  blockDict["is_error"] = .bool(isError)
                }
              }
            case .thinking:
              break
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
        // Simple text content
        messageDict["content"] = .array([
          .object(["type": .string("text"), "text": .string(text)]),
        ])
      }
      messagesArray.append(.object(messageDict))
    }

    // Convert to Value for safe serialization
    var requestBody: [String: Value] = [
      "model": .string(params.model),
      "messages": .array(messagesArray),
      "stream": .bool(stream),
    ]
    // Anthropic requires max_tokens - use provided value or model-specific default
    let effectiveMaxTokens = params.maxTokens ?? Self.defaultMaxTokens(for: params.model)
    requestBody["max_tokens"] = .int(effectiveMaxTokens)
    if let systemPrompt = params.system, !systemPrompt.isEmpty {
      requestBody["system"] = .string(systemPrompt)
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
    // Thinking
    // https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
    if let thinking = params.thinking {
      // If used, budget_tokens must be less than max_tokens and must be at least 1024
      let adjustedBudgetTokens: Int? = if let budgetTokens = thinking.budgetTokens,
                                          let maxTokens = params.maxTokens,
                                          budgetTokens >= maxTokens
      {
        min(budgetTokens - 1, maxTokens - 1)
      } else {
        thinking.budgetTokens
      }
      var thinkingDict: [String: Value] = [
        "type": .string(thinking.type.rawValue),
      ]
      // Only include budget_tokens when type is `enabled`
      if case .enabled = thinking.type, let budgetTokens = adjustedBudgetTokens {
        thinkingDict["budget_tokens"] = .int(budgetTokens)
      }
      requestBody["thinking"] = .object(thinkingDict)
    }
    // Add tool parameters
    if let tools = params.tools, !tools.isEmpty {
      requestBody["tools"] = try .array(tools.compactMap { tool in
        let toolData = try JSONEncoder().encode(tool)
        if let toolJson = try? JSONSerialization.jsonObject(with: toolData, options: []) {
          return try Value.fromAny(toolJson)
        } else {
          anthropicLogger.error("Failed to convert encoded tool to Value: \(String(describing: tool))")
          return nil
        }
      })
    }
    if let toolChoice = params.toolChoice {
      switch toolChoice {
        case .auto:
          requestBody["tool_choice"] = .object(["type": .string("auto")])
        case .any:
          requestBody["tool_choice"] = .object(["type": .string("any")])
        case .none:
          requestBody["tool_choice"] = .object(["type": .string("none")])
        case let .tool(name):
          requestBody["tool_choice"] = .object([
            "type": .string("tool"),
            "name": .string(name),
          ])
      }
    }
    if let disableParallelToolUse = params.disableParallelToolUse {
      requestBody["disable_parallel_tool_use"] = .bool(disableParallelToolUse)
    }
    if let metadata = params.metadata {
      var metadataDict: [String: Value] = [:]
      for (key, value) in metadata {
        metadataDict[key] = .string(value)
      }
      requestBody["metadata"] = .object(metadataDict)
    }
    // Convert Value to JSON data
    request.httpBody = try Value.object(requestBody).toData()
    return request
  }

  func createMessageStream(params: MessageCreateParams, apiKey: String) -> MessageStream {
    MessageStream.createMessage(client: self, params: params, apiKey: apiKey, session: session)
  }

  private func mapRole(_ role: Message.Role) -> AnthropicClient.Role {
    switch role {
      case .user, .tool:
        return .user
      case .assistant:
        return .assistant
      case .system, .developer:
        anthropicLogger.error("Message role \(role.rawValue) not supported in Anthropic API client")
        // Anthropic doesn't have direct equivalents for these roles
        // Default to user for these cases, or handle differently if needed
        return .user
    }
  }

  /// Cancels any ongoing generation task.
  @MainActor
  public func stop() {
    currentTask?.cancel()
  }

  // Helper method to make requests with retries
  private func makeRequest<T: Decodable>(
    endpoint: URL,
    method: String,
    apiKey: String,
    body: [String: any Sendable]? = nil,
    retries: Int? = nil
  ) async throws -> T {
    let retriesRemaining = retries ?? maxRetries
    do {
      var request = URLRequest(url: endpoint)
      request.httpMethod = method
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.setValue(version, forHTTPHeaderField: "anthropic-version")
      request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
      if let body {
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
      }
      let (data, response) = try await session.data(for: request)
      guard let httpResponse = response as? HTTPURLResponse else {
        throw AIError.network(underlying: URLError(.badServerResponse))
      }
      if !(200 ... 299).contains(httpResponse.statusCode) {
        throw AnthropicError.aiErrorFromHTTPResponse(status: httpResponse.statusCode, data: data)
      }
      return try JSONDecoder().decode(T.self, from: data)
    } catch {
      if retriesRemaining > 0, shouldRetry(error) {
        // Calculate backoff with jitter
        let delay = calculateRetryDelay(retriesRemaining: retriesRemaining)
        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        return try await makeRequest(
          endpoint: endpoint,
          method: method,
          apiKey: apiKey,
          body: body,
          retries: retriesRemaining - 1
        )
      }
      throw error
    }
  }

  private func shouldRetry(_ error: Error) -> Bool {
    if let urlError = error as? URLError {
      // Retry network errors
      return urlError.code != .cancelled
    }
    if let aiError = error as? AIError {
      return aiError.isRetryable
    }
    return false
  }

  private func calculateRetryDelay(retriesRemaining: Int) -> Double {
    let initialRetryDelay = 0.5
    let maxRetryDelay = 8.0
    let numRetries = maxRetries - retriesRemaining
    // Apply exponential backoff, but not more than the max
    let sleepSeconds = min(initialRetryDelay * pow(2.0, Double(numRetries)), maxRetryDelay)
    // Apply jitter, take up to at most 25 percent of the retry time
    let jitter = 1.0 - Double.random(in: 0.0 ... 0.25)
    return sleepSeconds * jitter
  }
}

extension AnthropicClient {
  struct MessageParam: Sendable {
    let role: Role
    let text: String?
    let contentBlocks: [ContentBlockParam]?
    let attachments: [Attachment]?

    init(role: Role, text: String?, contentBlocks: [ContentBlockParam]? = nil, attachments: [Attachment]? = nil) {
      self.role = role
      self.text = text
      self.contentBlocks = contentBlocks
      self.attachments = attachments
    }
  }

  struct ContentBlockParam: Codable {
    let type: ContentBlockType
    let text: String?
    let source: ContentBlockSource?
    let toolUse: ToolUseBlockParam?
    let toolResult: ToolResultBlockParam?
    let codeExecutionToolResult: CodeExecutionToolResultBlockParam?
    // Add other fields if necessary
    // ...
  }

  struct ToolUseBlockParam: Codable {
    let id: String
    let name: String
    let input: Value
  }

  struct ToolResultBlockParam: Codable {
    let toolUseId: String
    let content: ToolResultContent
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
      case isError = "is_error"
    }
  }

  /// Content for a tool result - can be a simple string or an array of content blocks
  enum ToolResultContent: Codable {
    case text(String)
    case blocks([ToolResultContentBlock])

    func encode(to encoder: Encoder) throws {
      var container = encoder.singleValueContainer()
      switch self {
        case let .text(string):
          try container.encode(string)
        case let .blocks(blocks):
          try container.encode(blocks)
      }
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      if let string = try? container.decode(String.self) {
        self = .text(string)
      } else if let blocks = try? container.decode([ToolResultContentBlock].self) {
        self = .blocks(blocks)
      } else {
        throw DecodingError.typeMismatch(
          ToolResultContent.self,
          DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected String or Array")
        )
      }
    }
  }

  /// A content block within a tool result (text or image)
  struct ToolResultContentBlock: Codable {
    let type: String
    let text: String?
    let source: ContentBlockSource?

    static func text(_ text: String) -> ToolResultContentBlock {
      ToolResultContentBlock(type: "text", text: text, source: nil)
    }

    static func image(mediaType: String, data: String) -> ToolResultContentBlock {
      ToolResultContentBlock(
        type: "image",
        text: nil,
        source: ContentBlockSource(type: "base64", mediaType: mediaType, data: data, url: nil)
      )
    }
  }

  struct CodeExecutionToolResultBlockParam: Codable {
    let toolUseId: String
    let content: Value

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  struct MessageCreateParams: Sendable {
    let model: String
    let messages: [MessageParam]
    let maxTokens: Int?
    let system: String?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    var tools: [APITool]?
    var toolChoice: ToolChoice?
    let metadata: [String: String]?
    let thinking: ThinkingConfig?
    let disableParallelToolUse: Bool?

    init(
      model: String,
      messages: [MessageParam],
      maxTokens: Int? = nil,
      system: String? = nil,
      temperature: Float? = nil,
      topP: Float? = nil,
      topK: Int? = nil,
      tools: [APITool]? = nil,
      toolChoice: ToolChoice? = nil,
      metadata: [String: String]? = nil,
      thinking: ThinkingConfig? = nil,
      disableParallelToolUse: Bool? = nil
    ) {
      self.model = model
      self.messages = messages
      self.maxTokens = maxTokens
      self.system = system
      self.temperature = temperature
      self.topP = topP
      self.topK = topK
      self.tools = tools
      self.toolChoice = toolChoice
      self.metadata = metadata
      self.thinking = thinking
      self.disableParallelToolUse = disableParallelToolUse
    }
  }

  enum APITool: Codable, Sendable {
    case custom(name: String, description: String, inputSchema: JSONSchema)
    case rawCustom(name: String, description: String, rawInputSchema: [String: Value])
    case webSearch
    case webFetch
    case codeExecution

    private enum CustomCodingKeys: String, CodingKey {
      case name, description
      case inputSchema = "input_schema"
    }

    private enum WebSearchCodingKeys: String, CodingKey {
      case name, type
    }

    private enum WebFetchCodingKeys: String, CodingKey {
      case name, type
    }

    private enum CodeExecutionCodingKeys: String, CodingKey {
      case name, type
    }

    func encode(to encoder: Encoder) throws {
      switch self {
        case let .custom(name, description, inputSchema):
          var container = encoder.container(keyedBy: CustomCodingKeys.self)
          try container.encode(name, forKey: .name)
          try container.encode(description, forKey: .description)
          try container.encode(inputSchema, forKey: .inputSchema)
        case let .rawCustom(name, description, rawInputSchema):
          var container = encoder.container(keyedBy: CustomCodingKeys.self)
          try container.encode(name, forKey: .name)
          try container.encode(description, forKey: .description)
          try container.encode(rawInputSchema, forKey: .inputSchema)
        case .webSearch:
          var container = encoder.container(keyedBy: WebSearchCodingKeys.self)
          try container.encode("web_search", forKey: .name)
          try container.encode("web_search_20250305", forKey: .type)
        case .webFetch:
          var container = encoder.container(keyedBy: WebFetchCodingKeys.self)
          try container.encode("web_fetch", forKey: .name)
          try container.encode("web_fetch_20250910", forKey: .type)
        case .codeExecution:
          var container = encoder.container(keyedBy: CodeExecutionCodingKeys.self)
          try container.encode("code_execution", forKey: .name)
          try container.encode("code_execution_20250522", forKey: .type)
      }
    }

    init(from decoder: Decoder) throws {
      // Attempt to decode as web_search first
      let webSearchContainer = try? decoder.container(keyedBy: WebSearchCodingKeys.self)
      if let name = try? webSearchContainer?.decode(String.self, forKey: .name),
         let type = try? webSearchContainer?.decode(String.self, forKey: .type),
         name == "web_search", type == "web_search_20250305"
      {
        self = .webSearch
        return
      }

      // Attempt to decode as web_fetch
      let webFetchContainer = try? decoder.container(keyedBy: WebFetchCodingKeys.self)
      if let name = try? webFetchContainer?.decode(String.self, forKey: .name),
         let type = try? webFetchContainer?.decode(String.self, forKey: .type),
         name == "web_fetch", type == "web_fetch_20250910"
      {
        self = .webFetch
        return
      }

      // Attempt to decode as code_execution
      let codeExecutionContainer = try? decoder.container(keyedBy: CodeExecutionCodingKeys.self)
      if let name = try? codeExecutionContainer?.decode(String.self, forKey: .name),
         let type = try? codeExecutionContainer?.decode(String.self, forKey: .type),
         name == "code_execution", type == "code_execution_20250522"
      {
        self = .codeExecution
        return
      }

      // Attempt to decode as custom tool
      let customContainer = try decoder.container(keyedBy: CustomCodingKeys.self)
      let name = try customContainer.decode(String.self, forKey: .name)
      let description = try customContainer.decode(String.self, forKey: .description)
      let inputSchema = try customContainer.decode(JSONSchema.self, forKey: .inputSchema)
      self = .custom(name: name, description: description, inputSchema: inputSchema)
    }
  }
}

public extension AnthropicClient {
  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Anthropic model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: The generation response with text and metadata.
  func generateText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init()
  ) async throws -> GenerationResponse {
    try await _generate(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: messages,
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
      update: { _ in }
    )
  }

  /// Streams a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Anthropic model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
  ///   - configuration: Additional configuration options.
  /// - Returns: An async stream of generation responses as they arrive.
  func streamText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init()
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          let finalResponse = try await _generate(
            modelId: modelId,
            tools: tools,
            systemPrompt: systemPrompt,
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            apiKey: apiKey,
            configuration: configuration,
            update: { response in
              continuation.yield(response)
            }
          )
          // Yield the final response with metadata
          continuation.yield(finalResponse)
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { @Sendable _ in
        task.cancel()
      }
    }
  }

  /// Generate a text response using a simple prompt string.
  func generateText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init()
  ) async throws -> GenerationResponse {
    try await generateText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration
    )
  }

  /// Generate a text response with streaming using a simple prompt string.
  func streamText(
    modelId: String,
    tools: [Tool] = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init()
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    streamText(
      modelId: modelId,
      tools: tools,
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration
    )
  }

  private func _generate(
    modelId: String,
    tools: [Tool],
    systemPrompt: String?,
    messages: [Message],
    maxTokens: Int?,
    temperature: Float?,
    apiKey: String?,
    configuration: Configuration,
    update: @Sendable @escaping (GenerationResponse) -> Void
  ) async throws -> GenerationResponse {
    guard let apiKey else {
      throw AIError.authentication(message: "Missing API key")
    }
    await MainActor.run { isGenerating = true }
    // Create a task that can be canceled and returns a result even when cancelled
    let task = Task<(GenerationResponse, Bool), Error> {
      defer {
        Task { @MainActor in
          isGenerating = false
          currentTask = nil
        }
      }
      var fullReasoningText = ""
      var fullResponseText = ""
      var webSearchCitationUrls: Set<String> = [] // Using a set because duplicate URLs are returned in the stream
      var toolUseBlocks: [ToolUseBlock] = []
      var wasCancelled = false
      var finalMessage: APIMessage? = nil
      // Convert Message to AnthropicClient.MessageParam
      let messageParams = messages.map { message in
        if let toolResults = message.toolResults, !toolResults.isEmpty {
          // Handle messages with function results (tool results)
          var contentBlocks: [AnthropicClient.ContentBlockParam] = []
          // Add text content if present
          if let content = message.content, !content.isEmpty {
            contentBlocks.append(AnthropicClient.ContentBlockParam(
              type: .text,
              text: content,
              source: nil,
              toolUse: nil,
              toolResult: nil,
              codeExecutionToolResult: nil
            ))
          }
          // Add function results as tool_result blocks
          // Anthropic natively supports multiple content blocks per tool result.
          // Supported: text, images. Unsupported: audio, non-image files (fallback to text).
          // TODO: Monitor Anthropic API updates for expanded content type support.
          for toolResult in toolResults {
            // Create a tool result block
            let resultContent: AnthropicClient.ToolResultContent

            // Process content items
            var resultContentBlocks: [AnthropicClient.ToolResultContentBlock] = []
            for content in toolResult.content {
              switch content {
                case let .text(text):
                  resultContentBlocks.append(.text(text))
                case let .image(data, mimeType):
                  let mediaType = mimeType ?? "image/png"
                  let base64Data = data.base64EncodedString()
                  resultContentBlocks.append(.image(mediaType: mediaType, data: base64Data))
                case let .audio(data, mimeType):
                  anthropicLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Anthropic. Using fallback text.")
                  resultContentBlocks.append(.text(ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription))
                case let .file(data, mimeType, filename):
                  if mimeType.hasPrefix("image/") {
                    let base64Data = data.base64EncodedString()
                    resultContentBlocks.append(.image(mediaType: mimeType, data: base64Data))
                  } else {
                    anthropicLogger.warning("Tool '\(toolResult.name)' returned a file (\(mimeType)), which is not supported by Anthropic. Using fallback text.")
                    resultContentBlocks.append(.text(ToolResult.Content.file(data, mimeType: mimeType, filename: filename).fallbackDescription))
                  }
              }
            }

            // Use text if single text block, otherwise use blocks array
            if resultContentBlocks.count == 1, let text = resultContentBlocks[0].text, resultContentBlocks[0].source == nil {
              resultContent = .text(text)
            } else if resultContentBlocks.isEmpty {
              resultContent = .text("")
            } else {
              resultContent = .blocks(resultContentBlocks)
            }

            let toolResultBlock = AnthropicClient.ToolResultBlockParam(
              toolUseId: toolResult.id,
              content: resultContent,
              isError: toolResult.isError
            )
            contentBlocks.append(AnthropicClient.ContentBlockParam(
              type: .toolResult,
              text: nil,
              source: nil,
              toolUse: nil,
              toolResult: toolResultBlock,
              codeExecutionToolResult: nil
            ))
          }
          return AnthropicClient.MessageParam(
            role: mapRole(message.role),
            text: nil, // Text is included in contentBlocks
            contentBlocks: contentBlocks,
            attachments: message.attachments.isEmpty ? nil : message.attachments
          )
        } else if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
          // Handle messages with function calls
          var contentBlocks: [AnthropicClient.ContentBlockParam] = []
          // Add text content if present
          if let content = message.content, !content.isEmpty {
            contentBlocks.append(AnthropicClient.ContentBlockParam(
              type: .text,
              text: content,
              source: nil,
              toolUse: nil,
              toolResult: nil,
              codeExecutionToolResult: nil
            ))
          }
          // Add function calls as tool_use blocks
          for toolCall in toolCalls {
            let toolUseBlock = AnthropicClient.ToolUseBlockParam(
              id: toolCall.id,
              name: toolCall.name,
              input: Value.object(toolCall.parameters)
            )
            contentBlocks.append(AnthropicClient.ContentBlockParam(
              type: .toolUse,
              text: nil,
              source: nil,
              toolUse: toolUseBlock,
              toolResult: nil,
              codeExecutionToolResult: nil
            ))
          }
          return AnthropicClient.MessageParam(
            role: mapRole(message.role),
            text: nil, // Text is included in contentBlocks
            contentBlocks: contentBlocks,
            attachments: message.attachments.isEmpty ? nil : message.attachments
          )
        } else {
          // Simple text message
          return AnthropicClient.MessageParam(
            role: mapRole(message.role),
            text: message.content,
            contentBlocks: nil,
            attachments: message.attachments.isEmpty ? nil : message.attachments
          )
        }
      }
      // Temperature must be set to 1 when thinking is enabled.
      let adjustedTemperature = configuration.thinkingConfig.type == .enabled ? 1.0 : temperature

      // Create parameters
      var params = MessageCreateParams(
        model: modelId,
        messages: messageParams,
        maxTokens: maxTokens,
        system: systemPrompt,
        temperature: adjustedTemperature,
        thinking: configuration.thinkingConfig
      )
      // Tools - rawInputSchema is always populated (either explicit or generated from parameters)
      var anthropicTools = tools.map { tool -> AnthropicClient.APITool in
        APITool.rawCustom(
          name: tool.name,
          description: tool.description,
          rawInputSchema: tool.rawInputSchema
        )
      }
      // Web search
      if configuration.webSearch {
        anthropicTools.append(.webSearch)
      }

      // Web fetch
      if configuration.webContent {
        anthropicTools.append(.webFetch)
      }

      // Code execution
      if configuration.codeExecution {
        anthropicTools.append(.codeExecution)
      }

      // Include tools if custom tools or web search tool are present
      if !anthropicTools.isEmpty {
        params.tools = anthropicTools
        params.toolChoice = .auto
      }
      // Create message stream using the provided API key
      let stream = createMessageStream(params: params, apiKey: apiKey)
      // Use AsyncStream for events
      let events = await stream.events()
      do {
        for await event in events {
//          print(event)
          try Task.checkCancellation()
          switch event {
            case let .thinking(delta, _):
              fullReasoningText += delta
              await MainActor.run {
                update(
                  .init(texts: .init(
                    reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                    response: fullResponseText.isEmpty ? nil : fullResponseText,
                    notes: nil
                  ), toolCalls: []))
              }
            case let .text(delta, _):
              fullResponseText += delta
              await MainActor.run {
                update(.init(texts: .init(
                  reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                  response: fullResponseText.isEmpty ? nil : fullResponseText,
                  notes: nil
                ), toolCalls: []))
              }
            case let .toolUse(toolUse):
              toolUseBlocks.append(toolUse)
            case let .serverToolUse(serverToolUse):
              // Handle server tool use (like code execution)
              if serverToolUse.name == "code_execution" {
                // Extract the code from the input
                if case let .object(inputDict) = serverToolUse.input,
                   case let .string(code) = inputDict["code"]
                {
                  // Format as markdown code block
                  let markdownCode = "\n\n```python\n\(code)\n```\n\n"
                  fullResponseText += markdownCode
                  await MainActor.run {
                    update(.init(texts: .init(
                      reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                      response: fullResponseText.isEmpty ? nil : fullResponseText,
                      notes: nil
                    ), toolCalls: []))
                  }
                }
              }
            case let .codeExecutionResult(codeExecutionResultBlock):
              // Handle code execution results by iterating through its content array
              for item in codeExecutionResultBlock.content {
                switch item {
                  case let .result(resultDetails):
                    // Extract stdout, stderr, and return_code
                    let stdout = resultDetails.stdout
                    let stderr = resultDetails.stderr
                    let returnCode = resultDetails.returnCode
                    // Format the output
                    var resultText = ""
                    if !stdout.isEmpty {
                      resultText += "\n\n```\n\(stdout)\(stdout.hasSuffix("\n") ? "" : "\n")```\n\n"
                    }
                    if !stderr.isEmpty {
                      resultText += "\n\n**Error:**\n```\n\(stderr)\(stderr.hasSuffix("\n") ? "" : "\n")```\n\n"
                    }
                    if returnCode != 0 || !stderr.isEmpty {
                      if resultText.isEmpty { resultText += "\n\n" } // Ensure spacing if only return code
                      resultText += "```Exit code: \(returnCode)```\n\n"
                    }
                    fullResponseText += resultText
                    await MainActor.run {
                      update(.init(texts: .init(
                        reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                        response: fullResponseText.isEmpty ? nil : fullResponseText,
                        notes: nil
                      ), toolCalls: []))
                    }
                  case let .error(errorDetails):
                    // Handle errors
                    let errorCode = errorDetails.errorCode
                    let errorText = "\n\n**Code execution error:** \(errorCode)\n\n"
                    fullResponseText += errorText
                    await MainActor.run {
                      update(.init(texts: .init(
                        reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                        response: fullResponseText.isEmpty ? nil : fullResponseText,
                        notes: nil
                      ), toolCalls: []))
                    }
                  case let .output(outputBlock): // New case for file outputs
                    let fileOutputText = "\n\n**File Output Generated:** `\(outputBlock.fileId)` (Content not displayed)\n\n"
                    fullResponseText += fileOutputText
                    await MainActor.run {
                      update(.init(texts: .init(
                        reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                        response: fullResponseText.isEmpty ? nil : fullResponseText,
                        notes: nil
                      ), toolCalls: []))
                    }
                }
              }
            case let .citation(citation, _):
              switch citation {
                case .text:
                  break
                case let .webSearch(webSearchCitation):
                  webSearchCitationUrls.insert(webSearchCitation.url)
                  update(.init(texts: .init(
                    reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                    response: fullResponseText.isEmpty ? nil : fullResponseText,
                    notes: nil
                  ), toolCalls: []))
              }
            case let .finalMessage(message):
              finalMessage = message
            case let .error(error):
              throw error
            case let .abort(error):
              wasCancelled = true
              throw error
            case .end:
              let toolCalls = toolUseBlocks.map { toolUseBlock -> GenerationResponse.ToolCall in
                let name = toolUseBlock.name
                var parameters: [String: Value] = [:]
                if case let .object(dict) = toolUseBlock.input {
                  // Copy all values except the internal jsonBuf key
                  for (key, value) in dict {
                    if key != Value.jsonBufKey {
                      parameters[key] = value
                    }
                  }
                }
                return .init(name: name, id: toolUseBlock.id, parameters: parameters)
              }

              // Build metadata from final message
              let metadata: GenerationResponse.Metadata?
              if let msg = finalMessage {
                let finishReason: GenerationResponse.FinishReason? = switch msg.stopReason {
                  case "end_turn", "stop_sequence": .stop
                  case "max_tokens": .maxTokens
                  case "tool_use": .toolUse
                  case .some: .other
                  case .none: nil
                }
                metadata = GenerationResponse.Metadata(
                  responseId: msg.id,
                  finishReason: finishReason,
                  inputTokens: msg.usage.inputTokens,
                  outputTokens: msg.usage.outputTokens,
                  cacheCreationInputTokens: msg.usage.cacheCreationInputTokens,
                  cacheReadInputTokens: msg.usage.cacheReadInputTokens
                )
              } else {
                metadata = nil
              }

              return (GenerationResponse(texts: .init(
                reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
                response: fullResponseText.isEmpty ? nil : fullResponseText,
                notes: formatEndnotesList(urlStrings: Array(webSearchCitationUrls))
              ), toolCalls: toolCalls, metadata: metadata), wasCancelled)
            default:
              continue
          }
        }
      } catch {
        // Check if the error is due to user cancellation
        if error is CancellationError {
          // Abort the stream to stop the background processing task
          await stream.abort()
          wasCancelled = true
        } else if let aiError = error as? AIError, case .cancelled = aiError {
          // Don't show error in UI
          wasCancelled = true
        } else if let aiError = error as? AIError {
          throw aiError
        } else {
          throw AIError.network(underlying: error)
        }
      }
      // Return the current results and cancellation status
      // Build metadata from final message if available
      let metadata: GenerationResponse.Metadata?
      if let msg = finalMessage {
        let finishReason: GenerationResponse.FinishReason? = switch msg.stopReason {
          case "end_turn", "stop_sequence": .stop
          case "max_tokens": .maxTokens
          case "tool_use": .toolUse
          case .some: .other
          case .none: nil
        }
        metadata = GenerationResponse.Metadata(
          responseId: msg.id,
          finishReason: finishReason,
          inputTokens: msg.usage.inputTokens,
          outputTokens: msg.usage.outputTokens,
          cacheCreationInputTokens: msg.usage.cacheCreationInputTokens,
          cacheReadInputTokens: msg.usage.cacheReadInputTokens
        )
      } else {
        metadata = nil
      }
      let result = GenerationResponse(texts: .init(
        reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
        response: fullResponseText.isEmpty ? nil : fullResponseText,
        notes: formatEndnotesList(urlStrings: Array(webSearchCitationUrls))
      ), toolCalls: [], metadata: metadata)
      return (result, wasCancelled)
    }
    // Store the task so we can cancel it
    await MainActor.run {
      currentTask = task
    }
    do {
      let (result, wasCancelled) = try await task.value
      // If the task was cancelled, we don't throw an error but just return the partial results
      if wasCancelled {
        return result
      }
      return result
    } catch {
      // For non-cancellation errors, propagate them
      if let aiError = error as? AIError {
        throw aiError
      } else if error is CancellationError {
        // This should be rare since we handle cancellation in the task
        return .init(texts: .init(reasoning: nil, response: nil, notes: nil), toolCalls: [])
      } else {
        throw AIError.network(underlying: error)
      }
    }
  }

  private func formatEndnotesList(urlStrings: [String]) -> String? {
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

// Model defaults

extension AnthropicClient {
  /// Returns the default max_tokens for a given model ID.
  /// Newer models default to 64000; older models use their documented limits.
  static func defaultMaxTokens(for modelId: String) -> Int {
    if modelId.contains("claude-3-5-haiku") {
      8192
    } else if modelId.contains("claude-3-haiku") {
      4096
    } else if modelId.contains("claude-opus-4-1") {
      32000
    } else {
      64000
    }
  }
}

// Tool use

extension AnthropicClient {
  struct JSONSchema: Codable, Sendable {
    let type: String
    let properties: [String: JSONSchemaProperty]?
    let required: [String]?

    init(type: String, properties: [String: JSONSchemaProperty]? = nil, required: [String]? = nil) {
      self.type = type
      self.properties = properties
      self.required = required
    }
  }

  final class JSONSchemaProperty: Codable, Sendable {
    let type: String
    let description: String?
    let enumValues: [String]?
    let items: JSONSchemaProperty?

    enum CodingKeys: String, CodingKey {
      case type
      case description
      case enumValues = "enum"
      case items
    }

    init(type: String, description: String? = nil, enumValues: [String]? = nil, items: JSONSchemaProperty? = nil) {
      self.type = type
      self.description = description
      self.enumValues = enumValues
      self.items = items
    }

    /// Creates a JSONSchemaProperty from a Tool.ParameterType.
    static func from(_ paramType: Tool.ParameterType, description: String, enumValues: [String]? = nil) -> JSONSchemaProperty {
      switch paramType {
        case .string:
          JSONSchemaProperty(type: "string", description: description, enumValues: enumValues)
        case .float, .integer:
          JSONSchemaProperty(type: "number", description: description)
        case .boolean:
          JSONSchemaProperty(type: "boolean", description: description)
        case let .array(itemType):
          JSONSchemaProperty(type: "array", description: description, items: from(itemType, description: ""))
        case .object:
          JSONSchemaProperty(type: "object", description: description)
      }
    }
  }

  enum ToolChoice: Codable, Sendable {
    case auto
    case any
    case none
    case tool(name: String)

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)

      switch self {
        case .auto:
          try container.encode("auto", forKey: .type)
        case .any:
          try container.encode("any", forKey: .type)
        case .none:
          try container.encode("none", forKey: .type)
        case let .tool(name):
          try container.encode("tool", forKey: .type)
          try container.encode(name, forKey: .name)
      }
    }

    enum CodingKeys: String, CodingKey {
      case type, name
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      let type = try container.decode(String.self, forKey: .type)

      switch type {
        case "auto":
          self = .auto
        case "any":
          self = .any
        case "none":
          self = .none
        case "tool":
          let name = try container.decode(String.self, forKey: .name)
          self = .tool(name: name)
        default:
          throw DecodingError.dataCorruptedError(
            forKey: .type,
            in: container,
            debugDescription: "Invalid tool choice type: \(type)"
          )
      }
    }
  }

  struct ToolUseBlock: Codable, Sendable {
    let id: String
    let name: String
    var input: Value

    enum CodingKeys: String, CodingKey {
      case id, name, input
    }
  }

  struct ToolResultBlock: Codable, Sendable {
    let toolUseId: String
    let content: String
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
      case isError = "is_error"
    }
  }

  struct ServerToolUseBlock: Codable, Sendable {
    let id: String
    let name: String
    var input: Value
  }

  struct WebSearchToolResultBlock: Codable, Sendable {
    let toolUseId: String
    let content: WebSearchToolResultBlockContent

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  struct CodeExecutionToolResultBlock: Codable, Sendable {
    let toolUseId: String
    let content: [CodeExecutionContentItem]

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      toolUseId = try container.decode(String.self, forKey: .toolUseId)
      // Try to decode content as an array first
      do {
        content = try container.decode([CodeExecutionContentItem].self, forKey: .content)
      } catch {
        // If array decoding fails, try decoding as a single item and wrap it
        let singleItem = try container.decode(CodeExecutionContentItem.self, forKey: .content)
        content = [singleItem]
      }
    }

    // If you add a custom init(from:), you might need to provide encode(to:) if default isn't sufficient
    // For this change, the default encode should be fine as `content` is now always an array internally.
    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(toolUseId, forKey: .toolUseId)
      try container.encode(content, forKey: .content)
    }
  }

  // Define the structure for successful code execution content
  struct CodeExecutionResultContent: Codable, Sendable {
    let type: String // Should be "code_execution_result"
    let stdout: String
    let stderr: String
    let returnCode: Int
    let content: [CodeExecutionOutputBlock]

    enum CodingKeys: String, CodingKey {
      case type, stdout, stderr, content
      case returnCode = "return_code"
    }
  }

  // Define the structure for code execution error content
  struct CodeExecutionToolResultErrorContent: Codable, Sendable {
    let type: String // Should be "code_execution_tool_result_error"
    let errorCode: String

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  enum CodeExecutionContentItem: Codable, Sendable {
    case result(CodeExecutionResultContent)
    case error(CodeExecutionToolResultErrorContent)
    case output(CodeExecutionOutputBlock) // New case

    private enum TypeCodingKey: String, CodingKey {
      case type
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: TypeCodingKey.self)
      let type = try container.decode(String.self, forKey: .type)

      switch type {
        case "code_execution_result":
          self = try .result(CodeExecutionResultContent(from: decoder))
        case "code_execution_tool_result_error":
          self = try .error(CodeExecutionToolResultErrorContent(from: decoder))
        case "code_execution_output": // Handle new case
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
        case let .error(content):
          try content.encode(to: encoder)
        case let .output(content): // Handle new case
          try content.encode(to: encoder)
      }
    }
  }

  // Define the structure for file outputs from code execution
  struct CodeExecutionOutputBlock: Codable, Sendable {
    let fileId: String
    let type: String // Should be "code_execution_output"

    enum CodingKeys: String, CodingKey {
      case fileId = "file_id"
      case type
    }
  }

  // MARK: - Web Search Tool Result Types

  enum WebSearchErrorCode: String, Codable, Sendable {
    case invalidToolInput = "invalid_tool_input"
    case unavailable
    case maxUsesExceeded = "max_uses_exceeded"
    case tooManyRequests = "too_many_requests"
    case queryTooLong = "query_too_long"
    // Potentially other generic error codes if the API can send them here.
  }

  struct WebSearchErrorDetails: Codable, Sendable {
    let type: String // Should be "web_search_tool_result_error"
    let errorCode: WebSearchErrorCode

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  struct WebSearchResultItem: Codable, Sendable {
    let encryptedContent: String
    let pageAge: String?
    let title: String
    let type: String // Should be "web_search_result"
    let url: String

    enum CodingKeys: String, CodingKey {
      case encryptedContent = "encrypted_content"
      case pageAge = "page_age"
      case title, type, url
    }
  }

  enum WebSearchToolResultBlockContent: Codable, Sendable {
    case results(items: [WebSearchResultItem])
    case error(details: WebSearchErrorDetails)

    // Custom Codable implementation is needed because the JSON is either
    // an array (for results) or an object (for error).
    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      // Try decoding as an array of results first
      if let items = try? container.decode([WebSearchResultItem].self) {
        self = .results(items: items)
        return
      }
      // If that fails, try decoding as an error object
      if let errorDetails = try? container.decode(WebSearchErrorDetails.self) {
        // We need to ensure the type field in the error object is correct.
        // The WebSearchErrorDetails struct already decodes 'type'.
        if errorDetails.type == "web_search_tool_result_error" {
          self = .error(details: errorDetails)
          return
        }
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

  // MARK: - Web Fetch Tool Result Types

  struct WebFetchToolResultBlock: Codable, Sendable {
    let toolUseId: String
    let content: WebFetchToolResultBlockContent

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  enum WebFetchToolResultBlockContent: Codable, Sendable {
    case result(WebFetchResult)
    case error(WebFetchErrorDetails)

    init(from decoder: Decoder) throws {
      // Try to decode as error first
      if let errorDetails = try? WebFetchErrorDetails(from: decoder),
         errorDetails.type.contains("error")
      {
        self = .error(errorDetails)
        return
      }

      // Otherwise decode as result
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

  struct WebFetchResult: Codable, Sendable {
    let type: String // "web_fetch_result"
    let url: String
    let retrievedAt: String
    let content: WebFetchContent

    enum CodingKeys: String, CodingKey {
      case type, url, content
      case retrievedAt = "retrieved_at"
    }
  }

  struct WebFetchErrorDetails: Codable, Sendable {
    let type: String // Contains "error"
    let errorCode: String

    enum CodingKeys: String, CodingKey {
      case type
      case errorCode = "error_code"
    }
  }

  struct WebFetchContent: Codable, Sendable {
    let type: String // "document"
    let source: WebFetchDocumentSource
    let title: String?
  }

  struct WebFetchDocumentSource: Codable, Sendable {
    let type: String // "text" or "base64"
    let mediaType: String
    let data: String

    enum CodingKeys: String, CodingKey {
      case type, data
      case mediaType = "media_type"
    }
  }
}

extension AnthropicClient {
  enum AnthropicError: LocalizedError {
    case api(status: Int, error: APIErrorDetails?, message: String?)
    case connection(message: String, cause: Error?)
    case connectionTimeout
    case userAbort
    case rateLimit(message: String, headers: [String: String]?)
    case authentication(message: String)
    case permissionDenied(message: String)
    case notFound(message: String)
    case invalidRequest(message: String)
    case server(message: String)
    case parsing(message: String)
    case unexpectedEventOrder(event: String, expectedPriorEvent: String)
    case missingApiKey

    struct APIErrorDetails: Sendable, Codable {
      let type: String?
      let message: String?
      let param: String?
      let code: String?
    }

    var errorDescription: String? {
      switch self {
        case let .api(_, _, message):
          message ?? "API error occurred"
        case let .connection(message, _):
          "Connection error: \(message)"
        case .connectionTimeout:
          "Request timed out"
        case .userAbort:
          "Request was aborted"
        case let .rateLimit(message, _):
          "Rate limit exceeded: \(message)"
        case let .authentication(message):
          "Authentication error: \(message)"
        case let .permissionDenied(message):
          "Permission denied: \(message)"
        case let .notFound(message):
          "Not found: \(message)"
        case let .invalidRequest(message):
          "Invalid request: \(message)"
        case let .server(message):
          "Server error: \(message)"
        case let .parsing(message):
          "Parsing error: \(message)"
        case let .unexpectedEventOrder(event, expectedPriorEvent):
          "Unexpected event order: received \(event) before \(expectedPriorEvent)"
        case .missingApiKey:
          "Missing API key"
      }
    }

    static func aiErrorFromHTTPResponse(status: Int, data: Data) -> AIError {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      if let errorResponse = try? decoder.decode(ErrorResponse.self, from: data) {
        let message = errorResponse.error.message ?? "Unknown error"
        switch status {
          case 400:
            return .invalidRequest(message: message)
          case 401:
            return .authentication(message: "There's an issue with your API key")
          case 403:
            return .authentication(message: "Your API key does not have permission to use the specified resource")
          case 404:
            return .invalidRequest(message: "Not found: \(message)")
          case 429:
            return .rateLimit(retryAfter: nil)
          case 500 ... 599:
            return .serverError(statusCode: status, message: message, context: nil)
          default:
            return .serverError(statusCode: status, message: message, context: nil)
        }
      }

      return .serverError(statusCode: status, message: "Unknown error", context: nil)
    }

    struct ErrorResponse: Codable {
      let error: ErrorDetails

      struct ErrorDetails: Codable {
        let type: String?
        let message: String?
        let param: String?
        let code: String?
      }
    }
  }
}

private let anthropicLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "AnthropicClient")
