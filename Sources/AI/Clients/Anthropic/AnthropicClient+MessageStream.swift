// Copyright © Anthony DePasquale

import Foundation
import SSE

extension AnthropicClient {
  actor MessageStream {
    enum Event {
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
    private var processingTask: Task<Void, Never>?

    init() {}

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
      processingTask?.cancel()
      emit(.abort(error: AIError.cancelled))
      emit(.end)
    }

    static func createMessage(
      client: AnthropicClient,
      params: MessageCreateParams,
      apiKey: String,
      session: URLSession,
    ) async -> MessageStream {
      let stream = MessageStream()

      let task = Task.detached {
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
      await stream.setProcessingTask(task)
      return stream
    }

    private func setProcessingTask(_ task: Task<Void, Never>) {
      processingTask = task
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
      session: URLSession,
    ) async throws {
      beginRequest()

      let byteStream: URLSession.AsyncBytes
      var retriesRemaining = client.retryHandler.maxRetries
      while true {
        var lastResponseHeaders: [AnyHashable: Any]?
        do {
          let request = try await client.buildMessagesRequest(params: params, stream: true, apiKey: apiKey)
          let (stream, response) = try await session.bytes(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          lastResponseHeaders = httpResponse.allHeaderFields
          connected()
          if !(200 ... 299).contains(httpResponse.statusCode) {
            var errorData = Data()
            for try await byte in stream {
              try Task.checkCancellation()
              errorData.append(byte)
            }
            throw AnthropicClient.aiErrorFromHTTPResponse(httpResponse: httpResponse, data: errorData)
          }
          byteStream = stream
          break
        } catch let urlError as URLError {
          let aiError: AIError = switch urlError.code {
            case .timedOut: .timeout
            default: .network(underlying: urlError)
          }
          if retriesRemaining > 0, client.retryHandler.shouldRetry(aiError, responseHeaders: lastResponseHeaders) {
            let delay = client.retryHandler.retryDelay(retriesRemaining: retriesRemaining, responseHeaders: lastResponseHeaders)
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            retriesRemaining -= 1
            continue
          }
          throw aiError
        } catch {
          let aiError = (error as? AIError) ?? .network(underlying: error)
          if retriesRemaining > 0, client.retryHandler.shouldRetry(aiError, responseHeaders: lastResponseHeaders) {
            let delay = client.retryHandler.retryDelay(retriesRemaining: retriesRemaining, responseHeaders: lastResponseHeaders)
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            retriesRemaining -= 1
            continue
          }
          throw aiError
        }
      }

      do {
        for try await event in byteStream.events {
          try Task.checkCancellation()

          if aborted {
            throw AIError.cancelled
          }

          let jsonString = event.data

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
        _ = endRequest()
      } catch let urlError as URLError {
        switch urlError.code {
          case .timedOut:
            throw AIError.timeout
          default:
            throw AIError.network(underlying: urlError)
        }
      } catch {
        if let aiError = error as? AIError {
          throw aiError
        }
        throw AIError.network(underlying: error)
      }
    }

    func addStreamEvent(_ event: MessageStreamEvent) async {
      if ended { return }
      if event.type == .error {
        let errorMessage = event.error?.message ?? "Unknown error"
        let error = AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
        handleError(error)
        return
      }
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
            let contentBlockParams: [ContentBlockParam] = messageSnapshot.content.compactMap { block in
              switch block.type {
                case .thinking:
                  return ContentBlockParam(type: .thinking, thinking: block.thinking, signature: block.signature)
                case .redactedThinking:
                  return ContentBlockParam(type: .redactedThinking, data: block.data)
                case .text:
                  return ContentBlockParam(type: .text, text: block.text)
                case .toolUse:
                  guard let toolUse = block.toolUse else { return nil }
                  return ContentBlockParam(type: .toolUse, toolUse: ToolUseBlockParam(id: toolUse.id, name: toolUse.name, input: toolUse.input))
                case .toolResult:
                  guard let toolResult = block.toolResult else { return nil }
                  return ContentBlockParam(type: .toolResult, toolResult: ToolResultBlockParam(toolUseId: toolResult.toolUseId, content: toolResult.content, isError: toolResult.isError))
                default:
                  return nil
              }
            }
            let messageParam = MessageParam(
              role: messageSnapshot.role,
              text: nil,
              contentBlocks: contentBlockParams.isEmpty ? nil : contentBlockParams,
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
      if event.type == .error {
        let emptyMessage = APIMessage(
          id: "error_\(UUID().uuidString)",
          role: .assistant,
          content: [],
          usage: APIMessage.Usage(),
        )
        return currentMessageSnapshot ?? emptyMessage
      }
      guard var snapshot = currentMessageSnapshot else {
        var errorDetails = "Unexpected event order: got \(event.type) before message_start"

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
            if let inputTokens = usage.inputTokens {
              snapshot.usage.inputTokens = inputTokens
            }
            if let cacheCreationInputTokens = usage.cacheCreationInputTokens {
              snapshot.usage.cacheCreationInputTokens = cacheCreationInputTokens
            }
            if let cacheReadInputTokens = usage.cacheReadInputTokens {
              snapshot.usage.cacheReadInputTokens = cacheReadInputTokens
            }
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
                } else if snapshot.content[index].type == .serverToolUse, var serverToolUse = snapshot.content[index].serverToolUse {
                  let existingJsonBuf: String = if case let .object(currentInputDict) = serverToolUse.input,
                                                   case let .string(buf) = currentInputDict[Value.jsonBufKey]
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
                          serverToolUse.input = .object(dict)
                        } else {
                          var dict: [String: Value] = [:]
                          dict["value"] = jsonValue
                          dict[Value.jsonBufKey] = .string(jsonBuf)
                          serverToolUse.input = .object(dict)
                        }
                      } else {
                        var dict: [String: Value] = [:]
                        dict[Value.jsonBufKey] = .string(jsonBuf)
                        serverToolUse.input = .object(dict)
                      }
                      snapshot.content[index].serverToolUse = serverToolUse
                    } catch {
                      var dict: [String: Value] = [:]
                      dict[Value.jsonBufKey] = .string(jsonBuf)
                      serverToolUse.input = .object(dict)
                      snapshot.content[index].serverToolUse = serverToolUse
                    }
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
      emit(.connect)
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
        emit(.finalMessage(message: finalMessage))
      }
    }

    func handleError(_ error: Error) {
      errored = true
      if let error = error as? URLError, error.code == .cancelled {
        aborted = true
        emit(.abort(error: AIError.cancelled))
        return
      }
      if let aiError = error as? AIError {
        emit(.error(error: aiError))
        emit(.end)
        return
      }
      emit(.error(error: AIError.network(underlying: error)))
      emit(.end)
    }
  }
}
