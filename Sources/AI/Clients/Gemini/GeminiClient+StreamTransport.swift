// Copyright © Anthony DePasquale

import Foundation
import os.log
import SSE

struct GeminiStreamChunk {
  let text: String?
  let thought: Bool?
  let thoughtSignature: String?
  let groundingMetadata: GeminiClient.GroundingMetadata?
  let toolCall: ToolCall?
  let opaqueBlock: OpaqueBlock?
  let usageMetadata: GeminiClient.UsageMetadata?
  let finishReason: GeminiClient.FinishReason?
}

struct GeminiPartialToolCallState {
  var current: ToolCall?
}

enum GeminiJSONPathComponent {
  case key(String)
  case index(Int)
}

enum GeminiStreamTransport {
  static func processResponseChunk(
    _ jsonObject: [String: Any]?,
    continuation: AsyncThrowingStream<GeminiStreamChunk, Error>.Continuation,
    partialToolCallState: inout GeminiPartialToolCallState,
  ) throws -> Bool {
    if let promptFeedback = jsonObject?["promptFeedback"] as? [String: Any],
       let blockReason = promptFeedback["blockReason"] as? String
    {
      let blockMessage = (promptFeedback["blockReasonMessage"] as? String) ?? "Content was blocked."
      geminiTransportLogger.warning("Prompt blocked: \(blockReason) - \(blockMessage)")

      let errorResponse = GeminiClient.GenerateContentResponse(
        candidates: nil,
        promptFeedback: GeminiClient.PromptFeedback(),
        usageMetadata: nil,
      )

      continuation.finish(throwing: GeminiClient.GeminiError(
        message: "Your request was blocked: \(blockMessage)",
        response: errorResponse,
      ))
      return true
    }

    if let candidates = GeminiStreamTransport.jsonObjectArray(from: jsonObject?["candidates"]),
       let firstCandidate = candidates.first,
       let content = firstCandidate["content"] as? [String: Any],
       let parts = GeminiStreamTransport.jsonObjectArray(from: content["parts"])
    {
      for part in parts {
        if let text = part["text"] as? String {
          let isThinkingText =
            if let thought = part["thought"] as? Bool { thought }
            else if let thought = part["thought"] as? Int { thought == 1 }
            else { false }
          let signature = isThinkingText ? part["thoughtSignature"] as? String : nil
          continuation.yield(GeminiStreamChunk(
            text: text,
            thought: isThinkingText,
            thoughtSignature: signature,
            groundingMetadata: nil,
            toolCall: nil,
            opaqueBlock: nil,
            usageMetadata: nil,
            finishReason: nil,
          ))
        } else if let functionCall = part["functionCall"] as? [String: Any],
                  let sendableFunctionCall = GeminiStreamTransport.sendableJSONObject(from: functionCall),
                  let sendablePart = GeminiStreamTransport.sendableJSONObject(from: part)
        {
          if let toolCallResponse = try parseToolCall(
            from: sendableFunctionCall,
            part: sendablePart,
            partialToolCallState: &partialToolCallState,
          ) {
            continuation.yield(GeminiStreamChunk(
              text: nil,
              thought: nil,
              thoughtSignature: nil,
              groundingMetadata: nil,
              toolCall: toolCallResponse,
              opaqueBlock: nil,
              usageMetadata: nil,
              finishReason: nil,
            ))
          }
        } else if let serverToolCallDict = part["toolCall"] as? [String: Any] {
          if let block = geminiOpaqueBlock(type: "toolCall", jsonObject: serverToolCallDict) {
            continuation.yield(GeminiStreamChunk(
              text: nil,
              thought: nil,
              thoughtSignature: nil,
              groundingMetadata: nil,
              toolCall: nil,
              opaqueBlock: block,
              usageMetadata: nil,
              finishReason: nil,
            ))
          }
        } else if let serverToolResponseDict = part["toolResponse"] as? [String: Any] {
          if let block = geminiOpaqueBlock(type: "toolResponse", jsonObject: serverToolResponseDict) {
            continuation.yield(GeminiStreamChunk(
              text: nil,
              thought: nil,
              thoughtSignature: nil,
              groundingMetadata: nil,
              toolCall: nil,
              opaqueBlock: block,
              usageMetadata: nil,
              finishReason: nil,
            ))
          }
        } else if let executableCodeDict = part["executableCode"] as? [String: Any] {
          do {
            let jsonData = try JSONSerialization.data(withJSONObject: executableCodeDict)
            let executableCode = try JSONDecoder().decode(GeminiClient.ExecutableCode.self, from: jsonData)
            let languageTag = (executableCode.language ?? "").lowercased()
            let displayText = executableCode.code.map { "\n\n```\(!languageTag.isEmpty ? languageTag : "")\n\($0)\n```\n\n" }
            if let block = geminiOpaqueBlock(
              type: "executableCode",
              jsonObject: executableCodeDict,
              content: displayText,
              isResponseContent: displayText != nil,
            ) {
              continuation.yield(GeminiStreamChunk(
                text: nil,
                thought: nil,
                thoughtSignature: nil,
                groundingMetadata: nil,
                toolCall: nil,
                opaqueBlock: block,
                usageMetadata: nil,
                finishReason: nil,
              ))
            }
          } catch {
            geminiTransportLogger.error("Failed to decode ExecutableCode: \(error.localizedDescription)")
          }
        } else if let codeExecutionResultDict = part["codeExecutionResult"] as? [String: Any] {
          do {
            let jsonData = try JSONSerialization.data(withJSONObject: codeExecutionResultDict)
            let executionResult = try JSONDecoder().decode(GeminiClient.CodeExecutionResult.self, from: jsonData)
            let displayText = executionResult.output.map { "\n\n```\n\($0)\($0.last == "\n" ? "" : "\n")```\n\n" }
            if let block = geminiOpaqueBlock(
              type: "codeExecutionResult",
              jsonObject: codeExecutionResultDict,
              content: displayText,
              isResponseContent: displayText != nil,
            ) {
              continuation.yield(GeminiStreamChunk(
                text: nil,
                thought: nil,
                thoughtSignature: nil,
                groundingMetadata: nil,
                toolCall: nil,
                opaqueBlock: block,
                usageMetadata: nil,
                finishReason: nil,
              ))
            }
          } catch {
            geminiTransportLogger.error("Failed to decode CodeExecutionResult: \(error.localizedDescription)")
          }
        } else if let thoughtSignature = part["thoughtSignature"] as? String {
          continuation.yield(GeminiStreamChunk(
            text: nil,
            thought: nil,
            thoughtSignature: thoughtSignature,
            groundingMetadata: nil,
            toolCall: nil,
            opaqueBlock: nil,
            usageMetadata: nil,
            finishReason: nil,
          ))
        }
      }
    }

    if let candidates = GeminiStreamTransport.jsonObjectArray(from: jsonObject?["candidates"]),
       let firstCandidate = candidates.first,
       let groundingMetadataDict = firstCandidate["groundingMetadata"] as? [String: Any]
    {
      let metadataData = try JSONSerialization.data(withJSONObject: groundingMetadataDict)
      let groundingMetadata = try JSONDecoder().decode(GeminiClient.GroundingMetadata.self, from: metadataData)
      continuation.yield(GeminiStreamChunk(
        text: nil,
        thought: nil,
        thoughtSignature: nil,
        groundingMetadata: groundingMetadata,
        toolCall: nil,
        opaqueBlock: nil,
        usageMetadata: nil,
        finishReason: nil,
      ))
    }

    if let candidates = GeminiStreamTransport.jsonObjectArray(from: jsonObject?["candidates"]),
       let firstCandidate = candidates.first,
       let urlContextMetadataDict = firstCandidate["urlContextMetadata"] as? [String: Any],
       let block = geminiOpaqueBlock(type: "urlContextMetadata", jsonObject: urlContextMetadataDict)
    {
      continuation.yield(GeminiStreamChunk(
        text: nil,
        thought: nil,
        thoughtSignature: nil,
        groundingMetadata: nil,
        toolCall: nil,
        opaqueBlock: block,
        usageMetadata: nil,
        finishReason: nil,
      ))
    }

    if let usageMetadataDict = jsonObject?["usageMetadata"] as? [String: Any] {
      do {
        let metadataData = try JSONSerialization.data(withJSONObject: usageMetadataDict)
        let decodedUsageMetadata = try JSONDecoder().decode(GeminiClient.UsageMetadata.self, from: metadataData)
        continuation.yield(GeminiStreamChunk(
          text: nil,
          thought: nil,
          thoughtSignature: nil,
          groundingMetadata: nil,
          toolCall: nil,
          opaqueBlock: nil,
          usageMetadata: decodedUsageMetadata,
          finishReason: nil,
        ))
      } catch {
        geminiTransportLogger.error("Failed to decode usageMetadata: \(error.localizedDescription)")
      }
    }

    if let candidates = GeminiStreamTransport.jsonObjectArray(from: jsonObject?["candidates"]),
       let firstCandidate = candidates.first,
       let finishReasonString = firstCandidate["finishReason"] as? String
    {
      let finishReason = GeminiClient.FinishReason(rawValue: finishReasonString) ?? .other
      continuation.yield(GeminiStreamChunk(
        text: nil,
        thought: nil,
        thoughtSignature: nil,
        groundingMetadata: nil,
        toolCall: nil,
        opaqueBlock: nil,
        usageMetadata: nil,
        finishReason: finishReason,
      ))
    }

    return false
  }

  static func processStreamBytes(
    result: URLSession.AsyncBytes,
    response: URLResponse,
  ) -> AsyncThrowingStream<GeminiStreamChunk, Error> {
    let (stream, continuation) = AsyncThrowingStream<GeminiStreamChunk, Error>.makeStream()
    let task = Task {
      do {
        var partialToolCallState = GeminiPartialToolCallState()
        guard let httpResponse = response as? HTTPURLResponse else {
          throw AIError.network(underlying: URLError(.badServerResponse))
        }

        if !(200 ... 299).contains(httpResponse.statusCode) {
          var errorMessage: String?
          do {
            var errorData = Data()
            for try await byte in result {
              try Task.checkCancellation()
              errorData.append(byte)
            }
            errorMessage = parseGeminiErrorMessage(from: errorData)
          } catch {
            geminiTransportLogger.error("Failed to read error response: \(error.localizedDescription)")
          }

          throw geminiHTTPError(statusCode: httpResponse.statusCode, message: errorMessage)
        }

        for try await event in result.events {
          try Task.checkCancellation()
          guard let data = event.data.data(using: .utf8) else {
            throw AIError.parsing(message: "Failed to convert SSE payload to UTF-8 data")
          }
          let jsonObject = try decodedJSONObject(from: data)
          if try processResponseChunk(
            jsonObject,
            continuation: continuation,
            partialToolCallState: &partialToolCallState,
          ) {
            return
          }
        }
        continuation.finish()
      } catch {
        geminiTransportLogger.error("Stream processing error: \(error.localizedDescription)")
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  static func processBufferedResponse(
    data: Data,
    response: URLResponse,
  ) -> AsyncThrowingStream<GeminiStreamChunk, Error> {
    let (stream, continuation) = AsyncThrowingStream<GeminiStreamChunk, Error>.makeStream()
    let task = Task {
      do {
        var partialToolCallState = GeminiPartialToolCallState()
        guard let httpResponse = response as? HTTPURLResponse else {
          throw AIError.network(underlying: URLError(.badServerResponse))
        }

        if !(200 ... 299).contains(httpResponse.statusCode) {
          let errorMessage = parseGeminiErrorMessage(from: data)
          throw geminiHTTPError(statusCode: httpResponse.statusCode, message: errorMessage)
        }

        let jsonObject = try decodedJSONObject(from: data)
        _ = try processResponseChunk(
          jsonObject,
          continuation: continuation,
          partialToolCallState: &partialToolCallState,
        )
        continuation.finish()
      } catch {
        geminiTransportLogger.error("Buffered response processing error: \(error.localizedDescription)")
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  static func geminiOpaqueBlock(
    type: String,
    jsonObject: [String: Any],
    content: String? = nil,
    isResponseContent: Bool = false,
  ) -> OpaqueBlock? {
    guard let jsonData = try? JSONSerialization.data(withJSONObject: jsonObject),
          let rawJSON = String(data: jsonData, encoding: .utf8)
    else {
      return nil
    }

    return OpaqueBlock(
      provider: "gemini",
      type: type,
      content: content,
      data: rawJSON,
      isResponseContent: isResponseContent,
    )
  }

  static func geminiJSONObject(from opaque: OpaqueBlock) -> [String: any Sendable]? {
    guard let jsonString = opaque.data,
          let jsonData = jsonString.data(using: .utf8),
          let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
          let value = try? Value.fromAny(jsonObject),
          case let .object(dictionary) = value
    else {
      return nil
    }

    return Value.toSendable(dictionary)
  }

  static func decodedJSONObject(from data: Data) throws -> [String: Any]? {
    try JSONSerialization.jsonObject(with: data) as? [String: Any]
  }

  static func sendableJSONObject(from jsonObject: [String: Any]) -> [String: any Sendable]? {
    guard let value = try? Value.fromAny(jsonObject),
          case let .object(dictionary) = value
    else {
      return nil
    }

    return Value.toSendable(dictionary)
  }

  static func jsonObjectArray(from value: Any?) -> [[String: Any]]? {
    guard let array = value as? [Any] else {
      return nil
    }

    var objects = [[String: Any]]()
    objects.reserveCapacity(array.count)
    for element in array {
      guard let object = element as? [String: Any] else {
        return nil
      }
      objects.append(object)
    }
    return objects
  }

  static func parseGeminiErrorMessage(from data: Data) -> String? {
    struct GeminiErrorResponse: Codable {
      struct ErrorDetail: Codable {
        let message: String
        let status: String
        let code: Int
      }

      let error: ErrorDetail
    }

    if let errorResponse = try? JSONDecoder().decode(GeminiErrorResponse.self, from: data) {
      return errorResponse.error.message
    } else if let errorArray = try? JSONDecoder().decode([GeminiErrorResponse].self, from: data),
              let firstError = errorArray.first
    {
      return firstError.error.message
    }
    return nil
  }

  private static func geminiHTTPError(statusCode: Int, message: String?) -> AIError {
    switch statusCode {
      case 400: .invalidRequest(message: message ?? "There was a problem with the request body.")
      case 403: .authentication(message: "Ensure your API key is set correctly and has the right access.")
      case 404: .invalidRequest(message: message.map { "Not found: \($0)" } ?? "The requested resource wasn't found.")
      case 429: .rateLimit(retryAfter: nil)
      case 500: .serverError(statusCode: 500, message: message ?? "An unexpected error occurred. Try reducing your input context, switching to another model temporarily, or retry after a short wait.", context: nil)
      case 503: .serverError(statusCode: 503, message: message ?? "The service may be temporarily overloaded. Try switching to another model temporarily or retry after a short wait.", context: nil)
      case 504: .timeout
      default: .serverError(statusCode: statusCode, message: message ?? "HTTP error \(statusCode)", context: nil)
    }
  }

  private static func parseToolCall(
    from functionCall: [String: any Sendable],
    part: [String: any Sendable],
    partialToolCallState: inout GeminiPartialToolCallState,
  ) throws -> ToolCall? {
    let thoughtSignature = part["thoughtSignature"] as? String
    let existingToolCall = partialToolCallState.current

    if let args = functionCall["args"] as? [String: any Sendable] {
      let explicitName = functionCall["name"] as? String
      let name = explicitName ?? existingToolCall?.name
      guard let name else {
        geminiTransportLogger.warning("Dropping Gemini functionCall chunk with args but no function name")
        return nil
      }
      let continuationToolCall: ToolCall? = if explicitName == nil || explicitName == existingToolCall?.name {
        existingToolCall
      } else {
        nil
      }
      let toolCallId = (functionCall["id"] as? String)
        ?? continuationToolCall?.id
        ?? generateShortId()
      let parameters = try args.mapValues { try Value.fromAny($0) }
      let toolCall = ToolCall(
        name: name,
        id: toolCallId,
        parameters: parameters,
        providerMetadata: mergedProviderMetadata(
          existing: continuationToolCall?.providerMetadata,
          thoughtSignature: thoughtSignature,
        ),
      )
      if let willContinue = functionCall["willContinue"] as? Bool, willContinue {
        partialToolCallState.current = toolCall
      } else {
        partialToolCallState.current = nil
      }
      return toolCall
    }

    if let partialArgs = functionCall["partialArgs"] as? [[String: any Sendable]] {
      let name = (functionCall["name"] as? String) ?? existingToolCall?.name
      guard let name else {
        geminiTransportLogger.warning("Dropping Gemini partial functionCall chunk with no active function name")
        return nil
      }
      let toolCallId = (functionCall["id"] as? String) ?? existingToolCall?.id ?? generateShortId()
      var toolCall = existingToolCall ?? ToolCall(
        name: name,
        id: toolCallId,
        parameters: [:],
        providerMetadata: nil,
      )
      toolCall.name = name
      toolCall.providerMetadata = mergedProviderMetadata(
        existing: existingToolCall?.providerMetadata,
        thoughtSignature: thoughtSignature,
      )
      try applyPartialArgs(partialArgs, to: &toolCall.parameters)
      if let willContinue = functionCall["willContinue"] as? Bool, willContinue {
        partialToolCallState.current = toolCall
      } else {
        partialToolCallState.current = nil
      }
      return toolCall
    }

    if let name = functionCall["name"] as? String {
      let toolCallId = (functionCall["id"] as? String) ?? generateShortId()
      let toolCall = ToolCall(
        name: name,
        id: toolCallId,
        parameters: [:],
        providerMetadata: mergedProviderMetadata(existing: nil, thoughtSignature: thoughtSignature),
      )
      partialToolCallState.current = toolCall
      return toolCall
    }

    if thoughtSignature != nil, var toolCall = partialToolCallState.current {
      toolCall.providerMetadata = mergedProviderMetadata(
        existing: toolCall.providerMetadata,
        thoughtSignature: thoughtSignature,
      )
      partialToolCallState.current = toolCall
      return toolCall
    }

    return nil
  }

  private static func mergedProviderMetadata(
    existing: [String: String]?,
    thoughtSignature: String?,
  ) -> [String: String]? {
    guard existing != nil || thoughtSignature != nil else { return nil }
    var metadata = existing ?? [:]
    if let thoughtSignature {
      metadata["thoughtSignature"] = thoughtSignature
    }
    return metadata
  }

  private static func applyPartialArgs(
    _ partialArgs: [[String: any Sendable]],
    to parameters: inout [String: Value],
  ) throws {
    for partialArg in partialArgs {
      guard let jsonPath = partialArg["jsonPath"] as? String else {
        geminiTransportLogger.warning("Dropping Gemini partial functionCall arg without jsonPath")
        continue
      }
      guard let path = parseJSONPath(jsonPath) else {
        geminiTransportLogger.warning("Dropping Gemini partial functionCall arg with unsupported jsonPath '\(jsonPath)'")
        continue
      }
      guard let newValue = try partialArgValue(
        from: partialArg,
        existing: value(at: path, in: .object(parameters)),
      ) else {
        continue
      }
      var root = Value.object(parameters)
      setValue(newValue, at: ArraySlice(path), in: &root)
      if case let .object(updatedParameters) = root {
        parameters = updatedParameters
      }
    }
  }

  private static func partialArgValue(
    from partialArg: [String: any Sendable],
    existing: Value?,
  ) throws -> Value? {
    if let stringValue = partialArg["stringValue"] as? String {
      let existingText = existing?.stringValue ?? ""
      return .string(existingText + stringValue)
    }
    if let boolValue = partialArg["boolValue"] as? Bool {
      return .bool(boolValue)
    }
    if let numberValue = partialArg["numberValue"] {
      return try Value.fromAny(numberValue)
    }
    if let nullValue = partialArg["nullValue"] as? String, nullValue == "NULL_VALUE" {
      return .null
    }
    return nil
  }

  private static func parseJSONPath(_ path: String) -> [GeminiJSONPathComponent]? {
    guard path.first == "$" else { return nil }
    var components: [GeminiJSONPathComponent] = []
    var index = path.index(after: path.startIndex)

    while index < path.endIndex {
      switch path[index] {
        case ".":
          index = path.index(after: index)
          let start = index
          while index < path.endIndex {
            let character = path[index]
            if character == "." || character == "[" {
              break
            }
            index = path.index(after: index)
          }
          guard start < index else { return nil }
          components.append(.key(String(path[start ..< index])))
        case "[":
          index = path.index(after: index)
          guard index < path.endIndex else { return nil }
          if path[index] == "\"" || path[index] == "'" {
            let quote = path[index]
            index = path.index(after: index)
            var key = ""
            while index < path.endIndex {
              let character = path[index]
              if character == quote {
                break
              }
              if character == "\\" {
                index = path.index(after: index)
                guard index < path.endIndex else { return nil }
              }
              key.append(path[index])
              index = path.index(after: index)
            }
            guard index < path.endIndex, path[index] == quote else { return nil }
            index = path.index(after: index)
            guard index < path.endIndex, path[index] == "]" else { return nil }
            index = path.index(after: index)
            components.append(.key(key))
          } else {
            let start = index
            while index < path.endIndex, path[index].wholeNumberValue != nil {
              index = path.index(after: index)
            }
            guard start < index,
                  let arrayIndex = Int(path[start ..< index]),
                  index < path.endIndex,
                  path[index] == "]"
            else {
              return nil
            }
            index = path.index(after: index)
            components.append(.index(arrayIndex))
          }
        default:
          return nil
      }
    }

    return components
  }

  private static func value(
    at path: [GeminiJSONPathComponent],
    in value: Value,
  ) -> Value? {
    var current = value
    for component in path {
      switch component {
        case let .key(key):
          guard case let .object(object) = current, let next = object[key] else { return nil }
          current = next
        case let .index(index):
          guard case let .array(array) = current, array.indices.contains(index) else { return nil }
          current = array[index]
      }
    }
    return current
  }

  private static func setValue(
    _ newValue: Value,
    at path: ArraySlice<GeminiJSONPathComponent>,
    in value: inout Value,
  ) {
    guard let component = path.first else {
      value = newValue
      return
    }

    switch component {
      case let .key(key):
        var object = value.objectValue ?? [:]
        var child = object[key] ?? defaultContainer(for: path.dropFirst().first)
        setValue(newValue, at: path.dropFirst(), in: &child)
        object[key] = child
        value = .object(object)
      case let .index(index):
        var array = value.arrayValue ?? []
        if array.count <= index {
          array.append(contentsOf: Array(repeating: .null, count: index - array.count + 1))
        }
        var child = array[index]
        if child == .null {
          child = defaultContainer(for: path.dropFirst().first)
        }
        setValue(newValue, at: path.dropFirst(), in: &child)
        array[index] = child
        value = .array(array)
    }
  }

  private static func defaultContainer(for next: GeminiJSONPathComponent?) -> Value {
    switch next {
      case .key:
        .object([:])
      case .index:
        .array([])
      case nil:
        .null
    }
  }
}

extension GeminiClient {
  static func geminiJSONObject(from opaque: OpaqueBlock) -> [String: any Sendable]? {
    GeminiStreamTransport.geminiJSONObject(from: opaque)
  }
}

private let geminiTransportLogger = Logger(
  subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence",
  category: "GeminiClient",
)
