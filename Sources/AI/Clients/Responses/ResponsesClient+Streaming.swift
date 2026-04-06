// Copyright © Anthony DePasquale

import Foundation
import os
import SSE

extension ResponsesClient {
  func streamResponse(
    input: [Message],
    systemPrompt: String?,
    modelId: String,
    apiKey: String?,
    maxTokens: Int?,
    temperature: Float?,
    stream: Bool,
    reasoningEffortLevel: ReasoningEffortLevel?,
    verbosityLevel: VerbosityLevel?,
    serverSideTools: [ServerSideTool],
    backgroundMode: Bool,
    provider: ResponsesProvider?,
    textFormat: ResponseFormat? = nil,
    tools: [Tool] = [],
    enableStrictModeForTools: Bool = true,
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 600.0
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let resolvedBackend = try resolveResponsesBackend(for: endpoint, provider: provider)
    if resolvedBackend == nil, replayContainsReasoningHistory(input) {
      emitResponsesReplayWarning(
        "Custom Responses endpoint is missing responsesProvider; replay may omit provider-specific reasoning capture. Pass `.openAI` or `.xAI` when the backend family is known.",
      )
    }

    let replayPlan = try await ResponsesReplayNormalizer.normalize(input)
    let captureRequirements = ReplayCapturePolicy.requirements(for: .responses(
      modelId: modelId,
      backend: resolvedBackend,
    ))

    var body: [String: any Sendable] = [
      "model": modelId,
      "input": replayPlan.inputItems,
      "stream": stream,
    ]

    if backgroundMode {
      body["background"] = true
      body["store"] = true
    }

    var toolsArray: [[String: any Sendable]] = []
    for serverSideTool in serverSideTools {
      toolsArray.append(serverSideTool.definition)
    }

    if !tools.isEmpty {
      for tool in tools {
        if let baseSchemaBuildErrorMessage = tool.baseSchemaBuildErrorMessage {
          throw AIError.invalidRequest(
            message: "Tool '\(tool.name)' has an invalid input schema: \(baseSchemaBuildErrorMessage)",
          )
        }
        if enableStrictModeForTools, let schemaBuildErrorMessage = tool.schemaBuildErrorMessage {
          throw AIError.invalidRequest(
            message: "Tool '\(tool.name)' has an invalid strict schema: \(schemaBuildErrorMessage)",
          )
        }
        let parameters: [String: any Sendable] = if enableStrictModeForTools {
          try Value.schemaForStrictMode(tool.rawInputSchema)
        } else {
          Value.toSendable(tool.rawInputSchema)
        }
        toolsArray.append([
          "type": "function",
          "name": tool.name,
          "description": tool.description,
          "parameters": parameters,
          "strict": enableStrictModeForTools,
        ])
      }
    }

    if !toolsArray.isEmpty {
      body["tools"] = toolsArray
      body["tool_choice"] = "auto"
    }

    if let systemPrompt, !systemPrompt.isEmpty {
      body["instructions"] = systemPrompt
    }

    if let maxTokens {
      body["max_output_tokens"] = maxTokens
    }

    if let temperature {
      body["temperature"] = temperature
    }

    if let reasoningEffortLevel {
      body["reasoning"] = [
        "effort": reasoningEffortLevel.rawValue,
        "summary": "auto",
      ]
    }

    var includeFields: [String] = []
    if captureRequirements.requiresOpenAIResponsesReasoningEncryptedContent {
      includeFields.append("reasoning.encrypted_content")
    }
    if !includeFields.isEmpty {
      body["include"] = includeFields
    }

    var textConfig: [String: any Sendable] = [:]
    if let textFormat {
      var formatConfig: [String: any Sendable] = [:]
      switch textFormat {
        case .text:
          formatConfig["type"] = "text"
        case .jsonObject:
          formatConfig["type"] = "json_object"
        case let .jsonSchema(schema, name, description):
          formatConfig["type"] = "json_schema"
          formatConfig["schema"] = schema
          if let name { formatConfig["name"] = name }
          if let description { formatConfig["description"] = description }
      }
      textConfig["format"] = formatConfig
    }
    if let verbosityLevel {
      textConfig["verbosity"] = verbosityLevel.rawValue
    }
    if !textConfig.isEmpty {
      body["text"] = textConfig
    }

    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let finalRequest = request

    let (resultStream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
    let streamTask = Task { @Sendable in
      let request = finalRequest
      do {
        if backgroundMode, stream {
          openAIResponsesLogger.log("Initiating background mode response with streaming in OpenAI Responses client")
          try await streamBackgroundResponseDirect(
            request: request,
            apiKey: apiKey,
            continuation: continuation,
          )
        } else if backgroundMode {
          openAIResponsesLogger.log("Initiating background mode response without streaming in OpenAI Responses client")
          let (data, response) = try await session.data(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            try handleErrorResponse(httpResponse, data: data)
          }

          let decodedResponse = try JSONDecoder().decode(ResponseObject.self, from: data)
          guard let responseId = decodedResponse.id else {
            throw AIError.parsing(message: "Failed to parse background response ID")
          }
          await MainActor.run {
            activeBackgroundResponseId = responseId
            activeBackgroundResponseApiKey = apiKey
          }
          try await pollBackgroundResponse(responseId: responseId, apiKey: apiKey, continuation: continuation)
        } else if stream {
          openAIResponsesLogger.log("Initiating standard streamed response in OpenAI Responses client")
          try await performSSEStream(
            request: request,
            continuation: continuation,
            logPrefix: "Standard Stream",
          )
        } else {
          let (data, response) = try await session.data(for: request)
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }
          if !(200 ... 299).contains(httpResponse.statusCode) {
            try handleErrorResponse(httpResponse, data: data)
          }

          do {
            let response = try JSONDecoder().decode(ResponseObject.self, from: data)
            continuation.yield(response.toGenerationResponse())
          } catch {
            openAIResponsesLogger.error("Non-streaming response parsing error: \(error)")
            throw AIError.parsing(message: "Failed to parse non-streamed response: \(error.localizedDescription)")
          }
        }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }

    continuation.onTermination = { @Sendable termination in
      if case .cancelled = termination {
        openAIResponsesLogger.log("AsyncThrowingStream cancelled by consumer - cancelling stream task")
        streamTask.cancel()
      }
    }
    return resultStream
  }

  func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    try AIError.throwOpenAIHTTPError(httpResponse, data: data, logger: openAIResponsesLogger)
  }

  func processStreamingEvent(
    event: StreamEvent,
    streamingState: inout StreamingResponseState,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) throws {
    if let errorMessage = event.error?.message {
      throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
    }

    guard let eventType = event.type else { return }

    func yieldCurrentState() {
      continuation.yield(GenerationResponse(content: streamingState.content))
    }

    switch eventType {
      case StreamEventType.outputTextDelta:
        if let delta = event.delta {
          streamingState.appendTextDelta(delta, outputIndex: event.outputIndex, contentIndex: event.contentIndex)
          yieldCurrentState()
        }

      case StreamEventType.outputTextDone:
        if let text = event.text {
          streamingState.setFinalizedText(text, outputIndex: event.outputIndex, contentIndex: event.contentIndex)
          yieldCurrentState()
        }

      case StreamEventType.outputTextAnnotationAdded:
        if let annotation = event.annotation?.raw {
          streamingState.addTextAnnotation(
            annotation,
            outputIndex: event.outputIndex,
            contentIndex: event.contentIndex,
            annotationIndex: event.annotationIndex,
          )
          yieldCurrentState()
        }

      case StreamEventType.refusalDelta:
        if let delta = event.delta {
          streamingState.appendRefusalDelta(delta, outputIndex: event.outputIndex, contentIndex: event.contentIndex)
          yieldCurrentState()
        }

      case StreamEventType.refusalDone:
        if let refusal = event.refusal {
          streamingState.setFinalizedRefusal(refusal, outputIndex: event.outputIndex, contentIndex: event.contentIndex)
          yieldCurrentState()
        }

      case StreamEventType.reasoningTextDelta, StreamEventType.reasoningDelta:
        if let delta = event.delta {
          streamingState.appendReasoningDelta(delta, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.reasoningTextDone:
        if let text = event.text {
          streamingState.setFinalizedReasoningText(text, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.reasoningSummaryDelta:
        if let delta = event.delta {
          streamingState.appendSummaryDelta(delta, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.reasoningSummaryDone:
        if let text = event.text {
          streamingState.setSummaryText(text, outputIndex: event.outputIndex)
          yieldCurrentState()
        }

      case StreamEventType.contentPartAdded:
        break

      case StreamEventType.outputItemAdded:
        if let item = event.item, let itemType = item.type {
          switch itemType {
            case OutputItemType.functionCall:
              if let name = item.name, let callId = item.callId {
                streamingState.setToolCall(ToolCall(
                  name: name,
                  id: callId,
                  parameters: [:],
                ), outputIndex: event.outputIndex, itemId: item.id)
                yieldCurrentState()
              }
            case OutputItemType.reasoning:
              break
            case OutputItemType.codeInterpreterCall:
              openAIResponsesLogger.log("Received code_interpreter_call item")
            case OutputItemType.webSearchCall:
              openAIResponsesLogger.log("Received web_search_call item")
            case OutputItemType.message:
              openAIResponsesLogger.log("Received message item")
            default:
              openAIResponsesLogger.log("Ignoring added output item type: \(itemType)")
          }
        }

      case StreamEventType.functionCallArgumentsDelta:
        if let delta = event.delta {
          streamingState.appendToolCallArgumentsDelta(delta, outputIndex: event.outputIndex, itemId: event.itemId)
          yieldCurrentState()
        }

      case StreamEventType.functionCallArgumentsDone:
        if let argumentsString = event.arguments {
          streamingState.completeToolCallArguments(
            argumentsString,
            outputIndex: event.outputIndex,
            itemId: event.itemId,
            name: event.name,
          )
          yieldCurrentState()
        }

      default:
        break
    }
  }

  func yieldFinalResponse(
    _ generationResponse: GenerationResponse,
    mergingToolCallsFrom streamingState: StreamingResponseState,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) {
    var finalContent = generationResponse.content

    var seenToolCallIDs = Set(finalContent.compactMap { block -> String? in
      guard case let .toolCall(toolCall) = block else { return nil }
      return toolCall.id
    })
    for toolCall in streamingState.content.compactMap({ block -> ToolCall? in
      guard case let .toolCall(toolCall) = block else { return nil }
      return toolCall
    }) where !seenToolCallIDs.contains(toolCall.id) {
      finalContent.append(.toolCall(toolCall))
      seenToolCallIDs.insert(toolCall.id)
    }

    continuation.yield(GenerationResponse(
      content: finalContent,
      metadata: generationResponse.metadata,
    ))
  }

  func streamBackgroundResponseDirect(
    request: URLRequest,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    retryCount: Int = 0,
    maxRetries: Int = 3,
  ) async throws {
    openAIResponsesLogger.log("Background Stream Direct: Starting attempt \(retryCount + 1)/\(maxRetries + 1)")

    try await performBackgroundStream(
      request: request,
      responseId: nil,
      apiKey: apiKey,
      continuation: continuation,
      startingAfter: 0,
      retryCount: retryCount,
      maxRetries: maxRetries,
      isDirect: true,
    )
  }

  func streamBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int? = nil,
    retryCount: Int = 0,
    maxRetries: Int = 3,
  ) async throws {
    openAIResponsesLogger.log("Background Stream: Starting for response \(responseId), attempt \(retryCount + 1)/\(maxRetries + 1), startingAfter: \(startingAfter ?? 0)")

    let streamUrl = endpoint.appendingPathComponent(responseId)
    guard var urlComponents = URLComponents(url: streamUrl, resolvingAgainstBaseURL: false) else {
      throw AIError.invalidRequest(message: "Failed to construct URL components for response: \(responseId)")
    }
    urlComponents.queryItems = [
      URLQueryItem(name: "stream", value: "true"),
    ]
    if let startingAfter {
      urlComponents.queryItems?.append(URLQueryItem(name: "starting_after", value: String(startingAfter)))
    }
    guard let requestURL = urlComponents.url else {
      throw AIError.invalidRequest(message: "Failed to construct request URL for response: \(responseId)")
    }

    var request = URLRequest(url: requestURL)
    request.httpMethod = "GET"
    request.timeoutInterval = 600.0
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    try await performBackgroundStream(
      request: request,
      responseId: responseId,
      apiKey: apiKey,
      continuation: continuation,
      startingAfter: startingAfter ?? 0,
      retryCount: retryCount,
      maxRetries: maxRetries,
      isDirect: false,
    )
  }

  func performBackgroundStream(
    request: URLRequest,
    responseId: String?,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    startingAfter: Int,
    retryCount: Int,
    maxRetries: Int,
    isDirect: Bool,
  ) async throws {
    let logPrefix = isDirect ? "Background Stream Direct" : "Background Stream"
    var lastSequenceNumber: Int = startingAfter
    var currentResponseId: String? = responseId

    do {
      try await performSSEStream(
        request: request,
        continuation: continuation,
        logPrefix: logPrefix,
        responseIdHandler: isDirect ? { id in
          guard !Task.isCancelled else { return }
          currentResponseId = id
          openAIResponsesLogger.log("\(logPrefix): Got response ID: \(id)")
          await MainActor.run { [weak self] in
            self?.activeBackgroundResponseId = id
            self?.activeBackgroundResponseApiKey = apiKey
          }
        } : nil,
        sequenceHandler: { sequenceNumber in
          guard !Task.isCancelled else { return }
          if sequenceNumber > lastSequenceNumber {
            if sequenceNumber % 10 == 0 || sequenceNumber - lastSequenceNumber > 10 {
              openAIResponsesLogger.log("\(logPrefix): Progress update - sequence: \(sequenceNumber)")
            }
          }
          lastSequenceNumber = sequenceNumber
        },
      )
    } catch {
      let isCancellationError = error is CancellationError || (error as NSError).code == NSURLErrorCancelled

      if isCancellationError {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled by user")
        if let responseId = currentResponseId {
          openAIResponsesLogger.log("\(logPrefix): Cancelling background response \(responseId) on server")
          try? await cancelBackgroundResponse(responseId: responseId, apiKey: apiKey)
        }
        return
      }

      openAIResponsesLogger.log("\(logPrefix): Error occurred - \(error)")
      if retryCount < maxRetries {
        let isTimeoutError = (error as NSError).code == NSURLErrorTimedOut ||
          (error as NSError).code == NSURLErrorNetworkConnectionLost ||
          (error as NSError).code == NSURLErrorCannotConnectToHost

        if isTimeoutError {
          openAIResponsesLogger.warning("\(logPrefix) timeout (attempt \(retryCount + 1)/\(maxRetries + 1))")

          if isDirect, let responseId = currentResponseId {
            openAIResponsesLogger.log("\(logPrefix): Switching to resumption mode with response ID: \(responseId)")
            try await streamBackgroundResponse(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: lastSequenceNumber,
              retryCount: retryCount,
              maxRetries: maxRetries,
            )
            return
          }

          if let responseId = currentResponseId ?? responseId {
            openAIResponsesLogger.log("\(logPrefix): Checking response status before retry...")
            if try await checkResponseStatusAndHandle(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              logPrefix: logPrefix,
            ) {
              return
            }
          }

          let backoffDelay = TimeInterval(pow(2.0, Double(retryCount)))
          openAIResponsesLogger.log("\(logPrefix): Waiting \(backoffDelay)s before retry...")
          try await Task.sleep(nanoseconds: UInt64(backoffDelay * 1_000_000_000))

          if isDirect {
            openAIResponsesLogger.warning("\(logPrefix): Cannot safely retry initial create request without a response ID")
            throw error
          } else {
            openAIResponsesLogger.log("\(logPrefix): Retrying from sequence \(lastSequenceNumber)...")
            try await streamBackgroundResponse(
              responseId: responseId!,
              apiKey: apiKey,
              continuation: continuation,
              startingAfter: lastSequenceNumber,
              retryCount: retryCount + 1,
              maxRetries: maxRetries,
            )
          }
          return
        } else {
          openAIResponsesLogger.log("\(logPrefix): Non-retryable error (code: \((error as NSError).code)): \(error)")
        }
      } else {
        openAIResponsesLogger.log("\(logPrefix): Max retries exceeded (\(maxRetries))")
      }

      if (error as NSError).code != NSURLErrorCancelled {
        openAIResponsesLogger.error("\(logPrefix) failed after \(retryCount) retries: \(error)")
      } else {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled after \(retryCount) retries")
      }
      throw error
    }
  }

  func performSSEStream(
    request: URLRequest,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String,
    responseIdHandler: ((String) async -> Void)? = nil,
    sequenceHandler: ((Int) -> Void)? = nil,
  ) async throws {
    openAIResponsesLogger.log("\(logPrefix): Connecting to stream...")

    let (result, response) = try await session.bytes(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }

    if !(200 ... 299).contains(httpResponse.statusCode) {
      var errorData = Data()
      for try await byte in result {
        try Task.checkCancellation()
        errorData.append(byte)
      }
      try handleErrorResponse(httpResponse, data: errorData)
    }

    var streamingState = StreamingResponseState()
    var receivedCompletedEvent = false
    var accumulatedSnapshot: AccumulatedResponseSnapshot?

    for try await event in result.events {
      try Task.checkCancellation()
      let jsonString = event.data

      if jsonString == "[DONE]" {
        break
      }

      guard let jsonData = jsonString.data(using: .utf8) else {
        throw AIError.parsing(message: "Failed to convert streamed response to data: \(jsonString)")
      }

      do {
        let event = try JSONDecoder().decode(StreamEvent.self, from: jsonData)

        let isTerminalEvent = event.type == StreamEventType.completed
          || event.type == StreamEventType.failed
          || event.type == StreamEventType.incomplete

        if event.type == StreamEventType.created || isTerminalEvent {
          guard let response = event.response else {
            throw AIError.parsing(message: "Responses stream \(event.type ?? "unknown") event missing response payload")
          }
          if accumulatedSnapshot == nil {
            accumulatedSnapshot = AccumulatedResponseSnapshot(response)
          }
          if event.type == StreamEventType.created, let id = response.id, let responseIdHandler {
            await responseIdHandler(id)
          }
        }

        if let sequenceHandler, let sequenceNumber = event.sequenceNumber {
          sequenceHandler(sequenceNumber)
        }

        if var snapshot = accumulatedSnapshot {
          snapshot.apply(event)
          accumulatedSnapshot = snapshot
        }

        if isTerminalEvent {
          receivedCompletedEvent = true
          guard let snapshot = accumulatedSnapshot else {
            throw AIError.parsing(message: "Responses stream ended (\(event.type ?? "unknown")) without an accumulated response snapshot")
          }
          yieldFinalResponse(
            snapshot.finalize(),
            mergingToolCallsFrom: streamingState,
            continuation: continuation,
          )
          continue
        }

        try processStreamingEvent(
          event: event,
          streamingState: &streamingState,
          continuation: continuation,
        )
      } catch let error as AIError {
        throw error
      } catch {
        openAIResponsesLogger.error("\(logPrefix) parsing error for JSON: \(jsonString). Error: \(error)")
        throw AIError.parsing(message: "Failed to parse streamed JSON: \(jsonString)")
      }
    }

    if !receivedCompletedEvent {
      guard let snapshot = accumulatedSnapshot else {
        throw AIError.parsing(message: "Responses stream ended without producing a response snapshot")
      }

      yieldFinalResponse(
        snapshot.finalize(),
        mergingToolCallsFrom: streamingState,
        continuation: continuation,
      )
    }
  }

  func checkResponseStatusAndHandle(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
    logPrefix: String,
  ) async throws -> Bool {
    do {
      let statusUrl = endpoint.appendingPathComponent(responseId)
      var statusRequest = URLRequest(url: statusUrl)
      statusRequest.httpMethod = "GET"
      if let apiKey {
        statusRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
      }
      let (statusData, statusResponse) = try await session.data(for: statusRequest)
      if let httpResponse = statusResponse as? HTTPURLResponse,
         (200 ... 299).contains(httpResponse.statusCode),
         let response = try? JSONDecoder().decode(ResponseObject.self, from: statusData),
         let statusString = response.status
      {
        let status = BackgroundResponseStatus(rawValue: statusString)

        switch status {
          case .completed, .incomplete:
            openAIResponsesLogger.log("\(logPrefix): Response \(statusString) during disconnection. Parsing final response.")
            parseCompletedResponse(response, continuation: continuation)
            return true
          case .failed:
            let errorMessage = response.error?.message ?? "Background response failed"
            openAIResponsesLogger.log("\(logPrefix): Response failed - \(errorMessage)")
            parseCompletedResponse(response, continuation: continuation)
            return true
          case .cancelled:
            openAIResponsesLogger.log("\(logPrefix): Response was cancelled")
            return true
          case .queued, .in_progress:
            openAIResponsesLogger.log("\(logPrefix): Response still in progress (status: \(statusString)). Continuing retry...")
            return false
          case .none:
            openAIResponsesLogger.warning("\(logPrefix): Unknown response status: \(statusString). Continuing retry...")
            return false
        }
      }
    } catch {
      openAIResponsesLogger.warning("\(logPrefix): Could not check response status: \(error). Continuing with retry...")
    }
    return false
  }

  func parseCompletedResponse(
    _ response: ResponseObject,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) {
    continuation.yield(response.toGenerationResponse())
  }

  func pollBackgroundResponse(
    responseId: String,
    apiKey: String?,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) async throws {
    let pollUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: pollUrl)
    request.httpMethod = "GET"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    while true {
      try Task.checkCancellation()

      let (data, urlResponse) = try await session.data(for: request)
      guard let httpResponse = urlResponse as? HTTPURLResponse else {
        throw AIError.network(underlying: URLError(.badServerResponse))
      }
      if !(200 ... 299).contains(httpResponse.statusCode) {
        try handleErrorResponse(httpResponse, data: data)
      }
      let response = try JSONDecoder().decode(ResponseObject.self, from: data)
      guard let statusString = response.status,
            let status = BackgroundResponseStatus(rawValue: statusString)
      else {
        throw AIError.parsing(message: "Failed to parse background response status")
      }
      switch status {
        case .queued, .in_progress:
          try await Task.sleep(nanoseconds: 2_000_000_000)
          continue

        case .completed, .incomplete, .failed:
          parseCompletedResponse(response, continuation: continuation)
          return

        case .cancelled:
          openAIResponsesLogger.log("Background response was cancelled")
          return
      }
    }
  }

  public func getBackgroundResponseStatus(responseId: String, apiKey: String?) async throws -> BackgroundResponse {
    let statusUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: statusUrl)
    request.httpMethod = "GET"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (data, urlResponse) = try await session.data(for: request)
    guard let httpResponse = urlResponse as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      try handleErrorResponse(httpResponse, data: data)
    }
    let response = try JSONDecoder().decode(ResponseObject.self, from: data)
    guard let statusString = response.status,
          let status = BackgroundResponseStatus(rawValue: statusString),
          let id = response.id
    else {
      throw AIError.parsing(message: "Failed to parse background response status")
    }

    let generationResponse: GenerationResponse? = if status == .completed || status == .incomplete || status == .failed {
      response.toGenerationResponse()
    } else {
      nil
    }

    return BackgroundResponse(
      id: id,
      status: status,
      response: generationResponse,
      error: response.error?.message,
    )
  }

  public func cancelBackgroundResponse(responseId: String, apiKey: String?) async throws {
    let cancelUrl = endpoint.appendingPathComponent(responseId).appendingPathComponent("cancel")
    var request = URLRequest(url: cancelUrl)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (_, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      if httpResponse.statusCode != 409 {
        throw AIError.fromHTTPStatusCode(httpResponse.statusCode)
      }
    }
  }

  public func deleteResponse(responseId: String, apiKey: String?) async throws {
    let deleteUrl = endpoint.appendingPathComponent(responseId)
    var request = URLRequest(url: deleteUrl)
    request.httpMethod = "DELETE"
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let (_, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      throw AIError.fromHTTPStatusCode(httpResponse.statusCode)
    }
  }

  public func resumeBackgroundStream(
    responseId: String,
    apiKey: String?,
    startingAfter: Int,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    await MainActor.run {
      isGenerating = true
      activeBackgroundResponseId = responseId
      activeBackgroundResponseApiKey = apiKey
    }

    let task = Task<GenerationResponse, Error> {
      var finalContent: [Message.Content] = []
      var finalMetadata: GenerationResponse.Metadata?

      let (stream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
      let backgroundTask = Task {
        do {
          try await streamBackgroundResponse(
            responseId: responseId,
            apiKey: apiKey,
            continuation: continuation,
            startingAfter: startingAfter,
          )
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { @Sendable _ in
        backgroundTask.cancel()
      }

      for try await chunk in stream {
        try Task.checkCancellation()

        finalContent = chunk.content
        finalMetadata = chunk.metadata

        await MainActor.run {
          update(chunk)
        }
      }

      return .init(content: finalContent, metadata: finalMetadata)
    }

    await MainActor.run {
      currentTask = task
    }
    let result = await task.result
    await cleanUpGeneration()
    return try result.get()
  }
}

private func replayContainsReasoningHistory(_ messages: [Message]) -> Bool {
  messages.contains { message in
    message.content.contains { item in
      switch item {
        case .thinking, .redactedThinking:
          true
        case let .providerOpaque(opaque):
          switch (opaque.provider, opaque.type) {
            case (OpaqueBlock.ProviderID.anthropic, OpaqueBlock.AnthropicType.thinking),
                 (OpaqueBlock.ProviderID.anthropic, OpaqueBlock.AnthropicType.redactedThinking),
                 (OpaqueBlock.ProviderID.openAIResponses, OpaqueBlock.OpenAIResponsesType.reasoning),
                 (OpaqueBlock.ProviderID.gemini, OpaqueBlock.GeminiType.thinking):
              true
            default:
              false
          }
        default:
          false
      }
    }
  }
}

private let responsesReplayWarningObserver = OSAllocatedUnfairLock(initialState: (@Sendable (String) -> Void)?.none)

func setResponsesReplayWarningObserver(_ observer: (@Sendable (String) -> Void)?) {
  responsesReplayWarningObserver.withLock { $0 = observer }
}

private func emitResponsesReplayWarning(_ message: String) {
  openAIResponsesLogger.warning("\(message, privacy: .public)")
  let observer = responsesReplayWarningObserver.withLock { $0 }
  observer?(message)
}
