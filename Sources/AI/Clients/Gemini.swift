// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log
import SSE

/// A client for the Google Gemini API.
///
/// Supports Gemini models with features like tool use, streaming, multimodal inputs,
/// and thinking mode for reasoning models.
///
/// ## Example
///
/// ```swift
/// let client = GeminiClient()
/// let response = try await client.generateText(
///   modelId: "gemini-2.0-flash",
///   prompt: "Hello, Gemini!",
///   apiKey: "your-api-key"
/// )
/// print(response.content)
/// ```
@Observable
public final class GeminiClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .audio, .file]

  private static let defaultModelsEndpoint = URL(string: "https://generativelanguage.googleapis.com/v1beta/models")!

  private let modelsEndpoint: URL

  /// URLSession with no timeout, since Gemini thinking requests can take several minutes.
  /// Callers that need a timeout can pass their own URLSession.
  public static let defaultSession: URLSession = {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = .infinity
    config.timeoutIntervalForResource = .infinity
    return URLSession(configuration: config)
  }()

  private let session: URLSession

  struct GeminiError: LocalizedError {
    let message: String
    let response: GenerateContentResponse?

    var errorDescription: String? {
      message
    }

    init(message: String, response: GenerateContentResponse? = nil) {
      self.message = message
      self.response = response
    }
  }

  struct GenerateContentResponse: Codable {
    var candidates: [Candidate]?
    let promptFeedback: PromptFeedback?
    var usageMetadata: UsageMetadata?
  }

  struct Candidate: Codable {
    var content: Content?
    var finishReason: FinishReason?
    var safetyRatings: [SafetyRating]?
    var citationMetadata: CitationMetadata?
    let tokenCount: Int?
    let avgLogprobs: Double?
    let index: Int?
    var groundingMetadata: GroundingMetadata?
  }

  struct Content: Codable {
    var parts: [Part]?
    let role: String?
  }

  struct Part: Codable {
    let text: String?
  }

  enum FinishReason: String, Codable {
    case stop = "STOP"
    case maxTokens = "MAX_TOKENS"
    case safety = "SAFETY"
    case recitation = "RECITATION"
    case language = "LANGUAGE"
    case blocklist = "BLOCKLIST"
    case prohibitedContent = "PROHIBITED_CONTENT"
    case spii = "SPII"
    case malformedFunctionCall = "MALFORMED_FUNCTION_CALL"
    case imageSafety = "IMAGE_SAFETY"
    case unexpectedToolCall = "UNEXPECTED_TOOL_CALL"
    case imageProhibitedContent = "IMAGE_PROHIBITED_CONTENT"
    case imageRecitation = "IMAGE_RECITATION"
    case imageOther = "IMAGE_OTHER"
    case noImage = "NO_IMAGE"
    case other = "OTHER"
    case unspecified = "FINISH_REASON_UNSPECIFIED"
  }

  struct SafetyRating: Codable {
    let category: String
    let probability: String
  }

  struct CitationMetadata: Codable {
    // Add relevant fields
  }

  struct PromptFeedback: Codable {
    // Add relevant fields
  }

  struct UsageMetadata: Codable {
    let promptTokenCount: Int?
    let candidatesTokenCount: Int?
    let totalTokenCount: Int?
    let cachedContentTokenCount: Int?
    let thoughtsTokenCount: Int?
  }

  /// Code execution
  struct ExecutableCode: Codable {
    let code: String?
    let language: String?
  }

  struct CodeExecutionResult: Codable {
    let outcome: String?
    let output: String?
  }

  /// Grounding metadata structures
  struct GroundingMetadata: Codable {
    let webSearchQueries: [String]?
    let groundingChunks: [GroundingChunk]?
    let groundingSupports: [GroundingSupport]?
    let searchEntryPoint: SearchEntryPoint?
  }

  struct SearchEntryPoint: Codable {
    let renderedContent: String?
  }

  struct GroundingSupport: Codable {
    let segment: Segment?
    let groundingChunkIndices: [Int]?
    let confidenceScores: [Double]?
  }

  struct Segment: Codable {
    let startIndex: Int?
    let endIndex: Int?
    let text: String?
  }

  struct GroundingChunk: Codable {
    let web: WebSource?
  }

  struct WebSource: Codable {
    let uri: String
    let title: String
  }

  @MainActor public private(set) var isGenerating: Bool = false

  @MainActor private var currentTask: Task<GenerationResponse, Error>?

  /// Creates a new Gemini client.
  ///
  /// - Parameters:
  ///   - session: URLSession to use for requests. Defaults to a session with no timeout.
  ///   - modelsEndpoint: Custom endpoint URL for the models API.
  public init(session: URLSession = GeminiClient.defaultSession, modelsEndpoint: URL? = nil) {
    self.session = session
    self.modelsEndpoint = modelsEndpoint ?? GeminiClient.defaultModelsEndpoint
  }

  private struct StreamResponse {
    let text: String?
    let thought: Bool?
    let thoughtSignature: String?
    let groundingMetadata: GroundingMetadata?
    let toolCall: ToolCall?
    let opaqueBlock: OpaqueBlock?
    let usageMetadata: UsageMetadata?
    let finishReason: FinishReason?
  }

  private static func assistantContent(
    reasoningText: String? = nil,
    reasoningSignature: String? = nil,
    responseText: String? = nil,
    notesText: String? = nil,
    toolCalls: [ToolCall] = [],
  ) -> [Message.Content] {
    // When a Gemini thought signature is present, store it as a provider-scoped opaque block
    // rather than in the generic .thinking(signature:) slot, which Anthropic treats as its own.
    if let reasoningText, !reasoningText.isEmpty, let reasoningSignature {
      var content: [Message.Content] = [
        .providerOpaque(OpaqueBlock(provider: "gemini", type: "thinking", content: reasoningText, signature: reasoningSignature)),
      ]
      if let responseText, !responseText.isEmpty {
        content.append(.text(responseText))
      }
      if let notesText, !notesText.isEmpty {
        content.append(.endnotes(notesText))
      }
      content.append(contentsOf: toolCalls.map(Message.Content.toolCall))
      return content
    }
    return Message.assistantContent(reasoningText: reasoningText, responseText: responseText, notesText: notesText, toolCalls: toolCalls)
  }

  private func requestParts(for message: Message, apiKey: String) async throws -> [[String: any Sendable]] {
    var parts: [[String: any Sendable]] = []

    for block in message.content {
      switch block {
        case let .toolCall(toolCall):
          var nativeArgs: [String: any Sendable] = [:]
          for (key, value) in toolCall.parameters {
            nativeArgs[key] = value.toAny()
          }
          var toolCallDict: [String: any Sendable] = [
            "name": toolCall.name,
            "args": nativeArgs,
          ]
          if !toolCall.id.isEmpty {
            toolCallDict["id"] = toolCall.id
          }
          var partDict: [String: any Sendable] = [
            "functionCall": toolCallDict,
          ]
          if let thoughtSignature = toolCall.providerMetadata?["thoughtSignature"] {
            partDict["thoughtSignature"] = thoughtSignature
          }
          parts.append(partDict)

        case let .toolResult(toolResult):
          var functionResponse: [String: any Sendable] = [
            "name": toolResult.name,
            "id": toolResult.id,
          ]

          if toolResult.isError == true {
            let errorText = toolResult.content.compactMap { content -> String? in
              if case let .text(text) = content { return text }
              return nil
            }.joined(separator: "\n")
            functionResponse["response"] = ["error": errorText.isEmpty ? "Unknown error" : errorText] as [String: any Sendable]
          } else {
            var inlineDataParts: [[String: any Sendable]] = []
            var textOutputs: [String] = []

            for content in toolResult.content {
              switch content {
                case let .text(text):
                  textOutputs.append(text)
                case let .image(data, mimeType):
                  let mediaType = mimeType ?? "image/png"
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mediaType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
                case let .audio(data, mimeType):
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mimeType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
                case let .file(data, mimeType, _):
                  inlineDataParts.append([
                    "inlineData": [
                      "mimeType": mimeType,
                      "data": data.base64EncodedString(),
                    ] as [String: any Sendable],
                  ])
              }
            }

            let joinedText = textOutputs.joined(separator: "\n")
            if !joinedText.isEmpty {
              functionResponse["response"] = ["output": joinedText] as [String: any Sendable]
            } else {
              functionResponse["response"] = [:] as [String: any Sendable]
            }
            if !inlineDataParts.isEmpty {
              functionResponse["parts"] = inlineDataParts
            }
          }

          parts.append(["functionResponse": functionResponse])

        case let .attachment(attachment):
          switch attachment.kind {
            case let .image(data, mimeType):
              let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
              parts.append([
                "inlineData": [
                  "mimeType": processedMimeType,
                  "data": processedImageData.base64EncodedString(),
                ],
              ])
            case let .video(data, mimeType):
              let fileUri = try await uploadFile(
                data: data,
                mimeType: mimeType,
                displayName: attachment.filename ?? "Video",
                apiKey: apiKey,
              )
              parts.append([
                "fileData": [
                  "mimeType": mimeType,
                  "fileUri": fileUri,
                ],
              ])
            case let .audio(data, mimeType):
              let fileUri = try await uploadFile(
                data: data,
                mimeType: mimeType,
                displayName: attachment.filename ?? "Audio",
                apiKey: apiKey,
              )
              parts.append([
                "fileData": [
                  "mimeType": mimeType,
                  "fileUri": fileUri,
                ],
              ])
            case let .document(data, mimeType):
              let mimeTypeForGemini = switch mimeType {
                case "net.daringfireball.markdown", "text/x-markdown": "text/md"
                default: mimeType
              }
              if data.count < 20_000_000 {
                parts.append([
                  "inlineData": [
                    "mimeType": mimeTypeForGemini,
                    "data": data.base64EncodedString(),
                  ],
                ])
              } else {
                let fileUri = try await uploadFile(
                  data: data,
                  mimeType: mimeTypeForGemini,
                  displayName: attachment.filename ?? "Document",
                  apiKey: apiKey,
                )
                parts.append([
                  "fileData": [
                    "mimeType": mimeTypeForGemini,
                    "fileUri": fileUri,
                  ],
                ])
              }
          }

        case let .text(text) where !text.isEmpty:
          parts.append(["text": text])

        case let .thinking(text, _):
          parts.append([
            "text": text,
            "thought": true,
          ])

        case let .providerOpaque(opaque) where opaque.provider == "gemini" && opaque.type == "thinking":
          var part: [String: any Sendable] = [
            "text": opaque.content ?? "",
            "thought": true,
          ]
          if let signature = opaque.signature {
            part["thoughtSignature"] = signature
          }
          parts.append(part)

        case let .providerOpaque(opaque) where opaque.provider == "gemini"
        && (opaque.type == "executableCode" || opaque.type == "codeExecutionResult"):
          if let jsonString = opaque.data,
             let jsonData = jsonString.data(using: .utf8),
             let jsonObject = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable]
          {
            parts.append([opaque.type: jsonObject])
          }

        default:
          break
      }
    }

    return parts
  }

  /// Sends a request to the Gemini API.
  ///
  /// When `streaming` is true, uses `:streamGenerateContent?alt=sse` and returns
  /// incremental SSE chunks. When false, uses `:generateContent` and returns the
  /// complete response as a single-element stream, matching the Gemini TS SDK's
  /// endpoint split.
  private func streamResponse(
    messages: [Message],
    systemPrompt: String?,
    modelId: String,
    apiKey: String,
    maxTokens: Int?,
    temperature: Float?,
    safetyThreshold: SafetyThreshold,
    searchGrounding: Bool,
    webContent: Bool,
    codeExecution: Bool,
    thinkingBudget: Int?, // For Gemini 2.5 models
    thinkingLevel: ThinkingLevel?, // For Gemini 3 models
    tools: [Tool] = [],
    streaming: Bool = true,
  ) async throws -> AsyncThrowingStream<StreamResponse, Error> {
    let action = streaming ? "streamGenerateContent" : "generateContent"
    let url = modelsEndpoint.appending(path: "\(modelId):\(action)")
    guard var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
      throw AIError.invalidRequest(message: "Failed to construct URL components for model: \(modelId)")
    }
    urlComponents.queryItems = [URLQueryItem(name: "key", value: apiKey)]
    if streaming {
      urlComponents.queryItems?.append(URLQueryItem(name: "alt", value: "sse"))
    }
    guard let requestURL = urlComponents.url else {
      throw AIError.invalidRequest(message: "Failed to construct request URL for model: \(modelId)")
    }

    var request = URLRequest(url: requestURL)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    // Note: X-Server-Timeout header is not set, matching TypeScript SDK behavior
    // The server will use its default timeout, which allows for long-running queries

    let patchedMessages = Message.patchingOrphanedToolCalls(messages)
    var processedMessages: [[String: any Sendable]] = []
    // Gemini only allows "user" and "model" roles in history (plus "function" for tool results).
    // System/developer messages are extracted and merged into system_instruction.
    var additionalSystemParts: [[String: any Sendable]] = []
    for message in patchedMessages {
      switch message.role {
        case .system, .developer:
          let text = message.content.compactMap { block -> String? in
            if case let .text(text) = block { return text }
            return nil
          }.joined(separator: "\n")
          if !text.isEmpty {
            additionalSystemParts.append(["text": text])
          }
        case .assistant, .user, .tool:
          let parts = try await requestParts(for: message, apiKey: apiKey)
          guard !parts.isEmpty else { continue }
          let role = switch message.role {
            case .assistant: "model"
            case .tool: "user"
            default: message.role.rawValue
          }
          // Merge consecutive same-role turns to maintain Gemini's required
          // user/model alternation (can happen when an empty model turn is skipped).
          if let lastRole = processedMessages.last?["role"] as? String, lastRole == role,
             var lastParts = processedMessages.last?["parts"] as? [[String: any Sendable]]
          {
            lastParts.append(contentsOf: parts)
            processedMessages[processedMessages.count - 1]["parts"] = lastParts
          } else {
            processedMessages.append([
              "role": role,
              "parts": parts,
            ])
          }
      }
    }

    var generationConfig: [String: any Sendable] = [:]

    if let maxTokens {
      generationConfig["maxOutputTokens"] = maxTokens
    }

    if let temperature {
      generationConfig["temperature"] = temperature
    }

    // Thinking configuration
    // Gemini 3 models use thinkingLevel, Gemini 2.5 models use thinkingBudget
    if let thinkingLevel {
      // Gemini 3 models: use thinkingLevel
      generationConfig["thinkingConfig"] = [
        "thinkingLevel": thinkingLevel.rawValue.uppercased(),
        "includeThoughts": true,
      ] as [String: any Sendable]
      // Don't set temperature for thinking models, since it can interfere with reasoning
      generationConfig["temperature"] = nil
    } else if let thinkingBudget {
      // Gemini 2.5 models: use thinkingBudget
      // A negative value (sentinel from thinkingConfig) means "enable thinking with server default budget"
      var thinkingConfig: [String: any Sendable] = ["includeThoughts": true]
      if thinkingBudget >= 0 {
        thinkingConfig["thinkingBudget"] = thinkingBudget
      }
      generationConfig["thinkingConfig"] = thinkingConfig
      // Don't set temperature for thinking models, since it can interfere with reasoning
      generationConfig["temperature"] = nil
    }

    // Tools
    var toolsArray: [[String: any Sendable]] = []

    // Search grounding
    if searchGrounding {
      toolsArray.append([
        "googleSearch": [:] as [String: any Sendable],
      ])
    }

    if webContent {
      toolsArray.append([
        "urlContext": [:] as [String: any Sendable],
      ])
    }

    // Code execution
    if codeExecution {
      toolsArray.append([
        "codeExecution": [:] as [String: any Sendable],
      ])
    }

    // Function calling - rawInputSchema is always populated
    if !tools.isEmpty {
      let functionDeclarations = tools.map { function in
        var declarationDict: [String: any Sendable] = [
          "name": function.name,
          "description": function.description,
        ]
        // Convert schema to Gemini format (uppercase types)
        let geminiSchema = Self.convertSchemaForGemini(function.rawInputSchema)
        declarationDict["parameters"] = geminiSchema
        return declarationDict
      }
      toolsArray.append([
        "functionDeclarations": functionDeclarations,
      ])
    }

    var body: [String: any Sendable] = [
      "contents": processedMessages,
      "generationConfig": generationConfig,
      // Safety threshold
      "safetySettings": [
        ["category": "HARM_CATEGORY_HARASSMENT", "threshold": safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": safetyThreshold.rawValue],
      ],
    ]

    // Only add tools if we have any
    if !toolsArray.isEmpty {
      body["tools"] = toolsArray

      // Add tool choice configuration if we have function declarations
      if !tools.isEmpty {
        body["toolConfig"] = [
          "functionCallingConfig": [
            "mode": "AUTO", // Can be "AUTO", "ANY", or "NONE"
          ],
        ]
      }
    }

    // System prompt and any system/developer messages extracted from history
    var systemParts: [[String: any Sendable]] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      systemParts.append(["text": systemPrompt])
    }
    systemParts.append(contentsOf: additionalSystemParts)
    if !systemParts.isEmpty {
      body["systemInstruction"] = ["parts": systemParts]
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    if streaming {
      let (result, response) = try await session.bytes(for: request)
      return processStreamBytes(result: result, response: response)
    } else {
      let (data, response) = try await session.data(for: request)
      return processBufferedResponse(data: data, response: response)
    }
  }

  /// Parse a single Gemini response JSON object and yield `StreamResponse` items.
  ///
  /// Returns `true` if parsing terminated the stream (e.g., a blocking error),
  /// meaning the caller should not continue processing further chunks.
  private func processResponseChunk(
    _ jsonObject: [String: any Sendable]?,
    continuation: AsyncThrowingStream<StreamResponse, Error>.Continuation,
  ) throws -> Bool {
    // Check for promptFeedback (indicates a blocked prompt)
    if let promptFeedback = jsonObject?["promptFeedback"] as? [String: any Sendable],
       let blockReason = promptFeedback["blockReason"] as? String
    {
      let blockMessage = (promptFeedback["blockReasonMessage"] as? String) ?? "Content was blocked."
      geminiLogger.warning("Prompt blocked: \(blockReason) - \(blockMessage)")

      let errorResponse = GenerateContentResponse(
        candidates: nil,
        promptFeedback: PromptFeedback(),
        usageMetadata: nil,
      )

      continuation.finish(throwing: GeminiError(
        message: "Your request was blocked: \(blockMessage)",
        response: errorResponse,
      ))
      return true
    }

    // Check for safety ratings in candidates
    if let candidates = jsonObject?["candidates"] as? [[String: any Sendable]],
       let firstCandidate = candidates.first
    {
      if let finishReason = firstCandidate["finishReason"] as? String {
        let contentBlockingReasons: Set = [
          "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT",
          "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
        ]
        if contentBlockingReasons.contains(finishReason) {
          let finishMessage = (firstCandidate["finishMessage"] as? String) ?? "Content was blocked due to finish reason \"\(finishReason.lowercased())\"."
          let candidate = Candidate(
            content: nil,
            finishReason: FinishReason(rawValue: finishReason) ?? .safety,
            safetyRatings: nil,
            citationMetadata: nil,
            tokenCount: nil,
            avgLogprobs: nil,
            index: 0,
            groundingMetadata: nil,
          )
          let errorResponse = GenerateContentResponse(
            candidates: [candidate],
            promptFeedback: nil,
            usageMetadata: nil,
          )
          continuation.finish(throwing: GeminiError(
            message: "\(finishMessage)",
            response: errorResponse,
          ))
          return true
        }
      }
    }

    // Extract text chunks
    if let candidates = jsonObject?["candidates"] as? [[String: any Sendable]],
       let firstCandidate = candidates.first,
       let content = firstCandidate["content"] as? [String: any Sendable],
       let parts = content["parts"] as? [[String: any Sendable]]
    {
      for part in parts {
        if let text = part["text"] as? String {
          let isThinkingText =
            if let thought = part["thought"] as? Bool { thought }
            else if let thought = part["thought"] as? Int { thought == 1 }
            else { false }
          let signature = isThinkingText ? part["thoughtSignature"] as? String : nil
          continuation.yield(StreamResponse(text: text, thought: isThinkingText, thoughtSignature: signature, groundingMetadata: nil, toolCall: nil, opaqueBlock: nil, usageMetadata: nil, finishReason: nil))
        } else if let functionCall = part["functionCall"] as? [String: any Sendable],
                  let name = functionCall["name"] as? String,
                  let args = functionCall["args"] as? [String: any Sendable]
        {
          let parameters = try args.mapValues { try Value.fromAny($0) }
          var providerMetadata: [String: String]?
          if let thoughtSignature = part["thoughtSignature"] as? String {
            providerMetadata = ["thoughtSignature": thoughtSignature]
          }
          let toolCallId = functionCall["id"] as? String ?? generateShortId()
          let toolCallResponse = ToolCall(
            name: name,
            id: toolCallId,
            parameters: parameters,
            providerMetadata: providerMetadata,
          )
          continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: nil, toolCall: toolCallResponse, opaqueBlock: nil, usageMetadata: nil, finishReason: nil))
        } else if let executableCodeDict = part["executableCode"] as? [String: any Sendable] {
          do {
            let jsonData = try JSONSerialization.data(withJSONObject: executableCodeDict)
            let executableCode = try JSONDecoder().decode(ExecutableCode.self, from: jsonData)
            let languageTag = (executableCode.language ?? "").lowercased()
            let displayText = executableCode.code.map { "\n\n```\(languageTag)\n\($0)\n```\n\n" }
            let rawJson = String(data: jsonData, encoding: .utf8)
            let block = OpaqueBlock(
              provider: "gemini", type: "executableCode",
              content: displayText, data: rawJson,
              isResponseContent: displayText != nil,
            )
            continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: nil, toolCall: nil, opaqueBlock: block, usageMetadata: nil, finishReason: nil))
          } catch {
            geminiLogger.error("Failed to decode ExecutableCode: \(error.localizedDescription)")
          }
        } else if let codeExecutionResultDict = part["codeExecutionResult"] as? [String: any Sendable] {
          do {
            let jsonData = try JSONSerialization.data(withJSONObject: codeExecutionResultDict)
            let executionResult = try JSONDecoder().decode(CodeExecutionResult.self, from: jsonData)
            let displayText = executionResult.output.map { "\n\n```\n\($0)\($0.last == "\n" ? "" : "\n")```\n\n" }
            let rawJson = String(data: jsonData, encoding: .utf8)
            let block = OpaqueBlock(
              provider: "gemini", type: "codeExecutionResult",
              content: displayText, data: rawJson,
              isResponseContent: displayText != nil,
            )
            continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: nil, toolCall: nil, opaqueBlock: block, usageMetadata: nil, finishReason: nil))
          } catch {
            geminiLogger.error("Failed to decode CodeExecutionResult: \(error.localizedDescription)")
          }
        } else if let thoughtSignature = part["thoughtSignature"] as? String {
          // Standalone thoughtSignature part (no text or functionCall)
          continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: thoughtSignature, groundingMetadata: nil, toolCall: nil, opaqueBlock: nil, usageMetadata: nil, finishReason: nil))
        }
      }
    }

    // Extract grounding metadata
    if let candidates = jsonObject?["candidates"] as? [[String: any Sendable]],
       let firstCandidate = candidates.first,
       let groundingMetadataDict = firstCandidate["groundingMetadata"] as? [String: any Sendable]
    {
      let metadataData = try JSONSerialization.data(withJSONObject: groundingMetadataDict)
      let groundingMetadata = try JSONDecoder().decode(GroundingMetadata.self, from: metadataData)
      continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: groundingMetadata, toolCall: nil, opaqueBlock: nil, usageMetadata: nil, finishReason: nil))
    }

    // Parse usage metadata
    if let usageMetadataDict = jsonObject?["usageMetadata"] as? [String: any Sendable] {
      do {
        let metadataData = try JSONSerialization.data(withJSONObject: usageMetadataDict)
        let decodedUsageMetadata = try JSONDecoder().decode(UsageMetadata.self, from: metadataData)
        continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: nil, toolCall: nil, opaqueBlock: nil, usageMetadata: decodedUsageMetadata, finishReason: nil))
      } catch {
        geminiLogger.error("Failed to decode usageMetadata: \(error.localizedDescription)")
      }
    }

    // Check for finish reason and yield it
    if let candidates = jsonObject?["candidates"] as? [[String: any Sendable]],
       let firstCandidate = candidates.first,
       let finishReasonString = firstCandidate["finishReason"] as? String
    {
      let finishReason = FinishReason(rawValue: finishReasonString) ?? .other
      continuation.yield(StreamResponse(text: nil, thought: nil, thoughtSignature: nil, groundingMetadata: nil, toolCall: nil, opaqueBlock: nil, usageMetadata: nil, finishReason: finishReason))
    }

    return false
  }

  /// Process SSE bytes into a stream of responses.
  private func processStreamBytes(result: URLSession.AsyncBytes, response: URLResponse) -> AsyncThrowingStream<StreamResponse, Error> {
    let (stream, continuation) = AsyncThrowingStream<StreamResponse, Error>.makeStream()
    let task = Task {
      do {
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
            errorMessage = Self.parseGeminiErrorMessage(from: errorData)
          } catch {
            geminiLogger.error("Failed to read error response: \(error)")
          }

          throw Self.geminiHTTPError(statusCode: httpResponse.statusCode, message: errorMessage)
        }

        for try await event in result.events {
          try Task.checkCancellation()
          let jsonString = event.data

          guard let data = jsonString.data(using: .utf8) else {
            throw AIError.parsing(message: "Failed to convert SSE payload to UTF-8 data")
          }
          let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: any Sendable]

          if try processResponseChunk(jsonObject, continuation: continuation) {
            return
          }
        }
        continuation.finish()
      } catch {
        geminiLogger.error("Stream processing error: \(error.localizedDescription)")
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  /// Process a buffered (non-streaming) Gemini response into a stream of responses.
  private func processBufferedResponse(data: Data, response: URLResponse) -> AsyncThrowingStream<StreamResponse, Error> {
    let (stream, continuation) = AsyncThrowingStream<StreamResponse, Error>.makeStream()
    let task = Task {
      do {
        guard let httpResponse = response as? HTTPURLResponse else {
          throw AIError.network(underlying: URLError(.badServerResponse))
        }

        if !(200 ... 299).contains(httpResponse.statusCode) {
          let errorMessage = Self.parseGeminiErrorMessage(from: data)
          throw Self.geminiHTTPError(statusCode: httpResponse.statusCode, message: errorMessage)
        }

        let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: any Sendable]
        _ = try processResponseChunk(jsonObject, continuation: continuation)
        continuation.finish()
      } catch {
        geminiLogger.error("Buffered response processing error: \(error.localizedDescription)")
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  /// Parse an error message from a Gemini error response body.
  private static func parseGeminiErrorMessage(from data: Data) -> String? {
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

  /// Extract grounding information from metadata
  private func formatGroundingInfo(from metadata: GroundingMetadata) async -> String? {
    var notes = [String]()
    // Add sources
    if let chunks = metadata.groundingChunks, !chunks.isEmpty {
      struct SourceReference {
        let index: Int
        let title: String
        let url: String
      }
      var resolvedSources = [SourceReference]() // Use the new struct
      // Resolve all URLs in parallel
      await withTaskGroup(of: SourceReference.self) { group in
        for (index, chunk) in chunks.enumerated() {
          if let webSource = chunk.web {
            group.addTask {
              let resolvedURL = await self.resolveRedirectURL(webSource.uri)
              return SourceReference(
                index: index,
                title: webSource.title,
                url: resolvedURL,
              )
            }
          }
        }
        // Collect results
        for await result in group {
          resolvedSources.append(result)
        }
      }
      // Sort by original index and format
      resolvedSources.sorted { $0.index < $1.index }.forEach { source in
        if source.url.starts(with: "https://vertexaisearch.cloud.google.com") {
          notes.append("- [\(source.title)](\(source.url))")
        } else {
          notes.append("- \(source.url)")
        }
      }
    }

    return notes.isEmpty ? nil : notes.joined(separator: "\n")
  }

  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Gemini model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature (0.0-2.0).
  ///   - apiKey: Gemini API key.
  ///   - configuration: Additional configuration options.
  /// - Returns: The generation response with text and metadata.
  public func generateText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await _generate(
      modelId: modelId,
      tools: Array(tools),
      systemPrompt: systemPrompt,
      messages: messages,
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
      update: { _ in },
    )
  }

  /// Streams a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The Gemini model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature (0.0-2.0).
  ///   - apiKey: Gemini API key.
  ///   - configuration: Additional configuration options.
  /// - Returns: An async stream of generation responses as they arrive.
  public func streamText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    messages: [Message],
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    let tools = Array(tools)
    let (stream, continuation) = AsyncThrowingStream<GenerationResponse, Error>.makeStream()
    let task = Task {
      do {
        let didYield = OSAllocatedUnfairLock(initialState: false)
        let finalResponse = try await _generate(
          modelId: modelId,
          tools: tools,
          systemPrompt: systemPrompt,
          messages: messages,
          maxTokens: maxTokens,
          temperature: temperature,
          apiKey: apiKey,
          configuration: configuration,
          streaming: true,
          update: { response in
            didYield.withLock { $0 = true }
            continuation.yield(response)
          },
        )
        // Only yield the final response if no updates were emitted (e.g. early cancellation)
        if !didYield.withLock({ $0 }) {
          continuation.yield(finalResponse)
        }
        continuation.finish()
      } catch {
        continuation.finish(throwing: error)
      }
    }
    continuation.onTermination = { @Sendable _ in
      task.cancel()
    }
    return stream
  }

  /// Generate a text response using a simple prompt string.
  public func generateText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) async throws -> GenerationResponse {
    try await generateText(
      modelId: modelId,
      tools: Array(tools),
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
    )
  }

  /// Generate a text response with streaming using a simple prompt string.
  public func streamText(
    modelId: String,
    tools: some Collection<Tool> = [],
    systemPrompt: String? = nil,
    prompt: String,
    maxTokens: Int? = nil,
    temperature: Float? = nil,
    apiKey: String? = nil,
    configuration: Configuration = .init(),
  ) -> AsyncThrowingStream<GenerationResponse, Error> {
    streamText(
      modelId: modelId,
      tools: Array(tools),
      systemPrompt: systemPrompt,
      messages: [Message(role: .user, content: prompt)],
      maxTokens: maxTokens,
      temperature: temperature,
      apiKey: apiKey,
      configuration: configuration,
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
    streaming: Bool = false,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    guard let apiKey else {
      throw AIError.authentication(message: "Missing API key")
    }
    await MainActor.run {
      isGenerating = true
    }
    let task = Task<GenerationResponse, Error> {
      var fullReasoningText = ""
      var reasoningSignature: String?
      var fullResponseText = ""
      var notesText: String?
      var toolCalls: [ToolCall] = []
      var opaqueBlocks: [OpaqueBlock] = []
      var usageMetadata: UsageMetadata?
      var finishReason: FinishReason?

      do {
        let stream = try await streamResponse(
          messages: messages,
          systemPrompt: systemPrompt,
          modelId: modelId,
          apiKey: apiKey,
          maxTokens: maxTokens,
          temperature: temperature,
          safetyThreshold: configuration.safetyThreshold,
          searchGrounding: configuration.searchGrounding,
          webContent: configuration.webContent,
          codeExecution: configuration.codeExecution,
          thinkingBudget: configuration.thinkingBudget,
          thinkingLevel: configuration.thinkingLevel,
          tools: tools,
          streaming: streaming,
        )

        func buildMetadata() -> GenerationResponse.Metadata {
          let generationFinishReason: GenerationResponse.FinishReason? = if let reason = finishReason {
            switch reason {
              case .stop: .stop
              case .maxTokens: .maxTokens
              case .safety, .recitation, .blocklist, .prohibitedContent, .spii,
                   .imageSafety, .imageProhibitedContent, .imageRecitation: .contentFilter
              case .malformedFunctionCall, .unexpectedToolCall,
                   .language, .noImage, .imageOther, .other, .unspecified: .other
            }
          } else {
            nil
          }
          // Gemini sends finishReason STOP even when the response contains function calls
          let effectiveFinishReason = if !toolCalls.isEmpty { GenerationResponse.FinishReason.toolUse } else { generationFinishReason }
          return GenerationResponse.Metadata(
            finishReason: effectiveFinishReason,
            inputTokens: usageMetadata?.promptTokenCount,
            outputTokens: usageMetadata?.candidatesTokenCount,
            totalTokens: usageMetadata?.totalTokenCount,
            cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
            reasoningTokens: usageMetadata?.thoughtsTokenCount,
          )
        }

        let sendUpdate = {
          let blocks = Self.assistantContent(
            reasoningText: fullReasoningText,
            reasoningSignature: reasoningSignature,
            responseText: fullResponseText,
            notesText: notesText,
            toolCalls: toolCalls,
          ) + opaqueBlocks.map(Message.Content.providerOpaque)
          let metadata = buildMetadata()
          await MainActor.run {
            update(.init(content: blocks, metadata: metadata))
          }
        }

        for try await chunk in stream {
          try Task.checkCancellation()

          if let metadata = chunk.usageMetadata {
            usageMetadata = metadata
          }

          if let reason = chunk.finishReason {
            finishReason = reason
          }

          // Handle text chunks
          if let text = chunk.text {
            if let isThinkingText = chunk.thought, isThinkingText {
              fullReasoningText += text
              if let signature = chunk.thoughtSignature {
                reasoningSignature = signature
              }
            } else {
              fullResponseText += text
            }
            await sendUpdate()
          } else if let signature = chunk.thoughtSignature {
            // Standalone thoughtSignature part (no text)
            reasoningSignature = signature
          }

          // Handle function calls
          if let toolCall = chunk.toolCall {
            toolCalls.append(toolCall)
            await sendUpdate()
          }

          // Handle opaque blocks (code execution parts)
          if let opaqueBlock = chunk.opaqueBlock {
            opaqueBlocks.append(opaqueBlock)
            await sendUpdate()
          }

          // Handle grounding metadata
          if let metadata = chunk.groundingMetadata {
            notesText = await formatGroundingInfo(from: metadata)
            if notesText != nil {
              await sendUpdate()
            }
          }
        }

        // Yield final state with complete metadata (usage and finish reason may arrive
        // in chunks that don't contain content, so the last content-triggered update
        // may lack them)
        await sendUpdate()

        // Return the final state
        return .init(
          content: Self.assistantContent(
            reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
            reasoningSignature: reasoningSignature,
            responseText: fullResponseText.isEmpty ? nil : fullResponseText,
            notesText: notesText,
            toolCalls: toolCalls,
          ) + opaqueBlocks.map(Message.Content.providerOpaque),
          metadata: buildMetadata(),
        )
      } catch let error as GeminiError {
        // Check if the task was cancelled
        if Task.isCancelled {
          // Build partial metadata
          let partialMetadata = GenerationResponse.Metadata(
            inputTokens: usageMetadata?.promptTokenCount,
            outputTokens: usageMetadata?.candidatesTokenCount,
            totalTokens: usageMetadata?.totalTokenCount,
            cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
            reasoningTokens: usageMetadata?.thoughtsTokenCount,
          )
          // Return partial results without throwing an error
          return .init(
            content: Self.assistantContent(
              reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
              reasoningSignature: reasoningSignature,
              responseText: fullResponseText.isEmpty ? nil : fullResponseText,
              notesText: notesText,
            ) + opaqueBlocks.map(Message.Content.providerOpaque),
            metadata: partialMetadata,
          )
        }

        geminiLogger.warning("Gemini error: \(error.message)")
        let errorMessage = if error.message.contains("SAFETY") {
          "Response blocked due to safety filters"
        } else if error.message.contains("RECITATION") {
          "Response blocked due to content recitation"
        } else {
          error.message
        }
        throw AIError.serverError(statusCode: 0, message: errorMessage, context: nil)
      } catch {
        // Handle cancellation
        if error is CancellationError || Task.isCancelled {
          // Build partial metadata
          let partialMetadata = GenerationResponse.Metadata(
            inputTokens: usageMetadata?.promptTokenCount,
            outputTokens: usageMetadata?.candidatesTokenCount,
            totalTokens: usageMetadata?.totalTokenCount,
            cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
            reasoningTokens: usageMetadata?.thoughtsTokenCount,
          )
          return .init(
            content: Self.assistantContent(
              reasoningText: fullReasoningText.isEmpty ? nil : fullReasoningText,
              reasoningSignature: reasoningSignature,
              responseText: fullResponseText.isEmpty ? nil : fullResponseText,
              notesText: notesText,
              toolCalls: toolCalls,
            ) + opaqueBlocks.map(Message.Content.providerOpaque),
            metadata: partialMetadata,
          )
        } else {
          throw error
        }
      }
    }
    await MainActor.run {
      currentTask = task
    }
    let result = await task.result
    await cleanUpGeneration()
    return try result.get()
  }

  @MainActor
  private func cleanUpGeneration() {
    isGenerating = false
    currentTask = nil
  }

  /// Cancels any ongoing generation task.
  @MainActor
  public func stop() {
    currentTask?.cancel()
  }

  private func uploadFile(data: Data, mimeType: String, displayName: String, apiKey: String) async throws -> String {
    // Derive upload URL from the configured models endpoint so custom endpoints (proxies, mocks) work.
    // modelsEndpoint path is e.g. "/v1beta/models" or "/prefix/v1beta/models";
    // replace the version+models suffix with the upload path, preserving any proxy prefix.
    var uploadComponents = URLComponents(url: modelsEndpoint, resolvingAgainstBaseURL: true)!
    let path = uploadComponents.path
    if let range = path.range(of: "/v1beta/models", options: .backwards) {
      uploadComponents.path = String(path[..<range.lowerBound]) + "/upload/v1beta/files"
    } else {
      uploadComponents.path = "/upload/v1beta/files"
    }
    uploadComponents.queryItems = nil
    let uploadURL = uploadComponents.url!
    var components = URLComponents(url: uploadURL, resolvingAgainstBaseURL: true)!
    components.queryItems = [URLQueryItem(name: "key", value: apiKey)]
    // Start resumable upload
    var request = URLRequest(url: components.url!)
    request.httpMethod = "POST"
    request.setValue("resumable", forHTTPHeaderField: "X-Goog-Upload-Protocol")
    request.setValue("start", forHTTPHeaderField: "X-Goog-Upload-Command")
    request.setValue("\(data.count)", forHTTPHeaderField: "X-Goog-Upload-Header-Content-Length")
    request.setValue(mimeType, forHTTPHeaderField: "X-Goog-Upload-Header-Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let metadata = ["file": ["displayName": displayName]]
    request.httpBody = try JSONSerialization.data(withJSONObject: metadata)
    let (responseData, response) = try await session.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
    if !(200 ... 299).contains(httpResponse.statusCode) {
      let errorMessage = String(data: responseData, encoding: .utf8) ?? "Unknown error"
      throw AIError.fromHTTPStatusCode(httpResponse.statusCode, message: errorMessage)
    }
    guard let uploadUrl = httpResponse.value(forHTTPHeaderField: "X-Goog-Upload-URL") else {
      throw AIError.parsing(message: "Failed to get upload URL from response headers")
    }
    // Upload the actual file
    guard let uploadURL = URL(string: uploadUrl) else {
      throw AIError.parsing(message: "Invalid upload URL from server: \(uploadUrl)")
    }
    var uploadRequest = URLRequest(url: uploadURL)
    uploadRequest.httpMethod = "POST"
    uploadRequest.setValue("\(data.count)", forHTTPHeaderField: "Content-Length")
    uploadRequest.setValue("0", forHTTPHeaderField: "X-Goog-Upload-Offset")
    uploadRequest.setValue("upload, finalize", forHTTPHeaderField: "X-Goog-Upload-Command")
    uploadRequest.httpBody = data
    let (uploadResponseData, uploadResponse) = try await session.data(for: uploadRequest)
    if let uploadHttpResponse = uploadResponse as? HTTPURLResponse, !(200 ... 299).contains(uploadHttpResponse.statusCode) {
      let errorMessage = String(data: uploadResponseData, encoding: .utf8) ?? "Unknown error"
      throw AIError.fromHTTPStatusCode(uploadHttpResponse.statusCode, message: errorMessage)
    }
    do {
      // Response structure for the upload
      struct FileResponse: Codable {
        struct File: Codable {
          let uri: String
          let state: String
        }

        let file: File
      }

      // Status check response structure
      struct StatusResponse: Codable {
        let uri: String
        let state: String

        struct ErrorInfo: Codable {
          let code: Int
          let message: String
        }

        let error: ErrorInfo?
      }

      var fileResponse = try JSONDecoder().decode(FileResponse.self, from: uploadResponseData)
      let fileUri = fileResponse.file.uri
      // Wait for video processing to complete
      while fileResponse.file.state == "PROCESSING" {
        try Task.checkCancellation()
        try await Task.sleep(for: .seconds(2))
        // Use the full URI from the response
        guard let checkURL = URL(string: fileUri),
              var checkComponents = URLComponents(url: checkURL, resolvingAgainstBaseURL: true)
        else {
          throw AIError.parsing(message: "Invalid file URI from server: \(fileUri)")
        }
        checkComponents.queryItems = [URLQueryItem(name: "key", value: apiKey)]
        guard let checkRequestURL = checkComponents.url else {
          throw AIError.parsing(message: "Failed to construct status check URL for file: \(fileUri)")
        }
        let (checkData, _) = try await session.data(from: checkRequestURL)
        // Decode the status check response
        let statusResponse = try JSONDecoder().decode(StatusResponse.self, from: checkData)
        // Check for processing failure
        if statusResponse.state == "FAILED" {
          if let error = statusResponse.error {
            throw AIError.serverError(statusCode: 0, message: error.message, context: nil)
          } else {
            throw AIError.serverError(statusCode: 0, message: "File processing failed with unknown error", context: nil)
          }
        }
        // Update state from status check
        fileResponse = FileResponse(file: .init(uri: fileResponse.file.uri, state: statusResponse.state))
      }
      // Return the complete URI for use with the Gemini API
      return fileUri
    } catch {
      geminiLogger.error("Decoding error: \(error)")
      // Try to decode as an error response format

      struct ErrorResponse: Codable {
        let error: ErrorDetail

        struct ErrorDetail: Codable {
          let code: Int
          let message: String
          let status: String
        }
      }
      if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: uploadResponseData) {
        throw AIError.serverError(statusCode: errorResponse.error.code, message: errorResponse.error.message, context: nil)
      }
      throw AIError.parsing(message: "Failed to decode response: \(error.localizedDescription)")
    }
  }

  /// Used to get actual URL in search results references instead of Google tracking URL
  private func resolveRedirectURL(_ url: String) async -> String {
    guard let originalURL = URL(string: url) else { return url }
    // Create a HEAD request to follow redirects without downloading content
    var request = URLRequest(url: originalURL)
    request.httpMethod = "HEAD"
    request.timeoutInterval = 5
    do {
      let (_, response) = try await session.data(for: request)
      if let httpResponse = response as? HTTPURLResponse, let finalURL = httpResponse.url {
        return finalURL.absoluteString
      }
    } catch {
      // In the case of ATS errors on http-only URLs, try to extract the blocked URL from the error
      if let urlError = error as? URLError, case .appTransportSecurityRequiresSecureConnection = urlError.code {
        if let blockedURL = extractURLFromATSError(error) {
          return blockedURL
        } else {
          geminiLogger.warning("Could not extract URL from ATS error")
        }
      } else {
        geminiLogger.warning("Failed to resolve redirect URL '\(originalURL.absoluteString)' for reference in response: \(error.localizedDescription)")
      }
    }
    // Return the original URL if resolution fails
    return url
  }

  /// Extract the blocked URL from ATS error
  private func extractURLFromATSError(_ error: Error) -> String? {
    let nsError = error as NSError
    // Check the error's userInfo dictionary for URL information
    if let failingURL = nsError.userInfo[NSURLErrorFailingURLErrorKey] as? URL {
      return failingURL.absoluteString
    }
    // For URLError, check the failureURLString property
    if let urlError = error as? URLError {
      if let failureURLString = urlError.failingURL?.absoluteString {
        return failureURLString
      }
    }
    return nil
  }
}

extension GeminiClient {
  /// Converts a raw JSON schema (from rawInputSchema) to Gemini format.
  /// Gemini requires uppercase type values ("STRING" not "string") and
  /// doesn't support "additionalProperties".
  static func convertSchemaForGemini(_ schema: [String: Value]) -> [String: any Sendable] {
    var result: [String: any Sendable] = [:]

    // Pre-process: if schema has anyOf with a null type, extract nullable and unwrap
    var effectiveSchema = schema
    if let anyOf = schema["anyOf"]?.arrayValue {
      let nullTypes = anyOf.filter { $0.objectValue?["type"]?.stringValue == "null" }
      let nonNullTypes = anyOf.filter { $0.objectValue?["type"]?.stringValue != "null" }
      if !nullTypes.isEmpty {
        result["nullable"] = true
      }
      if nonNullTypes.count == 1, let single = nonNullTypes.first?.objectValue {
        // Single non-null type: unwrap the anyOf and merge into the schema
        effectiveSchema = schema.filter { $0.key != "anyOf" }
        for (k, v) in single {
          effectiveSchema[k] = v
        }
      } else if nonNullTypes.count > 1 {
        // Multiple non-null types: keep as anyOf with null types removed
        effectiveSchema = schema
      }
    }

    for (key, value) in effectiveSchema {
      // Skip additionalProperties - Gemini doesn't support this field
      if key == "additionalProperties" {
        continue
      }

      if key == "type" {
        // Convert type values to uppercase
        if case let .string(typeStr) = value {
          result[key] = typeStr.uppercased()
        } else if case let .array(types) = value {
          // Handle nullable type arrays like ["string", "null"] → type: "STRING", nullable: true
          let typeStrings = types.compactMap(\.stringValue)
          let nonNullTypes = typeStrings.filter { $0 != "null" }
          if typeStrings.contains("null") {
            result["nullable"] = true
          }
          if nonNullTypes.count == 1 {
            result[key] = nonNullTypes[0].uppercased()
          } else if nonNullTypes.count > 1 {
            result["anyOf"] = nonNullTypes.map { ["type": $0.uppercased()] as [String: any Sendable] }
            continue
          } else {
            result[key] = "STRING"
          }
        } else {
          result[key] = value.toAny()
        }
      } else if key == "properties" {
        // Recursively convert property schemas
        if case let .object(props) = value {
          var convertedProps: [String: any Sendable] = [:]
          for (propName, propSchema) in props {
            if case let .object(propSchemaDict) = propSchema {
              convertedProps[propName] = convertSchemaForGemini(propSchemaDict)
            } else {
              convertedProps[propName] = propSchema.toAny()
            }
          }
          result[key] = convertedProps
        } else {
          result[key] = value.toAny()
        }
      } else if key == "items" {
        // Recursively convert array item schema
        if case let .object(itemSchema) = value {
          let convertedItems = convertSchemaForGemini(itemSchema)
          result[key] = convertedItems
        } else {
          result[key] = value.toAny()
        }
      } else if key == "anyOf" || key == "oneOf" {
        // Recursively convert anyOf/oneOf schemas, extracting null types
        // Gemini treats oneOf the same as anyOf
        if case let .array(schemas) = value {
          var converted: [[String: any Sendable]] = []
          for item in schemas {
            if item.objectValue?["type"]?.stringValue == "null" {
              result["nullable"] = true
              continue
            }
            if let obj = item.objectValue {
              converted.append(convertSchemaForGemini(obj))
            }
          }
          if !converted.isEmpty {
            result[key] = converted
          }
        }
      } else if key == "$defs" {
        // Recursively convert schema definitions
        if case let .object(defs) = value {
          var convertedDefs: [String: any Sendable] = [:]
          for (defName, defSchema) in defs {
            if case let .object(defSchemaDict) = defSchema {
              convertedDefs[defName] = convertSchemaForGemini(defSchemaDict)
            } else {
              convertedDefs[defName] = defSchema.toAny()
            }
          }
          result[key] = convertedDefs
        }
      } else {
        result[key] = value.toAny()
      }
    }

    return result
  }

  /// Maps an HTTP status code and optional error message to an `AIError`.
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
}

private let geminiLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "GeminiClient")

// MARK: - Configuration

public extension GeminiClient {
  /// Configuration options for Gemini API requests.
  struct Configuration: Sendable {
    /// Content safety filtering threshold.
    public var safetyThreshold: SafetyThreshold

    /// Enables Google Search grounding for factual responses.
    public var searchGrounding: Bool

    /// Enables web content fetching for retrieving page content from URLs.
    public var webContent: Bool

    /// Enables code execution in a sandboxed Python environment.
    public var codeExecution: Bool

    /// Token budget for extended thinking (Gemini 2.5 models).
    /// Use `thinkingLevel` for Gemini 3 models instead.
    public var thinkingBudget: Int?

    /// Thinking level for reasoning depth (Gemini 3 models).
    /// Use `thinkingBudget` for Gemini 2.5 models instead.
    public var thinkingLevel: ThinkingLevel?

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - safetyThreshold: Content safety filtering level.
    ///   - searchGrounding: Enable Google Search grounding.
    ///   - webContent: Enable web content fetching.
    ///   - codeExecution: Enable sandboxed code execution.
    ///   - thinkingBudget: Token budget for thinking (Gemini 2.5).
    ///   - thinkingLevel: Thinking level (Gemini 3).
    public init(
      safetyThreshold: SafetyThreshold = .none,
      searchGrounding: Bool = false,
      webContent: Bool = false,
      codeExecution: Bool = false,
      thinkingBudget: Int? = nil,
      thinkingLevel: ThinkingLevel? = nil,
    ) {
      self.safetyThreshold = safetyThreshold
      self.searchGrounding = searchGrounding
      self.webContent = webContent
      self.codeExecution = codeExecution
      self.thinkingBudget = thinkingBudget
      self.thinkingLevel = thinkingLevel
    }
  }

  /// Thinking level for Gemini 3 models.
  enum ThinkingLevel: String, CaseIterable, Identifiable, Codable, Equatable, Hashable, Sendable {
    /// Matches "no thinking" for most queries. Flash-only.
    case minimal
    /// Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput applications.
    case low
    /// Balanced thinking for most tasks. Flash-only.
    case medium
    /// Maximizes reasoning depth. Supported by both Pro and Flash.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Content safety filtering threshold levels.
  enum SafetyThreshold: String, CaseIterable, Identifiable, Sendable {
    /// No content filtering.
    case none = "BLOCK_NONE"
    /// Block only high-probability harmful content.
    case high = "BLOCK_ONLY_HIGH"
    /// Block medium and high probability harmful content.
    case medium = "BLOCK_MEDIUM_AND_ABOVE"
    /// Block low, medium, and high probability harmful content.
    case low = "BLOCK_LOW_AND_ABOVE"

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Returns the appropriate thinking configuration for a Gemini model.
  /// Defaults to `thinkingLevel` (Gemini 3+ paradigm). Uses `thinkingBudget` for
  /// Gemini 2.5 models, and disables thinking for older models.
  static func thinkingConfig(
    for modelId: String,
    reasoning: Bool,
  ) -> (thinkingLevel: ThinkingLevel?, thinkingBudget: Int?) {
    guard reasoning, modelId.hasPrefix("gemini-") else {
      return (nil, nil)
    }
    if modelId.hasPrefix("gemini-2.0") || modelId.hasPrefix("gemini-1") {
      return (nil, nil)
    }
    if modelId.hasPrefix("gemini-2.5") {
      return (nil, -1)
    }
    return (.high, nil)
  }
}
