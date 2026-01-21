// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log

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
/// print(response.texts.response ?? "")
/// ```
@Observable
public final class GeminiClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .audio, .file]

  private static let defaultModelsEndpoint = URL(string: "https://generativelanguage.googleapis.com/v1beta/models")!

  private let modelsEndpoint: URL

  // URLSession configured with no timeout for long-running requests (like extended thinking)
  // This mirrors the TypeScript SDK approach which doesn't set a default timeout
  public static let defaultSession: URLSession = {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = .infinity // No timeout for individual requests
    config.timeoutIntervalForResource = .infinity // No timeout for entire resource transfer
    return URLSession(configuration: config)
  }()

  private let session: URLSession

  struct GeminiError: LocalizedError {
    let message: String
    let response: GenerateContentResponse?

    var errorDescription: String? { message }

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
    case other = "FINISH_REASON_UNSPECIFIED"
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

  // Code execution
  struct ExecutableCode: Codable {
    let code: String?
    let language: String?
  }

  struct CodeExecutionResult: Codable {
    let outcome: String?
    let output: String?
  }

  // Grounding metadata structures
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
    let groundingMetadata: GroundingMetadata?
    let toolCall: GenerationResponse.ToolCall?
    let usageMetadata: UsageMetadata?
    let finishReason: FinishReason?
  }

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
    tools: [Tool] = []
  ) async throws -> AsyncThrowingStream<StreamResponse, Error> {
    let url = modelsEndpoint.appending(path: "\(modelId):streamGenerateContent")
    var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: true)!
    urlComponents.queryItems = [URLQueryItem(name: "key", value: apiKey)]
    urlComponents.queryItems?.append(URLQueryItem(name: "alt", value: "sse"))

    var request = URLRequest(url: urlComponents.url!)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    // Note: X-Server-Timeout header is not set, matching TypeScript SDK behavior
    // The server will use its default timeout, which allows for long-running queries

    var processedMessages: [[String: any Sendable]] = []
    for message in messages {
      var parts: [[String: any Sendable]] = []

      // Handle function calls
      if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
        for toolCall in toolCalls {
          // Convert swift types from parameters to native types compatible with JSON serialization
          var nativeArgs: [String: any Sendable] = [:]
          for (key, value) in toolCall.parameters {
            nativeArgs[key] = value.toAny()
          }
          let toolCallDict: [String: any Sendable] = [
            "name": toolCall.name,
            "args": nativeArgs,
          ]
          var partDict: [String: any Sendable] = [
            "functionCall": toolCallDict,
          ]
          // Include thoughtSignature if present (required for Gemini tool use)
          if let thoughtSignature = toolCall.providerMetadata?["thoughtSignature"] {
            partDict["thoughtSignature"] = thoughtSignature
          }
          parts.append(partDict)
        }
      }

      // Handle function results (tool results)
      // TODO: Verify Gemini's support for multi-content tool results.
      // Current approach: text goes in response.output, binary data in inlineData parts.
      // Need to confirm this structure is correct per Gemini API docs and test with
      // actual multi-content responses (text + image, multiple images, etc.)
      if let toolResults = message.toolResults, !toolResults.isEmpty {
        // Create a separate functionResponse part for each function result
        for toolResult in toolResults {
          var functionResponse: [String: any Sendable] = [
            "name": toolResult.name,
          ]

          // Handle error results
          if toolResult.isError == true {
            let errorText = toolResult.content.compactMap { content -> String? in
              if case let .text(text) = content { return text }
              return nil
            }.joined(separator: "\n")
            functionResponse["response"] = ["error": errorText.isEmpty ? "Unknown error" : errorText] as [String: any Sendable]
          } else {
            // Process content items
            var inlineDataParts: [[String: any Sendable]] = []
            var textOutput: String? = nil

            for content in toolResult.content {
              switch content {
                case let .text(text):
                  textOutput = text
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

            functionResponse["response"] = ["output": textOutput ?? (inlineDataParts.isEmpty ? "" : "Content provided")] as [String: any Sendable]
            if !inlineDataParts.isEmpty {
              functionResponse["parts"] = inlineDataParts
            }
          }

          parts.append(["functionResponse": functionResponse])
        }
      }

      // Add attachments
      if !message.attachments.isEmpty {
        for attachment in message.attachments {
          switch attachment.kind {
            case let .image(data, mimeType):
              do {
                // Resize image if necessary before encoding
                let processedImageData = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
                parts.append([
                  "inline_data": [
                    "mime_type": mimeType,
                    "data": processedImageData.base64EncodedString(),
                  ],
                ])
              } catch {
                geminiLogger.error("Failed to process image: \(error.localizedDescription)")
                throw error
              }
            case let .video(data, mimeType):
              // Videos must use the File API due to size
              do {
                let fileUri = try await uploadFile(
                  data: data,
                  mimeType: mimeType,
                  displayName: attachment.filename ?? "Video",
                  apiKey: apiKey
                )
                parts.append([
                  "file_data": [
                    "mime_type": mimeType,
                    "file_uri": fileUri,
                  ],
                ])
              } catch {
                geminiLogger.error("Failed to upload video: \(error.localizedDescription)")
                throw error
              }
            case let .audio(data, mimeType):
              do {
                let fileUri = try await uploadFile(
                  data: data,
                  mimeType: mimeType,
                  displayName: attachment.filename ?? "Audio",
                  apiKey: apiKey
                )
                parts.append([
                  "file_data": [
                    "mime_type": mimeType,
                    "file_uri": fileUri,
                  ],
                ])
              } catch {
                geminiLogger.error("Failed to upload audio: \(error.localizedDescription)")
                throw error
              }
            case let .document(data, mimeType):
              // For documents under 20MB, use inline data
              let mimeTypeForGemini = switch mimeType {
                case "net.daringfireball.markdown", "text/x-markdown": "text/md"
                default: mimeType
              }
              if data.count < 20_000_000 {
                parts.append([
                  "inline_data": [
                    "mime_type": mimeTypeForGemini,
                    "data": data.base64EncodedString(),
                  ],
                ])
              } else {
                // For larger documents, use the File API
                do {
                  let fileUri = try await uploadFile(
                    data: data,
                    mimeType: mimeTypeForGemini,
                    displayName: attachment.filename ?? "Document",
                    apiKey: apiKey
                  )
                  parts.append([
                    "file_data": [
                      "mime_type": mimeTypeForGemini,
                      "file_uri": fileUri,
                    ],
                  ])
                } catch {
                  geminiLogger.error("Failed to upload document: \(error.localizedDescription)")
                  throw error
                }
              }
          }
        }
      }

      // Add text part after attachments
      if let content = message.content, !content.isEmpty {
        parts.append(["text": content])
      }

      let role = switch message.role {
        case .assistant: "model" // Gemini uses "model" instead of "assistant"
        case .tool: "function" // For function responses
        default: message.role.rawValue
      }

      processedMessages.append([
        "role": role,
        "parts": parts,
      ])
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
      generationConfig["thinkingConfig"] = [
        "thinkingBudget": thinkingBudget,
        "includeThoughts": true,
      ] as [String: any Sendable]
      // Don't set temperature for thinking models, since it can interfere with reasoning
      generationConfig["temperature"] = nil
    }

    // Tools
    var toolsArray: [[String: any Sendable]] = []

    // Search grounding
    if searchGrounding {
      toolsArray.append([
        "google_search": [:] as [String: any Sendable],
      ])
    }

    if webContent {
      toolsArray.append([
        "url_context": [:] as [String: any Sendable],
      ])
    }

    // Code execution
    if codeExecution {
      toolsArray.append([
        "code_execution": [:] as [String: any Sendable],
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
        "function_declarations": functionDeclarations,
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
        body["tool_config"] = [
          "function_calling_config": [
            "mode": "AUTO", // Can be "AUTO", "ANY", or "NONE"
          ],
        ]
      }
    }

    // System prompt
    if let systemPrompt, !systemPrompt.isEmpty {
      body["system_instruction"] = ["parts": [["text": systemPrompt]]]
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: body)
    let (result, response) = try await session.bytes(for: request)
    return processStreamBytes(result: result, response: response)
  }

  // Process the raw bytes into a stream of responses
  private func processStreamBytes(result: URLSession.AsyncBytes, response: URLResponse) -> AsyncThrowingStream<StreamResponse, Error> {
    AsyncThrowingStream { continuation in
      Task {
        do {
          guard let httpResponse = response as? HTTPURLResponse else {
            throw AIError.network(underlying: URLError(.badServerResponse))
          }

          if !(200 ... 299).contains(httpResponse.statusCode) {
            //            geminiLogger.error("Error from Google Gemini API: \(httpResponse)")
            // Read and log the error response
            do {
              var errorData = Data()
              for try await byte in result {
                try Task.checkCancellation()
                errorData.append(byte)
              }
              //              if let errorJson = try? JSONSerialization.jsonObject(with: errorData) {
              //                if let prettyData = try? JSONSerialization.data(withJSONObject: errorJson, options: .prettyPrinted), let prettyString = String(data: prettyData, encoding: .utf8) {
              //                  geminiLogger.warning("Gemini API error response: \(prettyString)")
              //                }
              //              } else {
              //                geminiLogger.warning("Could not decode errorData from Gemini API error response")
              //              }

              // Decode the error message
              struct GeminiErrorResponse: Codable {
                struct ErrorDetail: Codable {
                  let message: String
                  let status: String
                  let code: Int
                }

                let error: ErrorDetail
              }

              // Try to decode as a direct object first
              if let errorResponse = try? JSONDecoder().decode(GeminiErrorResponse.self, from: errorData) {
                let errorMessage = errorResponse.error.message
                switch httpResponse.statusCode {
                  case 400: throw AIError.invalidRequest(message: errorMessage)
                  case 403: throw AIError.authentication(message: "Ensure your API key is set correctly and has the right access.")
                  case 404: throw AIError.invalidRequest(message: "Not found: \(errorMessage)")
                  case 429: throw AIError.rateLimit(retryAfter: nil)
                  case 500, 503: throw AIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage, context: nil)
                  case 504: throw AIError.timeout
                  default: throw AIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage, context: nil)
                }
              }
              // Fall back to array format if direct object fails
              else if let errorArray = try? JSONDecoder().decode([GeminiErrorResponse].self, from: errorData),
                      let firstError = errorArray.first
              {
                let errorMessage = firstError.error.message
                switch httpResponse.statusCode {
                  case 400: throw AIError.invalidRequest(message: errorMessage)
                  case 403: throw AIError.authentication(message: "Ensure your API key is set correctly and has the right access.")
                  case 404: throw AIError.invalidRequest(message: "Not found: \(errorMessage)")
                  case 429: throw AIError.rateLimit(retryAfter: nil)
                  case 500, 503: throw AIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage, context: nil)
                  case 504: throw AIError.timeout
                  default: throw AIError.serverError(statusCode: httpResponse.statusCode, message: errorMessage, context: nil)
                }
              }
            } catch let error as AIError {
              throw error
            } catch {
              geminiLogger.error("Failed to read error response: \(error)")
            }

            // Fallback if we couldn't parse the error message
            switch httpResponse.statusCode {
              case 400: throw AIError.invalidRequest(message: "There was a problem with the request body.")
              case 403: throw AIError.authentication(message: "Ensure your API key is set correctly and has the right access.")
              case 404: throw AIError.invalidRequest(message: "The requested resource wasn't found.")
              case 429: throw AIError.rateLimit(retryAfter: nil)
              case 500: throw AIError.serverError(statusCode: 500, message: "An unexpected error occurred. Try reducing your input context, switching to another model temporarily, or retry after a short wait.", context: nil)
              case 503: throw AIError.serverError(statusCode: 503, message: "The service may be temporarily overloaded. Try switching to another model temporarily or retry after a short wait.", context: nil)
              case 504: throw AIError.timeout
              default: throw AIError.serverError(statusCode: httpResponse.statusCode, message: "HTTP error \(httpResponse.statusCode)", context: nil)
            }
          }

          for try await jsonString in SSEParser.dataPayloads(from: result, terminateOnDone: false) {
            do {
              guard let data = jsonString.data(using: .utf8) else {
                geminiLogger.error("Failed to convert SSE payload to UTF-8 data")
                continue
              }
              let jsonObject = try JSONSerialization.jsonObject(with: data) as? [String: any Sendable]

              // Check for promptFeedback (indicates a blocked prompt)
              if let promptFeedback = jsonObject?["promptFeedback"] as? [String: any Sendable],
                 let blockReason = promptFeedback["blockReason"] as? String
              {
                let blockMessage = (promptFeedback["blockReasonMessage"] as? String) ?? "Content was blocked."
                geminiLogger.warning("Prompt blocked: \(blockReason) - \(blockMessage)")

                // Create a GenerateContentResponse to include in the error
                let errorResponse = GenerateContentResponse(
                  candidates: nil,
                  promptFeedback: PromptFeedback(),
                  usageMetadata: nil
                )

                continuation.finish(throwing: GeminiError(
                  message: "Your request was blocked: \(blockMessage)",
                  response: errorResponse
                ))
                return
              }

              // Check for safety ratings in candidates
              if let candidates = jsonObject?["candidates"] as? [[String: any Sendable]],
                 let firstCandidate = candidates.first
              {
                // Check for finish reason first
                if let finishReason = firstCandidate["finishReason"] as? String {
                  if finishReason == "SAFETY" || finishReason == "RECITATION" {
                    let finishMessage = (firstCandidate["finishMessage"] as? String) ?? "Content was blocked due to finish reason \"\(finishReason.lowercased())\"."
                    // Create a GenerateContentResponse to include in the error
                    let candidate = Candidate(
                      content: nil,
                      finishReason: FinishReason(rawValue: finishReason) ?? .safety,
                      safetyRatings: nil,
                      citationMetadata: nil,
                      tokenCount: nil,
                      avgLogprobs: nil,
                      index: 0,
                      groundingMetadata: nil
                    )
                    let errorResponse = GenerateContentResponse(
                      candidates: [candidate],
                      promptFeedback: nil,
                      usageMetadata: nil
                    )
                    continuation.finish(throwing: GeminiError(
                      message: "\(finishMessage)",
                      response: errorResponse
                    ))
                    return
                  } else if finishReason == "MAX_TOKENS" {
                    // MAX_TOKENS is handled later in the parsing flow
                    // Continue to extract text and yield the finish reason at the end
                  } else if finishReason != "FINISH_REASON_UNSPECIFIED", finishReason != "STOP" {
                    // For other finish reasons, just log and finish normally
                    geminiLogger.log("Generation stopped due to finish reason \"\(finishReason.lowercased())\"")
                    continuation.finish()
                    return
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
                    let isThinkingText = if let thought = part["thought"] as? Int { thought == 1 } else { false }
                    continuation.yield(StreamResponse(text: text, thought: isThinkingText, groundingMetadata: nil, toolCall: nil, usageMetadata: nil, finishReason: nil))
                  } else if let functionCall = part["functionCall"] as? [String: any Sendable],
                            let name = functionCall["name"] as? String,
                            let args = functionCall["args"] as? [String: any Sendable]
                  {
                    // Handle function call
                    let parameters = try convertToValue(args)
                    // Extract thoughtSignature if present (required for Gemini tool use)
                    var providerMetadata: [String: String]?
                    if let thoughtSignature = part["thoughtSignature"] as? String {
                      providerMetadata = ["thoughtSignature": thoughtSignature]
                    }
                    let toolCallResponse = GenerationResponse.ToolCall(
                      name: name,
                      id: generateShortId(),
                      parameters: parameters,
                      providerMetadata: providerMetadata
                    )
                    continuation.yield(StreamResponse(text: nil, thought: nil, groundingMetadata: nil, toolCall: toolCallResponse, usageMetadata: nil, finishReason: nil))
                  } else if let executableCodeDict = part["executableCode"] as? [String: any Sendable] {
                    do {
                      let jsonData = try JSONSerialization.data(withJSONObject: executableCodeDict)
                      let executableCode = try JSONDecoder().decode(ExecutableCode.self, from: jsonData)
                      // Format as Markdown code block with language
                      let languageTag = (executableCode.language ?? "").lowercased()
                      if let code = executableCode.code {
                        let markdownText = "\n\n```\(languageTag)\n\(code)\n```\n\n"
                        continuation.yield(StreamResponse(text: markdownText, thought: false, groundingMetadata: nil, toolCall: nil, usageMetadata: nil, finishReason: nil))
                      }
                    } catch {
                      geminiLogger.error("Failed to decode or format ExecutableCode: \(error.localizedDescription)")
                    }
                  } else if let codeExecutionResultDict = part["codeExecutionResult"] as? [String: any Sendable] {
                    do {
                      let jsonData = try JSONSerialization.data(withJSONObject: codeExecutionResultDict)
                      let executionResult = try JSONDecoder().decode(CodeExecutionResult.self, from: jsonData)
                      // Format as Markdown code block (no language tag for result)
                      if let output = executionResult.output {
                        let markdownText = "\n\n```\n\(output)\(output.last == "\n" ? "" : "\n")```\n\n"
                        continuation.yield(StreamResponse(text: markdownText, thought: false, groundingMetadata: nil, toolCall: nil, usageMetadata: nil, finishReason: nil))
                      }
                    } catch {
                      geminiLogger.error("Failed to decode or format CodeExecutionResult: \(error.localizedDescription)")
                    }
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
                continuation.yield(StreamResponse(text: nil, thought: nil, groundingMetadata: groundingMetadata, toolCall: nil, usageMetadata: nil, finishReason: nil))
              }

              // Parse usage metadata
              if let usageMetadataDict = jsonObject?["usageMetadata"] as? [String: any Sendable] {
                do {
                  let metadataData = try JSONSerialization.data(withJSONObject: usageMetadataDict)
                  let decodedUsageMetadata = try JSONDecoder().decode(UsageMetadata.self, from: metadataData)
                  // Yield a StreamResponse that only contains usageMetadata
                  continuation.yield(StreamResponse(text: nil, thought: nil, groundingMetadata: nil, toolCall: nil, usageMetadata: decodedUsageMetadata, finishReason: nil))
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
                continuation.yield(StreamResponse(text: nil, thought: nil, groundingMetadata: nil, toolCall: nil, usageMetadata: nil, finishReason: finishReason))
                // Only finish the stream on terminal finish reasons
                if finishReason == .stop || finishReason == .maxTokens {
                  continuation.finish()
                  return
                }
              }

            } catch {
              geminiLogger.error("Error parsing JSON response: \(error.localizedDescription)")
              // Continue processing rather than failing the whole stream
            }
          }
          // If we've reached the end without a finish reason, finish the stream
          // geminiLogger.warning("Stream ended without finish reason")
          continuation.finish()
        } catch {
          geminiLogger.error("Stream processing error: \(error.localizedDescription)")
          continuation.finish(throwing: error)
        }
      }
    }
  }

  // Extract grounding information from metadata
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
                url: resolvedURL
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

    let result = notes.isEmpty ? nil : notes.joined(separator: "\n")
    return result
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
  public func generateText(
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
  public func streamText(
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
      throw AIError.invalidRequest(message: "API key is not set")
    }
    await MainActor.run {
      isGenerating = true
    }
    let task = Task<GenerationResponse, Error> {
      defer {
        Task { @MainActor in
          isGenerating = false
          currentTask = nil
        }
      }
      var fullReasoningText = ""
      var fullResponseText = ""
      var notesText: String? = nil
      var toolCalls: [GenerationResponse.ToolCall] = []
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
          tools: tools
        )

        for try await chunk in stream {
          try Task.checkCancellation()

          // Handle text chunks
          if let text = chunk.text {
            if let isThinkingText = chunk.thought, isThinkingText {
              fullReasoningText += text
            } else {
              fullResponseText += text
            }
            let fullReasoningTextCopy = fullReasoningText
            let fullResponseTextCopy = fullResponseText
            let notesTextCopy = notesText
            let toolCallsCopy = toolCalls
            await MainActor.run {
              update(.init(texts: .init(reasoning: fullReasoningTextCopy, response: fullResponseTextCopy, notes: notesTextCopy), toolCalls: toolCallsCopy))
            }
          }

          // Handle function calls
          if let toolCall = chunk.toolCall {
            toolCalls.append(toolCall)
            let fullReasoningTextCopy = fullReasoningText
            let fullResponseTextCopy = fullResponseText
            let notesTextCopy = notesText
            let toolCallsCopy = toolCalls
            await MainActor.run {
              update(.init(texts: .init(reasoning: fullReasoningTextCopy, response: fullResponseTextCopy, notes: notesTextCopy), toolCalls: toolCallsCopy))
            }
          }

          // Handle grounding metadata
          if let metadata = chunk.groundingMetadata {
            notesText = await formatGroundingInfo(from: metadata)
            if notesText != nil {
              let fullReasoningTextCopy = fullReasoningText
              let fullResponseTextCopy = fullResponseText
              let notesTextCopy = notesText
              let toolCallsCopy = toolCalls
              await MainActor.run {
                update(.init(texts: .init(reasoning: fullReasoningTextCopy, response: fullResponseTextCopy, notes: notesTextCopy), toolCalls: toolCallsCopy))
              }
            }
          }

          if let metadata = chunk.usageMetadata {
            usageMetadata = metadata // Store the latest (should be the final one)
          }

          if let reason = chunk.finishReason {
            finishReason = reason
          }
        }

        // Build metadata
        let generationFinishReason: GenerationResponse.FinishReason? = if let reason = finishReason {
          switch reason {
            case .stop: .stop
            case .maxTokens: .maxTokens
            case .safety, .recitation: .contentFilter
            case .other: .other
          }
        } else {
          nil
        }

        let metadata = GenerationResponse.Metadata(
          finishReason: generationFinishReason,
          inputTokens: usageMetadata?.promptTokenCount,
          outputTokens: usageMetadata?.candidatesTokenCount,
          totalTokens: usageMetadata?.totalTokenCount,
          cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
          reasoningTokens: usageMetadata?.thoughtsTokenCount
        )

        // Return texts
        return .init(texts: .init(
          reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
          response: fullResponseText.isEmpty ? nil : fullResponseText,
          notes: notesText
        ), toolCalls: toolCalls, metadata: metadata)
      } catch let error as GeminiError {
        // Check if the task was cancelled
        if Task.isCancelled {
          // Build partial metadata
          let partialMetadata = GenerationResponse.Metadata(
            inputTokens: usageMetadata?.promptTokenCount,
            outputTokens: usageMetadata?.candidatesTokenCount,
            totalTokens: usageMetadata?.totalTokenCount,
            cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
            reasoningTokens: usageMetadata?.thoughtsTokenCount
          )
          // Return partial results without throwing an error
          return .init(texts: .init(
            reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
            response: fullResponseText.isEmpty ? nil : fullResponseText,
            notes: notesText
          ), toolCalls: [], metadata: partialMetadata)
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
            reasoningTokens: usageMetadata?.thoughtsTokenCount
          )
          return .init(texts: .init(
            reasoning: fullReasoningText.isEmpty ? nil : fullReasoningText,
            response: fullResponseText.isEmpty ? nil : fullResponseText,
            notes: notesText
          ), toolCalls: toolCalls, metadata: partialMetadata)
        } else {
          throw error
        }
      }
    }
    await MainActor.run {
      currentTask = task
    }
    return try await task.value
  }

  /// Cancels any ongoing generation task.
  @MainActor
  public func stop() {
    currentTask?.cancel()
  }

  private func uploadFile(data: Data, mimeType: String, displayName: String, apiKey: String) async throws -> String {
    let uploadURL = URL(string: "https://generativelanguage.googleapis.com/upload/v1beta/files")!
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
    let metadata = ["file": ["display_name": displayName]]
    request.httpBody = try JSONSerialization.data(withJSONObject: metadata)
//    geminiLogger.log("Initial request URL: \(components.url?.absoluteString ?? "")")
//    geminiLogger.log("Request headers: \(request.allHTTPHeaderFields ?? [:])")
//    if let body = request.httpBody, let bodyString = String(data: body, encoding: .utf8) {
//      geminiLogger.log("Request body: \(bodyString)")
//    }
    let (_, response) = try await session.data(for: request)
//    if let responseString = String(data: initialResponseData, encoding: .utf8) {
//      geminiLogger.log("Initial response data: \(responseString)")
//    }
    guard let httpResponse = response as? HTTPURLResponse else {
      throw AIError.network(underlying: URLError(.badServerResponse))
    }
//    geminiLogger.log("Initial response status code: \(httpResponse.statusCode)")
//    geminiLogger.log("Initial response headers: \(httpResponse.allHeaderFields)")
    guard let uploadUrl = httpResponse.value(forHTTPHeaderField: "X-Goog-Upload-URL") else {
      throw AIError.parsing(message: "Failed to get upload URL from response headers")
    }
    // Upload the actual file
    var uploadRequest = URLRequest(url: URL(string: uploadUrl)!)
    uploadRequest.httpMethod = "POST"
    uploadRequest.setValue("\(data.count)", forHTTPHeaderField: "Content-Length")
    uploadRequest.setValue("0", forHTTPHeaderField: "X-Goog-Upload-Offset")
    uploadRequest.setValue("upload, finalize", forHTTPHeaderField: "X-Goog-Upload-Command")
    uploadRequest.httpBody = data
    let (responseData, _) = try await session.data(for: uploadRequest)
//    if let responseString = String(data: responseData, encoding: .utf8) {
//      geminiLogger.log("Upload response data: \(responseString)")
//    }
//    guard let uploadHttpResponse = uploadResponse as? HTTPURLResponse else {
//      throw GeminiError(message: "Invalid HTTP response for upload")
//    }
//    geminiLogger.log("Upload response status code: \(uploadHttpResponse.statusCode)")
//    geminiLogger.log("Upload response headers: \(uploadHttpResponse.allHeaderFields)")
    do {
      // Response structure for the upload
      struct FileResponse: Codable {
        struct File: Codable {
          let name: String
          let displayName: String
          let mimeType: String
          let sizeBytes: String
          let createTime: String
          let updateTime: String
          let expirationTime: String
          let sha256Hash: String
          let uri: String
          let state: String
          let source: String
          // Create new instance
          init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            displayName = try container.decode(String.self, forKey: .displayName)
            mimeType = try container.decode(String.self, forKey: .mimeType)
            sizeBytes = try container.decode(String.self, forKey: .sizeBytes)
            createTime = try container.decode(String.self, forKey: .createTime)
            updateTime = try container.decode(String.self, forKey: .updateTime)
            expirationTime = try container.decode(String.self, forKey: .expirationTime)
            sha256Hash = try container.decode(String.self, forKey: .sha256Hash)
            uri = try container.decode(String.self, forKey: .uri)
            state = try container.decode(String.self, forKey: .state)
            source = try container.decode(String.self, forKey: .source)
          }

          // Create new instance with updated state
          init(copyFrom other: File, withState newState: String) {
            name = other.name
            displayName = other.displayName
            mimeType = other.mimeType
            sizeBytes = other.sizeBytes
            createTime = other.createTime
            updateTime = other.updateTime
            expirationTime = other.expirationTime
            sha256Hash = other.sha256Hash
            uri = other.uri
            state = newState
            source = other.source
          }
        }

        let file: File
        // Create new instance with updated file
        init(copyFrom _: FileResponse, withFile newFile: File) {
          file = newFile
        }
      }

      // Status check response structure
      struct StatusResponse: Codable {
        let name: String
        let displayName: String
        let mimeType: String
        let sizeBytes: String
        let createTime: String
        let updateTime: String
        let expirationTime: String
        let sha256Hash: String
        let uri: String
        let state: String
        let source: String

        struct ErrorInfo: Codable {
          let code: Int
          let message: String
        }

        let error: ErrorInfo?
      }

      var fileResponse = try JSONDecoder().decode(FileResponse.self, from: responseData)
      let fileUri = fileResponse.file.uri
      // Wait for video processing to complete
      while fileResponse.file.state == "PROCESSING" {
        try Task.checkCancellation()
        try await Task.sleep(for: .seconds(2))
        // Use the full URI from the response
        let checkURL = URL(string: fileUri)!
        components = URLComponents(url: checkURL, resolvingAgainstBaseURL: true)!
        components.queryItems = [URLQueryItem(name: "key", value: apiKey)]
        let (checkData, _) = try await session.data(from: components.url!)
//        if let checkResponseString = String(data: checkData, encoding: .utf8) {
//          geminiLogger.log("Check response data: \(checkResponseString)")
//        }
//        if let checkHttpResponse = checkResponse as? HTTPURLResponse {
//          geminiLogger.log("Check response status code: \(checkHttpResponse.statusCode)")
//        }
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
        // Create new instances with updated state
        let newFile = FileResponse.File(copyFrom: fileResponse.file, withState: statusResponse.state)
        fileResponse = FileResponse(copyFrom: fileResponse, withFile: newFile)
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
      if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: responseData) {
        throw AIError.serverError(statusCode: errorResponse.error.code, message: errorResponse.error.message, context: nil)
      }
      throw AIError.parsing(message: "Failed to decode response: \(error.localizedDescription)")
    }
  }

  // Used to get actual URL in search results references instead of Google tracking URL
  private func resolveRedirectURL(_ url: String) async -> String {
    guard let originalURL = URL(string: url) else { return url }
    // Create a HEAD request to follow redirects without downloading content
    var request = URLRequest(url: originalURL)
    request.httpMethod = "HEAD"
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

  // Extract the blocked URL from ATS error
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
  private func convertToValue(_ dict: [String: any Sendable]) throws -> [String: Value] {
    var result: [String: Value] = [:]
    for (key, value) in dict {
      result[key] = try Value.fromAny(value)
    }
    return result
  }

  /// Converts a raw JSON schema (from rawInputSchema) to Gemini format.
  /// Gemini requires uppercase type values ("STRING" not "string") and
  /// doesn't support "additionalProperties".
  static func convertSchemaForGemini(_ schema: [String: Value]) -> [String: any Sendable] {
    var result: [String: any Sendable] = [:]

    for (key, value) in schema {
      // Skip additionalProperties - Gemini doesn't support this field
      if key == "additionalProperties" {
        continue
      }

      if key == "type" {
        // Convert type values to uppercase
        if case let .string(typeStr) = value {
          result[key] = typeStr.uppercased()
        } else {
          result[key] = convertValueToSendable(value)
        }
      } else if key == "properties" {
        // Recursively convert property schemas
        if case let .object(props) = value {
          var convertedProps: [String: any Sendable] = [:]
          for (propName, propSchema) in props {
            if case let .object(propSchemaDict) = propSchema {
              convertedProps[propName] = convertSchemaForGemini(propSchemaDict)
            } else {
              convertedProps[propName] = convertValueToSendable(propSchema)
            }
          }
          result[key] = convertedProps
        } else {
          result[key] = convertValueToSendable(value)
        }
      } else if key == "items" {
        // Recursively convert array item schema
        if case let .object(itemSchema) = value {
          var convertedItems = convertSchemaForGemini(itemSchema)
          // Ensure items has a type (default to STRING if missing)
          if convertedItems["type"] == nil {
            convertedItems["type"] = "STRING"
          }
          result[key] = convertedItems
        } else {
          result[key] = convertValueToSendable(value)
        }
      } else {
        result[key] = convertValueToSendable(value)
      }
    }

    // Ensure schema has a type if it's missing
    if result["type"] == nil, !schema.isEmpty {
      result["type"] = "STRING"
    }

    return result
  }

  /// Converts a Value to a Sendable type for JSON serialization.
  private static func convertValueToSendable(_ value: Value) -> any Sendable {
    switch value {
      case let .string(s): s
      case let .int(i): i
      case let .double(d): d
      case let .bool(b): b
      case .null: NSNull()
      case let .array(arr): arr.map { convertValueToSendable($0) }
      case let .object(obj): obj.mapValues { convertValueToSendable($0) }
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

    /// A configuration with all features disabled.
    public static let disabled = Configuration()

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
      thinkingLevel: ThinkingLevel? = nil
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
    /// The default thinking level. Uses "high" which is supported by both Gemini 3 Pro and Flash.
    /// Note: "medium" and "minimal" are only supported by Gemini 3 Flash.
    public static let `default`: ThinkingLevel = .high

    /// Matches "no thinking" for most queries. Flash-only.
    case minimal
    /// Minimizes latency and cost. Best for simple instruction following, chat, or high-throughput applications.
    case low
    /// Balanced thinking for most tasks. Flash-only.
    case medium
    /// Maximizes reasoning depth. Supported by both Pro and Flash.
    case high

    /// The raw value identifier.
    public var id: String { rawValue }
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
    public var id: String { rawValue }
  }
}
