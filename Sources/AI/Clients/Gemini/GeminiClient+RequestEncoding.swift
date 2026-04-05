// Copyright © Anthony DePasquale

import Foundation

enum GeminiRequestEncoder {
  static func makeRequest(
    modelsEndpoint: URL,
    modelId: String,
    apiKey: String,
    messages: [Message],
    systemPrompt: String?,
    maxTokens: Int?,
    temperature: Float?,
    configuration: GeminiClient.Configuration,
    tools: [Tool],
    streaming: Bool,
    requestParts: (Message, String) async throws -> [[String: any Sendable]],
  ) async throws -> URLRequest {
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
    request.httpBody = try await JSONSerialization.data(withJSONObject: requestBody(
      messages: messages,
      systemPrompt: systemPrompt,
      apiKey: apiKey,
      maxTokens: maxTokens,
      temperature: temperature,
      configuration: configuration,
      tools: tools,
      requestParts: requestParts,
    ))
    return request
  }

  private static func requestBody(
    messages: [Message],
    systemPrompt: String?,
    apiKey: String,
    maxTokens: Int?,
    temperature: Float?,
    configuration: GeminiClient.Configuration,
    tools: [Tool],
    requestParts: (Message, String) async throws -> [[String: any Sendable]],
  ) async throws -> [String: any Sendable] {
    var processedMessages: [[String: any Sendable]] = []
    var additionalSystemParts: [[String: any Sendable]] = []

    for message in TranscriptReplay.prepare(messages, for: .gemini) {
      switch message.role {
        case .system, .developer:
          additionalSystemParts.append(contentsOf: GeminiClient.systemInstructionParts(for: message))
        case .assistant, .user, .tool:
          let parts = try await requestParts(message, apiKey)
          guard !parts.isEmpty else { continue }
          let role = switch message.role {
            case .assistant: "model"
            case .tool: "user"
            default: message.role.rawValue
          }
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

    var body: [String: any Sendable] = [
      "contents": processedMessages,
      "generationConfig": generationConfig(
        maxTokens: maxTokens,
        temperature: temperature,
        thinkingBudget: configuration.thinkingBudget,
        thinkingLevel: configuration.thinkingLevel,
      ),
      "safetySettings": [
        ["category": "HARM_CATEGORY_HARASSMENT", "threshold": configuration.safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_HATE_SPEECH", "threshold": configuration.safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": configuration.safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": configuration.safetyThreshold.rawValue],
        ["category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": configuration.safetyThreshold.rawValue],
      ],
    ]

    let toolPayload = try toolPayload(
      tools: tools,
      searchGrounding: configuration.searchGrounding,
      webContent: configuration.webContent,
      codeExecution: configuration.codeExecution,
    )
    if !toolPayload.tools.isEmpty {
      body["tools"] = toolPayload.tools
    }
    if let toolConfig = toolPayload.toolConfig {
      body["toolConfig"] = toolConfig
    }

    var systemParts: [[String: any Sendable]] = []
    if let systemPrompt, !systemPrompt.isEmpty {
      systemParts.append(["text": systemPrompt])
    }
    systemParts.append(contentsOf: additionalSystemParts)
    if !systemParts.isEmpty {
      body["systemInstruction"] = ["parts": systemParts]
    }

    return body
  }

  private static func generationConfig(
    maxTokens: Int?,
    temperature: Float?,
    thinkingBudget: Int?,
    thinkingLevel: GeminiClient.ThinkingLevel?,
  ) -> [String: any Sendable] {
    var generationConfig: [String: any Sendable] = [:]

    if let maxTokens {
      generationConfig["maxOutputTokens"] = maxTokens
    }

    if let temperature {
      generationConfig["temperature"] = temperature
    }

    if let thinkingLevel {
      generationConfig["thinkingConfig"] = [
        "thinkingLevel": thinkingLevel.rawValue.uppercased(),
        "includeThoughts": true,
      ] as [String: any Sendable]
      generationConfig["temperature"] = nil
    } else if let thinkingBudget {
      var thinkingConfig: [String: any Sendable] = ["includeThoughts": true]
      if thinkingBudget >= 0 {
        thinkingConfig["thinkingBudget"] = thinkingBudget
      }
      generationConfig["thinkingConfig"] = thinkingConfig
      generationConfig["temperature"] = nil
    }

    return generationConfig
  }

  private static func toolPayload(
    tools: [Tool],
    searchGrounding: Bool,
    webContent: Bool,
    codeExecution: Bool,
  ) throws -> (
    tools: [[String: any Sendable]],
    toolConfig: [String: any Sendable]?,
  ) {
    var toolsArray: [[String: any Sendable]] = []

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

    if codeExecution {
      toolsArray.append([
        "codeExecution": [:] as [String: any Sendable],
      ])
    }

    if !tools.isEmpty {
      let functionDeclarations = try tools.map { function in
        if let baseSchemaBuildErrorMessage = function.baseSchemaBuildErrorMessage {
          throw AIError.invalidRequest(
            message: "Tool '\(function.name)' has an invalid input schema: \(baseSchemaBuildErrorMessage)",
          )
        }
        return [
          "name": function.name,
          "description": function.description,
          "parametersJsonSchema": Value.toSendable(function.rawInputSchema),
        ] as [String: any Sendable]
      }
      toolsArray.append([
        "functionDeclarations": functionDeclarations,
      ])
    }

    guard !toolsArray.isEmpty else {
      return ([], nil)
    }

    var toolConfig: [String: any Sendable] = [:]

    if searchGrounding || webContent || codeExecution {
      toolConfig["includeServerSideToolInvocations"] = true
    }

    if !tools.isEmpty {
      toolConfig["functionCallingConfig"] = [
        "mode": "AUTO",
      ] as [String: any Sendable]
    }

    return (toolsArray, toolConfig.isEmpty ? nil : toolConfig)
  }
}
