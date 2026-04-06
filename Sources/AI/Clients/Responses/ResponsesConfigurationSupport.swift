// Copyright © Anthony DePasquale

import Foundation

#if canImport(os)
import os
#endif

struct ResolvedResponsesBackend: Equatable {
  let providerFamily: ResponsesProvider
  let isCustomEndpoint: Bool

  var requiresEncryptedReasoningCapture: Bool {
    providerFamily == .openAI
  }
}

private enum TopLevelResponsesSessionOverride {
  static let storage = OSAllocatedUnfairLock(initialState: URLSession?.none)
}

func makeTopLevelResponsesClient(endpoint: URL) -> ResponsesClient {
  let session = TopLevelResponsesSessionOverride.storage.withLock { $0 } ?? .shared
  return ResponsesClient(endpoint: endpoint, session: session)
}

func setTopLevelResponsesSessionOverride(_ session: URLSession?) {
  TopLevelResponsesSessionOverride.storage.withLock { $0 = session }
}

func defaultResponsesConfiguration(
  webSearch: Bool,
  endpoint: URL,
  provider: ResponsesProvider?,
) throws -> ResponsesClient.Configuration {
  let backend: ResolvedResponsesBackend? = if webSearch {
    try resolveResponsesBackendRequiringProvider(
      for: endpoint,
      provider: provider,
      purpose: "webSearch",
    )
  } else {
    try resolveResponsesBackend(for: endpoint, provider: provider)
  }
  let serverSideTools: [ResponsesClient.ServerSideTool]
  if webSearch {
    guard let backend else {
      preconditionFailure("Responses backend should be resolved when webSearch is enabled")
    }
    serverSideTools = responsesWebSearchTools(for: backend)
  } else {
    serverSideTools = []
  }

  return ResponsesClient.Configuration(
    serverSideTools: serverSideTools,
    provider: backend?.providerFamily,
  )
}

func resolveResponsesBackend(
  for endpoint: URL,
  provider: ResponsesProvider?,
) throws -> ResolvedResponsesBackend? {
  if let inferredProvider = inferredResponsesProvider(for: endpoint) {
    if let provider, provider != inferredProvider {
      throw AIError.invalidRequest(message:
        "responsesProvider \(provider.argumentName) conflicts with the built-in Responses endpoint for " +
          "\(inferredProvider.argumentName). Omit responsesProvider for known OpenAI/xAI endpoints, " +
          "or set it to \(inferredProvider.argumentName).")
    }
    return ResolvedResponsesBackend(providerFamily: inferredProvider, isCustomEndpoint: false)
  }

  guard let provider else {
    return nil
  }
  return ResolvedResponsesBackend(providerFamily: provider, isCustomEndpoint: true)
}

func resolveResponsesBackendRequiringProvider(
  for endpoint: URL,
  provider: ResponsesProvider?,
  purpose: String,
) throws -> ResolvedResponsesBackend {
  guard let backend = try resolveResponsesBackend(for: endpoint, provider: provider) else {
    throw AIError.invalidRequest(message:
      "Custom Responses endpoints require an explicit responsesProvider when using \(purpose). " +
        "Pass `.openAI` or `.xAI`, or provide an explicit `.responses` configuration.")
  }
  return backend
}

func responsesWebSearchTools(
  for backend: ResolvedResponsesBackend,
) -> [ResponsesClient.ServerSideTool] {
  switch backend.providerFamily {
    case .openAI:
      [ResponsesClient.ServerSideTool.OpenAI.webSearch(contextSize: .medium)]
    case .xAI:
      [ResponsesClient.ServerSideTool.xAI.webSearch()]
  }
}

private func inferredResponsesProvider(for endpoint: URL) -> ResponsesProvider? {
  switch endpoint.host {
    case ResponsesClient.Endpoint.openAI.url.host:
      .openAI
    case ResponsesClient.Endpoint.xAI.url.host:
      .xAI
    default:
      nil
  }
}

private extension ResponsesProvider {
  var argumentName: String {
    switch self {
      case .openAI:
        "`.openAI`"
      case .xAI:
        "`.xAI`"
    }
  }
}
