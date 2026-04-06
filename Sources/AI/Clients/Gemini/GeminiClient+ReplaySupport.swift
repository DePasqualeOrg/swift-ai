// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isGeminiThinking: Bool {
    provider == Self.ProviderID.gemini && type == Self.GeminiType.thinking
  }

  var isGeminiRoundTrippablePart: Bool {
    provider == Self.ProviderID.gemini
      && (type == Self.GeminiType.executableCode
        || type == Self.GeminiType.codeExecutionResult
        || type == Self.GeminiType.toolCall
        || type == Self.GeminiType.toolResponse)
  }

  var isGeminiURLContextMetadata: Bool {
    provider == Self.ProviderID.gemini && type == Self.GeminiType.urlContextMetadata
  }
}
