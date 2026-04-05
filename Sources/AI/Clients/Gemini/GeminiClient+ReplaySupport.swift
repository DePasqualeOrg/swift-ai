// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isGeminiThinking: Bool {
    provider == "gemini" && type == "thinking"
  }

  var isGeminiRoundTrippablePart: Bool {
    provider == "gemini"
      && (type == "executableCode"
        || type == "codeExecutionResult"
        || type == "toolCall"
        || type == "toolResponse")
  }

  var isGeminiURLContextMetadata: Bool {
    provider == "gemini" && type == "urlContextMetadata"
  }
}
