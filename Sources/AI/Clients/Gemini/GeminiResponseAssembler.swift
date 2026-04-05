// Copyright © Anthony DePasquale

import Foundation

struct GeminiResponseAssembler {
  private var orderedContent: [Message.Content] = []
  private var toolCallIndicesByID: [String: Int] = [:]
  private var pendingThinkingSignature: String?
  private var notesText: String?
  private var metadataOpaqueBlocks: [OpaqueBlock] = []
  private var usageMetadata: GeminiClient.UsageMetadata?
  private var finishReason: GeminiClient.FinishReason?

  mutating func consume(
    _ chunk: GeminiStreamChunk,
    formatGroundingInfo: (GeminiClient.GroundingMetadata) async -> String?,
  ) async -> Bool {
    if let metadata = chunk.usageMetadata {
      usageMetadata = metadata
    }

    if let reason = chunk.finishReason {
      finishReason = reason
    }

    var didChange = false

    if let text = chunk.text, !text.isEmpty {
      if chunk.thought == true {
        appendThinkingText(text, signature: chunk.thoughtSignature)
      } else {
        appendResponseText(text)
      }
      didChange = true
    } else if let signature = chunk.thoughtSignature {
      updateThinkingSignature(signature)
    }

    if let toolCall = chunk.toolCall {
      upsertToolCall(toolCall)
      didChange = true
    }

    if let opaqueBlock = chunk.opaqueBlock {
      appendOpaqueBlock(opaqueBlock)
      didChange = true
    }

    if let groundingMetadata = chunk.groundingMetadata {
      notesText = await formatGroundingInfo(groundingMetadata)
      if notesText != nil {
        didChange = true
      }
    }

    return didChange
  }

  func response() -> GenerationResponse {
    .init(
      content: GeminiClient.finalizedContent(
        orderedContent: orderedContent,
        notesText: notesText,
        metadataOpaqueBlocks: metadataOpaqueBlocks,
      ),
      metadata: metadata(),
    )
  }

  func partialResponse() -> GenerationResponse {
    .init(
      content: GeminiClient.finalizedContent(
        orderedContent: orderedContent,
        notesText: notesText,
        metadataOpaqueBlocks: metadataOpaqueBlocks,
      ),
      metadata: GenerationResponse.Metadata(
        inputTokens: usageMetadata?.promptTokenCount,
        outputTokens: usageMetadata?.candidatesTokenCount,
        totalTokens: usageMetadata?.totalTokenCount,
        cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
        reasoningTokens: usageMetadata?.thoughtsTokenCount,
      ),
    )
  }

  private func metadata() -> GenerationResponse.Metadata {
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

    let effectiveFinishReason = if orderedContent.contains(where: {
      if case .toolCall = $0 { true } else { false }
    }) {
      GenerationResponse.FinishReason.toolUse
    } else {
      generationFinishReason
    }

    return GenerationResponse.Metadata(
      finishReason: effectiveFinishReason,
      inputTokens: usageMetadata?.promptTokenCount,
      outputTokens: usageMetadata?.candidatesTokenCount,
      totalTokens: usageMetadata?.totalTokenCount,
      cacheReadInputTokens: usageMetadata?.cachedContentTokenCount,
      reasoningTokens: usageMetadata?.thoughtsTokenCount,
    )
  }

  private mutating func updateLastGeminiThinking(_ transform: (OpaqueBlock) -> OpaqueBlock) {
    guard let index = orderedContent.indices.reversed().first(where: { index in
      guard case let .providerOpaque(block) = orderedContent[index] else { return false }
      return block.provider == "gemini" && block.type == "thinking"
    }) else {
      return
    }
    guard case let .providerOpaque(block) = orderedContent[index] else { return }
    orderedContent[index] = .providerOpaque(transform(block))
  }

  private mutating func appendThinkingText(_ text: String, signature: String? = nil) {
    let resolvedSignature = signature ?? pendingThinkingSignature
    pendingThinkingSignature = nil

    if let lastIndex = orderedContent.indices.last,
       case let .providerOpaque(block) = orderedContent[lastIndex],
       block.provider == "gemini",
       block.type == "thinking"
    {
      orderedContent[lastIndex] = .providerOpaque(OpaqueBlock(
        provider: block.provider,
        type: block.type,
        content: (block.content ?? "") + text,
        signature: resolvedSignature ?? block.signature,
        data: block.data,
        isResponseContent: block.isResponseContent,
      ))
    } else {
      orderedContent.append(.providerOpaque(OpaqueBlock(
        provider: "gemini",
        type: "thinking",
        content: text,
        signature: resolvedSignature,
      )))
    }
  }

  private mutating func updateThinkingSignature(_ signature: String) {
    if orderedContent.contains(where: { item in
      guard case let .providerOpaque(block) = item else { return false }
      return block.provider == "gemini" && block.type == "thinking"
    }) {
      updateLastGeminiThinking { block in
        OpaqueBlock(
          provider: block.provider,
          type: block.type,
          content: block.content,
          signature: signature,
          data: block.data,
          isResponseContent: block.isResponseContent,
        )
      }
    } else {
      pendingThinkingSignature = signature
    }
  }

  private mutating func appendResponseText(_ text: String) {
    if let lastIndex = orderedContent.indices.last,
       case let .text(existingText) = orderedContent[lastIndex]
    {
      orderedContent[lastIndex] = .text(existingText + text)
    } else {
      orderedContent.append(.text(text))
    }
  }

  private mutating func upsertToolCall(_ toolCall: ToolCall) {
    if let existingIndex = toolCallIndicesByID[toolCall.id] {
      orderedContent[existingIndex] = .toolCall(toolCall)
    } else {
      toolCallIndicesByID[toolCall.id] = orderedContent.count
      orderedContent.append(.toolCall(toolCall))
    }
  }

  private mutating func appendOpaqueBlock(_ opaqueBlock: OpaqueBlock) {
    if opaqueBlock.provider == "gemini", opaqueBlock.type == "urlContextMetadata" {
      metadataOpaqueBlocks.append(opaqueBlock)
    } else {
      orderedContent.append(.providerOpaque(opaqueBlock))
    }
  }
}
