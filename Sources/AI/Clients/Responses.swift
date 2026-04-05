// Copyright © Anthony DePasquale

import Foundation
import Observation
import os.log
import SSE

/// A client for the OpenAI Responses API.
///
/// The Responses API is OpenAI's newer API format that supports built-in tools
/// like web search and file search, as well as custom function tools. Also works
/// with xAI (Grok) models.
///
/// ## Example
///
/// ```swift
/// let client = ResponsesClient()
/// let response = try await client.generateText(
///   modelId: "gpt-4o",
///   prompt: "Hello!",
///   apiKey: "your-api-key"
/// )
/// print(response.content)
/// ```
@Observable
public final class ResponsesClient: APIClient, Sendable {
  public static let supportedResultTypes: Set<ToolResult.ValueType> = [.text, .image, .file]

  /// Predefined API endpoints for the Responses API.
  public enum Endpoint {
    /// OpenAI's Responses API endpoint.
    case openAI
    /// xAI's Responses API endpoint.
    case xAI

    /// The URL for this endpoint.
    public var url: URL {
      switch self {
        case .openAI: URL(string: "https://api.openai.com/v1/responses")!
        case .xAI: URL(string: "https://api.x.ai/v1/responses")!
      }
    }
  }

  /// The API endpoint URL used by this client.
  public let endpoint: URL

  @MainActor public private(set) var isGenerating: Bool = false
  @MainActor private var currentTask: Task<GenerationResponse, Error>?
  /// The ID of the currently active background response, if any
  /// This can be used to manually interrupt and resume background streams
  @MainActor public private(set) var activeBackgroundResponseId: String?
  /// The API key associated with the active background response, used for authenticated cancellation
  @MainActor private var activeBackgroundResponseApiKey: String?

  private let session: URLSession

  /// Creates a new Responses client with a predefined endpoint.
  ///
  /// - Parameters:
  ///   - endpoint: The API endpoint to use (OpenAI or xAI).
  ///   - session: URLSession to use for requests.
  public init(endpoint: Endpoint = .openAI, session: URLSession = .shared) {
    self.endpoint = endpoint.url
    self.session = session
  }

  /// Creates a new Responses client with a custom endpoint URL.
  ///
  /// - Parameters:
  ///   - endpoint: Custom endpoint URL for the Responses API.
  ///   - session: URLSession to use for requests.
  public init(endpoint: URL, session: URLSession = .shared) {
    self.endpoint = endpoint
    self.session = session
  }

  // MARK: - Content Type Constants

  private enum ContentType {
    static let inputText = "input_text"
    static let inputImage = "input_image"
    static let inputFile = "input_file"
    static let outputText = "output_text"
    static let message = "message"
    static let functionCall = "function_call"
    static let functionCallOutput = "function_call_output"
  }

  /// Composite key for streaming content: (outputIndex, contentIndex).
  /// Content parts within the same output item (e.g., text + refusal in one message)
  /// have different contentIndex values and must not overwrite each other.
  private struct ContentKey: Hashable, Comparable {
    let outputIndex: Int
    let contentIndex: Int

    static func < (lhs: ContentKey, rhs: ContentKey) -> Bool {
      if lhs.outputIndex != rhs.outputIndex {
        return lhs.outputIndex < rhs.outputIndex
      }
      return lhs.contentIndex < rhs.contentIndex
    }
  }

  private struct StreamingResponseState {
    var indexedContent: [ContentKey: Message.Content] = [:]
    var fallbackContent: [Message.Content] = []
    var toolCallArgumentBuffers: [String: String] = [:]
    /// Maps item IDs to fallback content indices for tool calls without output_index.
    var itemIdToFallbackIndex: [String: Int] = [:]
    /// Separate buffer for reasoning summary deltas, keyed by output index.
    /// Used as a fallback when no full reasoning text is available at that index.
    var summaryContent: [Int: String] = [:]
    /// Tracks the next contentIndex to assign per outputIndex when events omit content_index.
    private var nextContentIndex: [Int: Int] = [:]

    /// Returns a ContentKey, using the provided contentIndex or auto-assigning from a counter.
    private mutating func key(outputIndex: Int, contentIndex: Int?) -> ContentKey {
      if let contentIndex {
        // Update the counter so future auto-assignments don't collide
        nextContentIndex[outputIndex] = max(nextContentIndex[outputIndex, default: 0], contentIndex + 1)
        return ContentKey(outputIndex: outputIndex, contentIndex: contentIndex)
      }
      // For events without content_index (reasoning, tool calls), use 0
      return ContentKey(outputIndex: outputIndex, contentIndex: 0)
    }

    var content: [Message.Content] {
      // For output indices that have summary but no reasoning text, use the summary
      var merged = indexedContent
      for (index, summary) in summaryContent {
        let key = ContentKey(outputIndex: index, contentIndex: 0)
        if merged[key] == nil {
          merged[key] = .thinking(text: summary, signature: nil)
        }
      }
      return merged.keys.sorted().compactMap { merged[$0] } + fallbackContent
    }

    mutating func appendTextDelta(_ delta: String, outputIndex: Int?, contentIndex: Int?) {
      append(delta: delta, outputIndex: outputIndex, contentIndex: contentIndex, as: { .text($0) })
    }

    mutating func setFinalizedText(_ text: String, outputIndex: Int?, contentIndex: Int?) {
      guard let outputIndex else {
        appendFallback(.text(text))
        return
      }
      let k = key(outputIndex: outputIndex, contentIndex: contentIndex)
      indexedContent[k] = .text(text)
    }

    mutating func appendRefusalDelta(_ delta: String, outputIndex: Int?, contentIndex: Int?) {
      append(delta: delta, outputIndex: outputIndex, contentIndex: contentIndex, as: {
        .providerOpaque(OpaqueBlock(provider: "openai-responses", type: "refusal", content: $0, isResponseContent: true))
      })
    }

    mutating func setFinalizedRefusal(_ text: String, outputIndex: Int?, contentIndex: Int?) {
      let block = Message.Content.providerOpaque(OpaqueBlock(
        provider: "openai-responses", type: "refusal", content: text, isResponseContent: true,
      ))
      guard let outputIndex else {
        appendFallback(block)
        return
      }
      let k = key(outputIndex: outputIndex, contentIndex: contentIndex)
      indexedContent[k] = block
    }

    mutating func appendReasoningDelta(_ delta: String, outputIndex: Int?) {
      append(delta: delta, outputIndex: outputIndex, contentIndex: nil, as: { .thinking(text: $0, signature: nil) })
    }

    mutating func appendSummaryDelta(_ delta: String, outputIndex: Int?) {
      guard let outputIndex else {
        appendFallback(.thinking(text: delta, signature: nil))
        return
      }
      summaryContent[outputIndex, default: ""] += delta
    }

    mutating func setSummaryText(_ text: String, outputIndex: Int?) {
      guard let outputIndex else {
        appendFallback(.thinking(text: text, signature: nil))
        return
      }
      summaryContent[outputIndex] = text
    }

    mutating func setToolCall(_ toolCall: ToolCall, outputIndex: Int?, itemId: String? = nil) {
      guard let outputIndex else {
        let index = fallbackContent.count
        appendFallback(.toolCall(toolCall))
        if let itemId {
          itemIdToFallbackIndex[itemId] = index
        }
        // Also map by call_id (toolCall.id) since subsequent function_call_arguments.* events
        // use item_id which may correspond to call_id rather than the output item's id.
        itemIdToFallbackIndex[toolCall.id] = index
        return
      }
      let k = key(outputIndex: outputIndex, contentIndex: nil)
      indexedContent[k] = .toolCall(toolCall)
      toolCallArgumentBuffers[String(outputIndex)] = ""
    }

    mutating func appendToolCallArgumentsDelta(_ delta: String, outputIndex: Int?, itemId: String?) {
      let bufferKey = outputIndex.map(String.init) ?? itemId ?? "_fallback"
      let existingArgsString = toolCallArgumentBuffers[bufferKey] ?? ""
      let newArgsString = existingArgsString + delta
      toolCallArgumentBuffers[bufferKey] = newArgsString

      if let outputIndex {
        let k = key(outputIndex: outputIndex, contentIndex: nil)
        if case let .toolCall(currentToolCall)? = indexedContent[k] {
          guard let argsData = newArgsString.data(using: .utf8),
                let partialArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
          else { return }
          var updatedToolCall = currentToolCall
          updatedToolCall.parameters = partialArgs
          indexedContent[k] = .toolCall(updatedToolCall)
        }
      } else {
        let matchIndex = itemId.flatMap { itemIdToFallbackIndex[$0] }
          ?? fallbackContent.lastIndex(where: { if case .toolCall = $0 { true } else { false } })
        guard let matchIndex,
              let argsData = newArgsString.data(using: .utf8),
              let partialArgs = try? JSONDecoder().decode([String: Value].self, from: argsData)
        else { return }
        if case let .toolCall(currentToolCall) = fallbackContent[matchIndex] {
          var updatedToolCall = currentToolCall
          updatedToolCall.parameters = partialArgs
          fallbackContent[matchIndex] = .toolCall(updatedToolCall)
        }
      }
    }

    mutating func completeToolCallArguments(_ argumentsString: String, outputIndex: Int?, itemId: String?) {
      let bufferKey = outputIndex.map(String.init) ?? itemId ?? "_fallback"
      toolCallArgumentBuffers.removeValue(forKey: bufferKey)

      var updatedToolCall: ToolCall?
      var matchedFallbackIndex: Int?
      if let outputIndex {
        let k = key(outputIndex: outputIndex, contentIndex: nil)
        if case let .toolCall(currentToolCall)? = indexedContent[k] {
          updatedToolCall = currentToolCall
        }
      } else {
        let idx = itemId.flatMap { itemIdToFallbackIndex[$0] }
          ?? fallbackContent.lastIndex(where: { if case .toolCall = $0 { true } else { false } })
        if let idx, case let .toolCall(currentToolCall) = fallbackContent[idx] {
          updatedToolCall = currentToolCall
          matchedFallbackIndex = idx
        }
      }

      guard var toolCall = updatedToolCall else { return }
      if let argumentsData = argumentsString.data(using: .utf8),
         let parsedArguments = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
      {
        toolCall.parameters = parsedArguments
      } else {
        openAIResponsesLogger.error("Failed to parse final function call arguments: \(argumentsString)")
        toolCall.parameters = ["_parseError": .string("Failed to parse arguments JSON"), "_rawArguments": .string(argumentsString)]
      }

      if let outputIndex {
        let k = key(outputIndex: outputIndex, contentIndex: nil)
        indexedContent[k] = .toolCall(toolCall)
      } else if let idx = matchedFallbackIndex {
        fallbackContent[idx] = .toolCall(toolCall)
      }
    }

    private mutating func append(
      delta: String,
      outputIndex: Int?,
      contentIndex: Int?,
      as createBlock: (String) -> Message.Content,
    ) {
      append(delta: delta, outputIndex: outputIndex, contentIndex: contentIndex) { existing in
        createBlock(existing + delta)
      } create: {
        createBlock(delta)
      }
    }

    private mutating func append(
      delta _: String,
      outputIndex: Int?,
      contentIndex: Int?,
      update: (String) -> Message.Content,
      create: () -> Message.Content,
    ) {
      guard let outputIndex else {
        appendFallback(create())
        return
      }

      let k = key(outputIndex: outputIndex, contentIndex: contentIndex)
      switch indexedContent[k] {
        case let .text(existingText)?:
          indexedContent[k] = update(existingText)
        case let .thinking(text: existingText, signature: nil)?:
          indexedContent[k] = update(existingText)
        case let .providerOpaque(block)? where block.content != nil:
          indexedContent[k] = update(block.content ?? "")
        default:
          indexedContent[k] = create()
      }
    }

    private mutating func appendFallback(_ block: Message.Content) {
      guard let lastBlock = fallbackContent.last else {
        fallbackContent.append(block)
        return
      }

      switch (lastBlock, block) {
        case let (.text(existing), .text(delta)):
          fallbackContent[fallbackContent.count - 1] = .text(existing + delta)
        case let (.thinking(text: existing, signature: existingSignature), .thinking(text: delta, signature: nil)):
          fallbackContent[fallbackContent.count - 1] = .thinking(text: existing + delta, signature: existingSignature)
        default:
          fallbackContent.append(block)
      }
    }
  }

  private struct AccumulatedResponseSnapshot {
    private var raw: [String: Value]
    private var outputs: [[String: Value]]
    private var outputIndexByItemID: [String: Int] = [:]
    private var outputIndexByCallID: [String: Int] = [:]
    private var implicitMessageOutputIndex: Int?
    private var implicitReasoningOutputIndex: Int?

    init(_ response: ResponseObject) {
      raw = response.raw
      outputs = response.output?.map(\.raw) ?? []
      rebuildIndexes()
    }

    mutating func apply(_ event: StreamEvent) {
      guard let eventType = event.type else { return }

      switch eventType {
        case StreamEventType.outputItemAdded:
          if let item = event.item?.raw {
            insertOutputItem(item, outputIndex: event.outputIndex)
          }
        case StreamEventType.contentPartAdded:
          guard let part = event.part?.raw else { break }
          let outputType = part["type"]?.stringValue == "reasoning_text" ? OutputItemType.reasoning : OutputItemType.message
          let resolvedOutputIndex = resolveOutputIndex(explicitIndex: event.outputIndex, defaultType: outputType)
          insertContentPart(part, atOutputIndex: resolvedOutputIndex, contentIndex: event.contentIndex)
        case StreamEventType.outputTextDelta:
          if let delta = event.delta {
            appendDelta(
              delta,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              defaultOutputType: OutputItemType.message,
              partType: OutputItemType.outputText,
              fieldName: "text",
            )
          }
        case StreamEventType.refusalDelta:
          if let delta = event.delta {
            appendDelta(
              delta,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              defaultOutputType: OutputItemType.message,
              partType: OutputItemType.refusal,
              fieldName: "refusal",
            )
          }
        case StreamEventType.refusalDone:
          if let refusal = event.refusal {
            setContentField(
              refusal,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              defaultOutputType: OutputItemType.message,
              partType: OutputItemType.refusal,
              fieldName: "refusal",
            )
          }
        case StreamEventType.reasoningTextDelta, StreamEventType.reasoningDelta:
          if let delta = event.delta {
            appendDelta(
              delta,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              defaultOutputType: OutputItemType.reasoning,
              partType: "reasoning_text",
              fieldName: "text",
            )
          }
        case StreamEventType.reasoningSummaryDelta:
          if let delta = event.delta {
            appendSummaryDelta(delta, outputIndex: event.outputIndex, summaryIndex: event.summaryIndex)
          }
        case StreamEventType.reasoningSummaryDone:
          if let text = event.text {
            setSummaryText(text, outputIndex: event.outputIndex, summaryIndex: event.summaryIndex)
          }
        case StreamEventType.functionCallArgumentsDelta:
          if let delta = event.delta {
            appendFunctionCallArgumentsDelta(delta, outputIndex: event.outputIndex, itemID: event.itemId)
          }
        case StreamEventType.functionCallArgumentsDone:
          if let arguments = event.arguments {
            completeFunctionCallArguments(arguments, outputIndex: event.outputIndex, itemID: event.itemId)
          }
        case StreamEventType.completed, StreamEventType.failed, StreamEventType.incomplete:
          if let response = event.response {
            self = Self(response)
          }
        default:
          break
      }
    }

    func finalize() -> GenerationResponse {
      var finalRaw = raw
      if !outputs.isEmpty || raw["output"] != nil {
        finalRaw["output"] = .array(outputs.map(Value.object))
      }
      return ResponseObject(raw: finalRaw).toGenerationResponse()
    }

    private mutating func rebuildIndexes() {
      outputIndexByItemID = [:]
      outputIndexByCallID = [:]
      implicitMessageOutputIndex = nil
      implicitReasoningOutputIndex = nil

      for (index, item) in outputs.enumerated() {
        if let itemID = item["id"]?.stringValue {
          outputIndexByItemID[itemID] = index
        }
        if let callID = item["call_id"]?.stringValue {
          outputIndexByCallID[callID] = index
        }
        switch item["type"]?.stringValue {
          case OutputItemType.message where implicitMessageOutputIndex == nil:
            implicitMessageOutputIndex = index
          case OutputItemType.reasoning where implicitReasoningOutputIndex == nil:
            implicitReasoningOutputIndex = index
          default:
            break
        }
      }
    }

    private mutating func insertOutputItem(_ item: [String: Value], outputIndex: Int?) {
      if let outputIndex {
        if outputIndex < outputs.count {
          outputs[outputIndex] = Self.mergeOutputItem(existing: outputs[outputIndex], with: item)
        } else {
          while outputs.count < outputIndex {
            outputs.append(Self.syntheticOutputItem(ofType: OutputItemType.message))
          }
          outputs.append(item)
        }
      } else if let existingIndex = existingOutputIndex(for: item) {
        outputs[existingIndex] = Self.mergeOutputItem(existing: outputs[existingIndex], with: item)
      } else {
        outputs.append(item)
      }

      rebuildIndexes()
    }

    private func existingOutputIndex(for item: [String: Value]) -> Int? {
      if let itemID = item["id"]?.stringValue, let existingIndex = outputIndexByItemID[itemID] {
        return existingIndex
      }
      if let callID = item["call_id"]?.stringValue, let existingIndex = outputIndexByCallID[callID] {
        return existingIndex
      }
      return nil
    }

    private mutating func resolveOutputIndex(explicitIndex: Int?, defaultType: String) -> Int {
      if let explicitIndex {
        ensureOutputItem(at: explicitIndex, type: defaultType)
        return explicitIndex
      }

      switch defaultType {
        case OutputItemType.reasoning:
          if let existingIndex = implicitReasoningOutputIndex {
            return existingIndex
          }
          let newIndex = appendOutputItem(Self.syntheticOutputItem(ofType: OutputItemType.reasoning))
          implicitReasoningOutputIndex = newIndex
          return newIndex
        default:
          if let existingIndex = implicitMessageOutputIndex {
            return existingIndex
          }
          let newIndex = appendOutputItem(Self.syntheticOutputItem(ofType: OutputItemType.message))
          implicitMessageOutputIndex = newIndex
          return newIndex
      }
    }

    private mutating func ensureOutputItem(at index: Int, type: String) {
      if index < outputs.count {
        if outputs[index]["type"] == nil {
          outputs[index] = Self.mergeOutputItem(existing: outputs[index], with: Self.syntheticOutputItem(ofType: type))
          rebuildIndexes()
        }
        return
      }

      while outputs.count <= index {
        let placeholderType = outputs.count == index ? type : OutputItemType.message
        outputs.append(Self.syntheticOutputItem(ofType: placeholderType))
      }
      rebuildIndexes()
    }

    @discardableResult
    private mutating func appendOutputItem(_ item: [String: Value]) -> Int {
      outputs.append(item)
      let index = outputs.count - 1
      rebuildIndexes()
      return index
    }

    private mutating func insertContentPart(_ part: [String: Value], atOutputIndex outputIndex: Int, contentIndex: Int?) {
      guard outputs.indices.contains(outputIndex) else { return }

      var output = outputs[outputIndex]
      var content = output["content"]?.arrayValue ?? []
      let resolvedContentIndex = contentIndex ?? content.count

      while content.count < resolvedContentIndex {
        content.append(.object(Self.syntheticContentPart(ofType: part["type"]?.stringValue ?? OutputItemType.outputText)))
      }

      if resolvedContentIndex < content.count, let existingPart = content[resolvedContentIndex].objectValue {
        content[resolvedContentIndex] = .object(Self.mergeContentPart(existing: existingPart, with: part))
      } else {
        content.append(.object(part))
      }

      output["content"] = .array(content)
      outputs[outputIndex] = output
    }

    private mutating func appendDelta(
      _ delta: String,
      outputIndex: Int?,
      contentIndex: Int?,
      defaultOutputType: String,
      partType: String,
      fieldName: String,
    ) {
      guard !delta.isEmpty else { return }

      let resolvedOutputIndex = resolveOutputIndex(explicitIndex: outputIndex, defaultType: defaultOutputType)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      var content = output["content"]?.arrayValue ?? []
      let resolvedContentIndex = Self.resolveContentIndex(in: content, requested: contentIndex, preferredType: partType)

      while content.count <= resolvedContentIndex {
        content.append(.object(Self.syntheticContentPart(ofType: partType)))
      }

      var part = content[resolvedContentIndex].objectValue ?? Self.syntheticContentPart(ofType: partType)
      if part["type"] == nil {
        part["type"] = .string(partType)
      }
      let existingValue = part[fieldName]?.stringValue ?? ""
      part[fieldName] = .string(existingValue + delta)

      content[resolvedContentIndex] = .object(part)
      output["content"] = .array(content)
      outputs[resolvedOutputIndex] = output
    }

    private mutating func setContentField(
      _ value: String,
      outputIndex: Int?,
      contentIndex: Int?,
      defaultOutputType: String,
      partType: String,
      fieldName: String,
    ) {
      let resolvedOutputIndex = resolveOutputIndex(explicitIndex: outputIndex, defaultType: defaultOutputType)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      var content = output["content"]?.arrayValue ?? []
      let resolvedContentIndex = Self.resolveContentIndex(in: content, requested: contentIndex, preferredType: partType)

      while content.count <= resolvedContentIndex {
        content.append(.object(Self.syntheticContentPart(ofType: partType)))
      }

      var part = content[resolvedContentIndex].objectValue ?? Self.syntheticContentPart(ofType: partType)
      part["type"] = .string(partType)
      part[fieldName] = .string(value)

      content[resolvedContentIndex] = .object(part)
      output["content"] = .array(content)
      outputs[resolvedOutputIndex] = output
    }

    private mutating func appendSummaryDelta(_ delta: String, outputIndex: Int?, summaryIndex: Int?) {
      guard !delta.isEmpty else { return }

      let resolvedOutputIndex = resolveOutputIndex(explicitIndex: outputIndex, defaultType: OutputItemType.reasoning)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      var summary = output["summary"]?.arrayValue ?? []
      let resolvedSummaryIndex = Self.resolveSummaryIndex(in: summary, requested: summaryIndex)
      while summary.count <= resolvedSummaryIndex {
        summary.append(.object(Self.syntheticSummaryItem()))
      }

      var item = summary[resolvedSummaryIndex].objectValue ?? Self.syntheticSummaryItem()
      let existingText = item["text"]?.stringValue ?? ""
      item["text"] = .string(existingText + delta)
      summary[resolvedSummaryIndex] = .object(item)

      output["summary"] = .array(summary)
      outputs[resolvedOutputIndex] = output
    }

    private mutating func setSummaryText(_ text: String, outputIndex: Int?, summaryIndex: Int?) {
      let resolvedOutputIndex = resolveOutputIndex(explicitIndex: outputIndex, defaultType: OutputItemType.reasoning)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      var summary = output["summary"]?.arrayValue ?? []
      let resolvedSummaryIndex = Self.resolveSummaryIndex(in: summary, requested: summaryIndex)
      while summary.count <= resolvedSummaryIndex {
        summary.append(.object(Self.syntheticSummaryItem()))
      }

      var item = summary[resolvedSummaryIndex].objectValue ?? Self.syntheticSummaryItem()
      item["text"] = .string(text)
      summary[resolvedSummaryIndex] = .object(item)

      output["summary"] = .array(summary)
      outputs[resolvedOutputIndex] = output
    }

    private mutating func appendFunctionCallArgumentsDelta(_ delta: String, outputIndex: Int?, itemID: String?) {
      guard !delta.isEmpty else { return }

      let resolvedOutputIndex = resolveToolCallOutputIndex(explicitIndex: outputIndex, itemID: itemID)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      let existingArguments = output["arguments"]?.stringValue ?? ""
      output["arguments"] = .string(existingArguments + delta)
      if output["type"] == nil {
        output["type"] = .string(OutputItemType.functionCall)
      }
      if let itemID, output["call_id"] == nil {
        output["call_id"] = .string(itemID)
      }

      outputs[resolvedOutputIndex] = output
    }

    private mutating func completeFunctionCallArguments(_ arguments: String, outputIndex: Int?, itemID: String?) {
      let resolvedOutputIndex = resolveToolCallOutputIndex(explicitIndex: outputIndex, itemID: itemID)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      output["type"] = .string(OutputItemType.functionCall)
      output["arguments"] = .string(arguments)
      if let itemID, output["call_id"] == nil {
        output["call_id"] = .string(itemID)
      }

      outputs[resolvedOutputIndex] = output
    }

    private mutating func resolveToolCallOutputIndex(explicitIndex: Int?, itemID: String?) -> Int {
      if let explicitIndex {
        ensureOutputItem(at: explicitIndex, type: OutputItemType.functionCall)
        return explicitIndex
      }

      if let itemID, let existingIndex = outputIndexByItemID[itemID] ?? outputIndexByCallID[itemID] {
        return existingIndex
      }

      var item = Self.syntheticOutputItem(ofType: OutputItemType.functionCall)
      if let itemID {
        item["call_id"] = .string(itemID)
      }
      return appendOutputItem(item)
    }

    private static func resolveSummaryIndex(in summary: [Value], requested: Int?) -> Int {
      if let requested {
        return requested
      }
      if let existingIndex = summary.lastIndex(where: {
        let type = $0.objectValue?["type"]?.stringValue
        return type == "summary_text" || type == "text"
      }) {
        return existingIndex
      }
      return summary.count
    }

    private static func resolveContentIndex(in content: [Value], requested: Int?, preferredType: String) -> Int {
      if let requested {
        return requested
      }
      if let existingIndex = content.lastIndex(where: { $0.objectValue?["type"]?.stringValue == preferredType }) {
        return existingIndex
      }
      return content.count
    }

    private static func mergeOutputItem(existing: [String: Value], with incoming: [String: Value]) -> [String: Value] {
      var merged = existing
      for (key, value) in incoming {
        merged[key] = value
      }

      if let existingContent = existing["content"],
         incoming["content"] == nil || isEmptyArray(incoming["content"])
      {
        merged["content"] = existingContent
      }
      if let existingSummary = existing["summary"],
         incoming["summary"] == nil || isEmptyArray(incoming["summary"])
      {
        merged["summary"] = existingSummary
      }
      if let existingArguments = existing["arguments"], incoming["arguments"] == nil {
        merged["arguments"] = existingArguments
      }

      return merged
    }

    private static func mergeContentPart(existing: [String: Value], with incoming: [String: Value]) -> [String: Value] {
      var merged = existing
      for (key, value) in incoming {
        merged[key] = value
      }
      return merged
    }

    private static func isEmptyArray(_ value: Value?) -> Bool {
      guard case let .array(array)? = value else { return false }
      return array.isEmpty
    }

    private static func syntheticSummaryItem() -> [String: Value] {
      [
        "type": .string("summary_text"),
        "text": .string(""),
      ]
    }

    private static func syntheticOutputItem(ofType type: String) -> [String: Value] {
      switch type {
        case OutputItemType.reasoning:
          [
            "type": .string(OutputItemType.reasoning),
            "summary": .array([.object(syntheticSummaryItem())]),
            "content": .array([]),
          ]
        case OutputItemType.functionCall:
          [
            "type": .string(OutputItemType.functionCall),
            "arguments": .string(""),
          ]
        default:
          [
            "type": .string(OutputItemType.message),
            "role": .string("assistant"),
            "content": .array([]),
          ]
      }
    }

    private static func syntheticContentPart(ofType type: String) -> [String: Value] {
      switch type {
        case "reasoning_text":
          [
            "type": .string("reasoning_text"),
            "text": .string(""),
          ]
        case OutputItemType.refusal:
          [
            "type": .string(OutputItemType.refusal),
            "refusal": .string(""),
          ]
        default:
          [
            "type": .string(OutputItemType.outputText),
            "text": .string(""),
          ]
      }
    }
  }

  private static func inputItems(for message: Message) async throws -> [[String: any Sendable]] {
    func messageAttachmentContentItem(for attachment: Attachment) async throws -> [String: any Sendable]? {
      switch attachment.kind {
        case let .image(data, mimeType):
          let (processedImageData, processedMimeType) = try await MediaProcessor.resizeImageIfNeeded(data, mimeType: mimeType)
          return [
            "type": ContentType.inputImage,
            "detail": "auto",
            "image_url": MediaProcessor.toBase64DataURL(processedImageData, mimeType: processedMimeType),
          ]
        case let .document(data, mimeType):
          // The API expects a data URL (e.g. "data:application/pdf;base64,...") for file_data,
          // despite the OpenAI TS SDK describing it as "base64-encoded data".
          var contentItem: [String: any Sendable] = [
            "type": ContentType.inputFile,
            "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
          ]
          if let fileName = attachment.filename {
            contentItem["filename"] = fileName
          }
          return contentItem
        case .audio, .video:
          openAIResponsesLogger.warning("Attachment type '\(attachment.kind.mimeType)' is not supported in Responses message content and will be omitted.")
          return nil
      }
    }

    func downgradedTextContentItem(for block: Message.Content) -> [String: any Sendable]? {
      let text: String? = switch block {
        case let .text(text) where !text.isEmpty:
          text
        case let .providerOpaque(opaque) where opaque.provider == "openai-responses" && opaque.type == "annotated_output_text":
          opaque.content
        case let .providerOpaque(opaque) where opaque.provider == "openai-responses" && opaque.type == "refusal":
          opaque.content
        case let .providerOpaque(opaque) where opaque.provider == "openai-chat-completions" && opaque.type == "refusal":
          opaque.content
        case let .providerOpaque(opaque) where opaque.isResponseContent:
          // Responses can replay its own opaque items natively, but foreign visible
          // opaque output should still survive provider switches as plain input text.
          opaque.content
        default:
          nil
      }

      guard let text, !text.isEmpty else { return nil }
      return [
        "type": ContentType.inputText,
        "text": text,
      ]
    }

    switch message.role {
      case .user:
        var contentItems: [[String: any Sendable]] = []
        for block in message.content {
          switch block {
            case let .text(text) where !text.isEmpty:
              contentItems.append([
                "type": ContentType.inputText,
                "text": text,
              ])
            case let .attachment(attachment):
              if let contentItem = try await messageAttachmentContentItem(for: attachment) {
                contentItems.append(contentItem)
              }
            default:
              break
          }
        }
        guard !contentItems.isEmpty else { return [] }
        return [[
          "type": ContentType.message,
          "role": "user",
          "content": contentItems,
        ]]

      case .assistant:
        var items: [[String: any Sendable]] = []
        var contentItems: [[String: any Sendable]] = []
        // Per-message metadata from a prior API response. When present, the message
        // is serialized as a ResponseOutputMessage (output_text, id, status). When
        // absent, it's serialized as an EasyInputMessage (input_text).
        var currentMetadata: [String: String]?
        // Preserve assistant phase across downgraded EasyInputMessage segments even
        // when id/status cannot be replayed.
        var currentPhase: String?

        /// The content type for text parts: output_text when replaying prior API output,
        /// input_text for manually constructed assistant messages.
        var textContentType: String {
          currentMetadata != nil ? ContentType.outputText : ContentType.inputText
        }

        func flushContentItems() {
          guard !contentItems.isEmpty else { return }
          var messageItem: [String: any Sendable] = [
            "type": ContentType.message,
            "role": "assistant",
            "content": contentItems,
          ]
          if let metadata = currentMetadata {
            if let id = metadata["id"] { messageItem["id"] = id }
            if let status = metadata["status"] { messageItem["status"] = status }
          }
          if let currentPhase {
            messageItem["phase"] = currentPhase
          }
          items.append(messageItem)
          contentItems.removeAll(keepingCapacity: true)
        }

        for block in message.content {
          switch block {
            case let .providerOpaque(opaque) where opaque.provider == "openai-responses" && opaque.type == "message_metadata":
              // Flush any preceding content items with the previous metadata,
              // then start a new message group with the new metadata
              flushContentItems()
              if let jsonString = opaque.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: String]
              {
                currentMetadata = parsed
                currentPhase = parsed["phase"]
              } else {
                // Metadata block is present but unusable — treat as absent so that
                // subsequent content is serialized as EasyInputMessage (input_text)
                // rather than as a ResponseOutputMessage missing required fields.
                currentMetadata = nil
                currentPhase = nil
              }
            case let .text(text) where !text.isEmpty:
              var item: [String: any Sendable] = [
                "type": textContentType,
                "text": text,
              ]
              if currentMetadata != nil {
                item["annotations"] = [[String: any Sendable]]()
              }
              contentItems.append(item)
            case let .providerOpaque(block) where block.provider == "openai-responses" && block.type == "annotated_output_text":
              if let text = block.content {
                var item: [String: any Sendable] = [
                  "type": textContentType,
                  "text": text,
                ]
                // Annotations are only valid on output_text (ResponseOutputMessage),
                // not on input_text (EasyInputMessage).
                if currentMetadata != nil {
                  let annotations: [[String: any Sendable]] = if let jsonString = block.data,
                                                                 let jsonData = jsonString.data(using: .utf8),
                                                                 let parsedAnnotations = try? JSONSerialization.jsonObject(with: jsonData) as? [[String: any Sendable]]
                  {
                    parsedAnnotations
                  } else {
                    []
                  }
                  item["annotations"] = annotations
                }
                contentItems.append(item)
              }
            case let .providerOpaque(block) where block.provider == "openai-responses" && block.type == "refusal":
              if let refusal = block.content {
                if currentMetadata != nil {
                  // Refusal is only valid in ResponseOutputMessage.
                  contentItems.append([
                    "type": OutputItemType.refusal,
                    "refusal": refusal,
                  ])
                } else {
                  // Without metadata, fall back to plain text for EasyInputMessage.
                  contentItems.append([
                    "type": ContentType.inputText,
                    "text": refusal,
                  ])
                }
              }
            case let .providerOpaque(block) where block.provider == "openai-chat-completions" && block.type == "refusal":
              if let refusal = block.content {
                contentItems.append([
                  "type": ContentType.inputText,
                  "text": refusal,
                ])
              }
            case let .providerOpaque(block) where block.isResponseContent && block.provider != "openai-responses":
              // Keep visible foreign opaque output in assistant history even though
              // only native Responses opaque blocks can be replayed structurally.
              if let text = block.content, !text.isEmpty {
                var item: [String: any Sendable] = [
                  "type": textContentType,
                  "text": text,
                ]
                if currentMetadata != nil {
                  item["annotations"] = [[String: any Sendable]]()
                }
                contentItems.append(item)
              }
            case let .attachment(attachment):
              if let contentItem = try await messageAttachmentContentItem(for: attachment) {
                // ResponseOutputMessage content only supports output_text/refusal.
                // If a caller stored attachments inside the same assistant turn, flush the
                // replayed output segment and continue as an EasyInputMessage.
                if currentMetadata != nil {
                  flushContentItems()
                  currentMetadata = nil
                }
                contentItems.append(contentItem)
              }
            case let .toolCall(toolCall):
              flushContentItems()
              let foundationParams = Value.toSendable(toolCall.parameters)
              let argumentsData = try JSONSerialization.data(withJSONObject: foundationParams, options: [])
              guard let argumentsString = String(data: argumentsData, encoding: .utf8) else {
                throw AIError.invalidRequest(message: "Failed to serialize function call arguments to JSON string")
              }
              items.append([
                "type": ContentType.functionCall,
                "call_id": toolCall.id,
                "name": toolCall.name,
                "arguments": argumentsString,
              ])
            case let .providerOpaque(block) where block.provider == "openai-responses" && block.type == "reasoning":
              flushContentItems()
              if let jsonString = block.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable]
              {
                items.append(parsed)
              } else {
                // Fallback for opaque blocks created before full raw storage
                var reasoningItem: [String: any Sendable] = [
                  "type": OutputItemType.reasoning,
                ]
                if let id = block.signature {
                  reasoningItem["id"] = id
                }
                if let summaryText = block.content {
                  reasoningItem["summary"] = [["type": "summary_text", "text": summaryText]]
                }
                if let encryptedContent = block.data {
                  reasoningItem["encrypted_content"] = encryptedContent
                }
                items.append(reasoningItem)
              }
            case let .providerOpaque(block) where block.provider == "openai-responses":
              // Round-trip preserved server-side tool items (web_search_call, code_interpreter_call, etc.)
              flushContentItems()
              if let jsonString = block.data,
                 let jsonData = jsonString.data(using: .utf8),
                 let parsed = try? JSONSerialization.jsonObject(with: jsonData) as? [String: any Sendable]
              {
                items.append(parsed)
              } else if block.isResponseContent, let text = block.content, !text.isEmpty {
                // If a native Responses opaque block is missing raw JSON, downgrade it
                // to plain assistant text instead of dropping the visible output.
                currentMetadata = nil
                contentItems.append([
                  "type": ContentType.inputText,
                  "text": text,
                ])
              }
            default:
              break
          }
        }

        flushContentItems()
        return items

      case .tool:
        return try message.content.compactMap { block -> [String: any Sendable]? in
          guard case let .toolResult(toolResult) = block else { return nil }

          let resultOutput: any Sendable
          if toolResult.isError == true {
            let errorText = toolResult.content.compactMap { content -> String? in
              if case let .text(text) = content { return text }
              return nil
            }.joined(separator: "\n")
            let errorPayload: [String: String] = ["error": errorText.isEmpty ? "Unknown error" : errorText]
            let errorData = try JSONSerialization.data(withJSONObject: errorPayload, options: [])
            resultOutput = String(data: errorData, encoding: .utf8) ?? "{\"error\":\"Unknown error\"}"
          } else {
            var outputItems: [[String: any Sendable]] = []
            var hasNonTextContent = false

            for content in toolResult.content {
              switch content {
                case let .text(text):
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": text,
                  ])
                case let .image(data, mimeType):
                  hasNonTextContent = true
                  let mediaType = mimeType ?? "image/png"
                  let dataURL = "data:\(mediaType);base64,\(data.base64EncodedString())"
                  outputItems.append([
                    "type": ContentType.inputImage,
                    "detail": "auto",
                    "image_url": dataURL,
                  ])
                case let .audio(data, mimeType):
                  openAIResponsesLogger.warning("Tool '\(toolResult.name)' returned audio, which is not supported by Responses API. Using fallback text.")
                  outputItems.append([
                    "type": ContentType.inputText,
                    "text": ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription,
                  ])
                case let .file(data, mimeType, filename):
                  hasNonTextContent = true
                  if mimeType.hasPrefix("image/") {
                    let dataURL = "data:\(mimeType);base64,\(data.base64EncodedString())"
                    outputItems.append([
                      "type": ContentType.inputImage,
                      "detail": "auto",
                      "image_url": dataURL,
                    ])
                  } else {
                    // Use data URL format consistent with input file encoding
                    var fileItem: [String: any Sendable] = [
                      "type": ContentType.inputFile,
                      "file_data": MediaProcessor.toBase64DataURL(data, mimeType: mimeType),
                    ]
                    if let filename {
                      fileItem["filename"] = filename
                    }
                    outputItems.append(fileItem)
                  }
              }
            }

            if hasNonTextContent {
              resultOutput = outputItems
            } else {
              // Text-only: use plain string shorthand
              let texts = outputItems.compactMap { $0["text"] as? String }
              resultOutput = texts.joined(separator: "\n")
            }
          }

          return [
            "type": ContentType.functionCallOutput,
            "call_id": toolResult.id,
            "output": resultOutput,
          ]
        }

      case .system, .developer:
        var contentItems: [[String: any Sendable]] = []
        for block in message.content {
          if let textContentItem = downgradedTextContentItem(for: block) {
            contentItems.append(textContentItem)
            continue
          }
          if case let .attachment(attachment) = block,
             let attachmentContentItem = try await messageAttachmentContentItem(for: attachment)
          {
            contentItems.append(attachmentContentItem)
          }
        }
        guard !contentItems.isEmpty else { return [] }
        return [[
          "type": ContentType.message,
          "role": message.role.rawValue,
          "content": contentItems,
        ]]
    }
  }

  // MARK: - Streaming Event Types

  private enum StreamEventType {
    static let outputTextDelta = "response.output_text.delta"
    static let reasoningTextDelta = "response.reasoning_text.delta"
    static let reasoningDelta = "response.reasoning.delta" // Deprecated, kept for older API versions
    static let reasoningSummaryDelta = "response.reasoning_summary_text.delta"
    static let reasoningSummaryDone = "response.reasoning_summary_text.done"
    static let outputItemAdded = "response.output_item.added"
    static let contentPartAdded = "response.content_part.added"
    static let refusalDelta = "response.refusal.delta"
    static let refusalDone = "response.refusal.done"
    static let functionCallArgumentsDelta = "response.function_call_arguments.delta"
    static let functionCallArgumentsDone = "response.function_call_arguments.done"
    static let completed = "response.completed"
    static let failed = "response.failed"
    static let incomplete = "response.incomplete"
    static let created = "response.created"
  }

  // MARK: - Output Item Types

  private enum OutputItemType {
    static let functionCall = "function_call"
    static let reasoning = "reasoning"
    static let codeInterpreterCall = "code_interpreter_call"
    static let webSearchCall = "web_search_call"
    static let message = "message"
    static let outputText = "output_text"
    static let refusal = "refusal"
  }

  private func streamResponse(
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
    textFormat: ResponseFormat? = nil,
    tools: [Tool] = [],
    enableStrictModeForTools: Bool = true,
  ) async throws -> AsyncThrowingStream<GenerationResponse, Error> {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.timeoutInterval = 600.0 // 10 minutes - suitable for o3's long response times
    if let apiKey {
      request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    }

    let patchedInput = Message.patchingOrphanedToolCalls(input)
    var inputContent: [[String: any Sendable]] = []
    for message in patchedInput {
      try await inputContent.append(contentsOf: Self.inputItems(for: message))
    }

    var body: [String: any Sendable] = [
      "model": modelId,
      "input": inputContent, // Use the constructed input array
      "stream": stream,
    ]

    // Add background mode support
    if backgroundMode {
      body["background"] = true
      body["store"] = true // Required for background mode
    }

    var toolsArray: [[String: any Sendable]] = []

    // Server-side tools (provider-specific)
    for serverSideTool in serverSideTools {
      toolsArray.append(serverSideTool.definition)
    }

    // Function calling tools - rawInputSchema is always populated
    if !tools.isEmpty {
      for tool in tools {
        let parameters: [String: any Sendable]
          // Transform schema for strict mode compliance if enabled
          = if enableStrictModeForTools
        {
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
      // Set tool_choice to auto by default when tools are present
      body["tool_choice"] = "auto" // Can also be "required", "none", or {"type": "function", "name": "my_func"}
    }

    // Add system prompt as instructions
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
      body["include"] = ["reasoning.encrypted_content"]
    }

    // Build text configuration (format and/or verbosity)
    var textConfig: [String: any Sendable] = [:]
    if let textFormat {
      var formatConfig: [String: any Sendable] = [:]
      switch textFormat {
        case .text:
          formatConfig["type"] = "text"
        case .jsonObject:
          formatConfig["type"] = "json_object" // Use this for older JSON mode if needed
        case let .jsonSchema(schema, name, description):
          formatConfig["type"] = "json_schema" // Preferred for structured JSON
          formatConfig["schema"] = schema // Should be [String: any Sendable] representing the JSON schema
          if let name { formatConfig["name"] = name }
          if let description { formatConfig["description"] = description }
          // Add strict mode if desired, e.g., formatConfig["strict"] = true
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
          // For background mode with streaming, stream directly but with proper retry logic
          try await streamBackgroundResponseDirect(
            request: request,
            apiKey: apiKey,
            continuation: continuation,
          )
        } else if backgroundMode {
          openAIResponsesLogger.log("Initiating background mode response without streaming in OpenAI Responses client")
          // For background mode without streaming, get response ID and poll
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
          // Poll for completion
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

    // Set up cancellation handler to cancel the stream task when the consumer cancels
    continuation.onTermination = { @Sendable termination in
      if case .cancelled = termination {
        openAIResponsesLogger.log("AsyncThrowingStream cancelled by consumer - cancelling stream task")
        streamTask.cancel()
      }
    }
    return resultStream
  }

  private func handleErrorResponse(_ httpResponse: HTTPURLResponse, data: Data) throws {
    try AIError.throwOpenAIHTTPError(httpResponse, data: data, logger: openAIResponsesLogger)
  }

  /// Generates a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
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
      stream: false,
      configuration: configuration,
      update: { _ in },
    )
  }

  /// Streams a text response from the given conversation messages.
  ///
  /// - Parameters:
  ///   - modelId: The model identifier.
  ///   - tools: Tools available for the model to use.
  ///   - systemPrompt: System instructions for the model.
  ///   - messages: The conversation history.
  ///   - maxTokens: Maximum tokens in the response.
  ///   - temperature: Sampling temperature.
  ///   - apiKey: API key for authentication.
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
          stream: true,
          configuration: configuration,
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
    stream: Bool,
    configuration: Configuration,
    update: @Sendable @escaping (GenerationResponse) -> Void,
  ) async throws -> GenerationResponse {
    await MainActor.run {
      isGenerating = true
    }

    let task = Task<GenerationResponse, Error> {
      var finalContent: [Message.Content] = []
      var finalMetadata: GenerationResponse.Metadata?

      do {
        let stream = try await streamResponse(
          input: messages,
          systemPrompt: systemPrompt,
          modelId: modelId,
          apiKey: apiKey,
          maxTokens: maxTokens,
          temperature: temperature,
          stream: stream,
          reasoningEffortLevel: configuration.reasoningEffortLevel,
          verbosityLevel: configuration.verbosityLevel,
          serverSideTools: configuration.serverSideTools,
          backgroundMode: configuration.backgroundMode,
          tools: tools,
          enableStrictModeForTools: configuration.enableStrictModeForTools,
        )

        for try await chunk in stream {
          try Task.checkCancellation()

          finalContent = chunk.content
          finalMetadata = chunk.metadata

          await MainActor.run {
            update(chunk)
          }
        }

        // If cancelled, return the state as it was when cancellation was detected
        if Task.isCancelled {
          openAIResponsesLogger.log("Generation task returning cancelled state")
          return .init(content: finalContent, metadata: finalMetadata)
        }

        // Stream finished normally, return the final state
        return .init(content: finalContent, metadata: finalMetadata)

      } catch {
        // Handle cancellation error specifically if it bubbles up
        if error is CancellationError || (error as NSError).code == NSURLErrorCancelled {
          openAIResponsesLogger.log("Generation task caught cancellation error.")
          return .init(content: finalContent, metadata: finalMetadata)
        } else {
          // Rethrow other errors
          openAIResponsesLogger.error("Generation failed: \(error)")
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
    activeBackgroundResponseId = nil
    activeBackgroundResponseApiKey = nil
  }

  /// Cancels any ongoing generation task and active background response.
  @MainActor
  public func stop() {
    openAIResponsesLogger.log("Stop called - cancelling current task")
    currentTask?.cancel()
    // Also cancel background response if one is active
    if let backgroundResponseId = activeBackgroundResponseId {
      let apiKey = activeBackgroundResponseApiKey
      openAIResponsesLogger.log("Stop called - cancelling background response \(backgroundResponseId)")
      Task {
        try? await cancelBackgroundResponse(responseId: backgroundResponseId, apiKey: apiKey)
      }
    }
  }

  // MARK: - Shared Event Processing

  private func processStreamingEvent(
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
        // Content parts are tracked by (output_index, content_index) in the streaming state,
        // so multiple parts within the same output item (e.g., text + refusal) are preserved
        // separately. Terminal responses are built from the completed or accumulated snapshot.
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
              // Reasoning text arrives via subsequent reasoning_text.delta events.
              // Don't append summary text here — it would be duplicated by the deltas.
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
          streamingState.completeToolCallArguments(argumentsString, outputIndex: event.outputIndex, itemId: event.itemId)
          yieldCurrentState()
        }

      default:
        break
    }
  }

  private func yieldFinalResponse(
    _ generationResponse: GenerationResponse,
    mergingToolCallsFrom streamingState: StreamingResponseState,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) {
    var finalContent = generationResponse.content

    // Some Responses API streams omit output_index on intermediate
    // function-call events. In that case, recover tool calls from the
    // incremental stream state instead of dropping them in the terminal chunk.
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

  // MARK: - Background Response Methods

  private func streamBackgroundResponseDirect(
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

  private func streamBackgroundResponse(
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
    request.timeoutInterval = 600.0 // 10 minutes - suitable for o3's long response times
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

  /// Shared streaming logic for both direct and resumption modes
  private func performBackgroundStream(
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
            // Only log every 10 sequences to reduce noise
            if sequenceNumber % 10 == 0 || sequenceNumber - lastSequenceNumber > 10 {
              openAIResponsesLogger.log("\(logPrefix): Progress update - sequence: \(sequenceNumber)")
            }
          }
          lastSequenceNumber = sequenceNumber
        },
      )
    } catch {
      // Check if this is a cancellation error (expected when user stops)
      let isCancellationError = error is CancellationError || (error as NSError).code == NSURLErrorCancelled

      if isCancellationError {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled by user")
        // Cancel the background response on the server if we have a response ID
        if let responseId = currentResponseId {
          openAIResponsesLogger.log("\(logPrefix): Cancelling background response \(responseId) on server")
          try? await cancelBackgroundResponse(responseId: responseId, apiKey: apiKey)
        }
        return // Don't retry or throw for cancellation
      }

      openAIResponsesLogger.log("\(logPrefix): Error occurred - \(error)")
      // Handle timeout and connection errors with retry logic
      if retryCount < maxRetries {
        let isTimeoutError = (error as NSError).code == NSURLErrorTimedOut ||
          (error as NSError).code == NSURLErrorNetworkConnectionLost ||
          (error as NSError).code == NSURLErrorCannotConnectToHost

        if isTimeoutError {
          openAIResponsesLogger.warning("\(logPrefix) timeout (attempt \(retryCount + 1)/\(maxRetries + 1))")

          // If we have a response ID and are in direct mode, switch to resumption mode
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

          // Check response status before retrying (only if we have a response ID)
          if let responseId = currentResponseId ?? responseId {
            openAIResponsesLogger.log("\(logPrefix): Checking response status before retry...")
            if try await checkResponseStatusAndHandle(
              responseId: responseId,
              apiKey: apiKey,
              continuation: continuation,
              logPrefix: logPrefix,
            ) {
              return // Response was completed or handled
            }
          }

          // Exponential backoff and retry
          let backoffDelay = TimeInterval(pow(2.0, Double(retryCount)))
          openAIResponsesLogger.log("\(logPrefix): Waiting \(backoffDelay)s before retry...")
          try await Task.sleep(nanoseconds: UInt64(backoffDelay * 1_000_000_000))

          if isDirect {
            // Don't replay the initial create request without a response ID.
            // The server may have accepted the first request before the transport
            // error occurred, and replaying would create a duplicate job.
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

      // If not a timeout error or max retries exceeded, rethrow
      // But don't log as error for cancellation since that's expected
      if (error as NSError).code != NSURLErrorCancelled {
        openAIResponsesLogger.error("\(logPrefix) failed after \(retryCount) retries: \(error)")
      } else {
        openAIResponsesLogger.log("\(logPrefix): Stream cancelled after \(retryCount) retries")
      }
      throw error
    }
  }

  /// Shared SSE (Server-Sent Events) streaming logic for all stream types
  private func performSSEStream(
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

        // Track sequence number for resumption (background mode only)
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

  /// Helper method to check response status and handle completion/failure
  private func checkResponseStatusAndHandle(
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

  /// Helper method to parse completed response
  private func parseCompletedResponse(
    _ response: ResponseObject,
    continuation: AsyncThrowingStream<GenerationResponse, Error>.Continuation,
  ) {
    continuation.yield(response.toGenerationResponse())
  }

  private func pollBackgroundResponse(
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
          // Continue polling
          try await Task.sleep(nanoseconds: 2_000_000_000) // Sleep for 2 seconds
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

  /// Gets the current status of a background response.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to check.
  ///   - apiKey: API key for authentication.
  /// - Returns: The background response status and result if completed.
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

    // Parse response if terminal with usable payload
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

  /// Cancels a background response that is in progress.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to cancel.
  ///   - apiKey: API key for authentication.
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
      // Cancelling twice is idempotent, so don't throw on certain error codes
      if httpResponse.statusCode != 409 { // Conflict - already cancelled
        throw AIError.fromHTTPStatusCode(httpResponse.statusCode)
      }
    }
  }

  /// Deletes a response and its associated data permanently.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the response to delete.
  ///   - apiKey: API key for authentication.
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

  /// Resumes streaming from a background response at a specific sequence number.
  ///
  /// Use this to continue receiving updates after a connection was dropped.
  ///
  /// - Parameters:
  ///   - responseId: The ID of the background response to resume.
  ///   - apiKey: API key for authentication.
  ///   - startingAfter: The sequence number to resume from.
  ///   - update: Callback invoked with each streamed response chunk.
  /// - Returns: The final generation response.
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

extension ResponsesClient {
  /// Reasoning effort level for models that support extended thinking.
  /// Maps to the `effort` value in the API's `reasoning` object.
  public enum ReasoningEffortLevel: String, CaseIterable, Identifiable, Sendable {
    /// No reasoning. Default for gpt-5.1.
    case none
    /// Minimal reasoning effort for simple tasks.
    case minimal
    /// Low reasoning effort for straightforward tasks.
    case low
    /// Medium reasoning effort for balanced performance.
    case medium
    /// High reasoning effort for complex tasks.
    case high
    /// Maximum reasoning effort. Supported for models after gpt-5.1-codex-max.
    case xhigh

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Response verbosity level controlling output detail.
  public enum VerbosityLevel: String, CaseIterable, Identifiable, Sendable {
    /// Concise responses.
    case low
    /// Balanced detail level.
    case medium
    /// Detailed responses.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// Context size for web search results.
  public enum WebSearchContextSize: String, CaseIterable, Identifiable, Sendable {
    /// Fewer search results, lower latency.
    case low
    /// Balanced search results.
    case medium
    /// More search results, higher latency.
    case high

    /// The raw value identifier.
    public var id: String {
      rawValue
    }
  }

  /// A server-side tool that runs on the provider's infrastructure.
  ///
  /// Different providers support different server-side tools. Use the
  /// static factory methods on `OpenAI` or `xAI` to create tools.
  public struct ServerSideTool: Sendable, Equatable {
    /// The raw tool definition dictionary sent to the API.
    public let definition: [String: any Sendable]

    /// Creates a server-side tool from a raw definition dictionary.
    ///
    /// - Parameter definition: The tool definition as expected by the API.
    public init(_ definition: [String: any Sendable]) {
      self.definition = definition
    }

    // MARK: - OpenAI Tools

    /// Factory methods for OpenAI server-side tools.
    public enum OpenAI {
      /// Web search tool with configurable context size.
      public static func webSearch(contextSize: WebSearchContextSize) -> ServerSideTool {
        ServerSideTool([
          "type": "web_search_preview",
          "search_context_size": contextSize.rawValue,
        ])
      }

      /// Code interpreter (Python execution environment).
      public static func codeInterpreter() -> ServerSideTool {
        ServerSideTool(["type": "code_interpreter", "container": ["type": "auto"]])
      }
    }

    // MARK: - xAI Tools

    /// Factory methods for xAI server-side tools.
    public enum xAI {
      /// Web search tool for searching the internet.
      public static func webSearch() -> ServerSideTool {
        ServerSideTool(["type": "web_search"])
      }

      /// X/Twitter search tool for searching posts, users, and threads.
      public static func xSearch() -> ServerSideTool {
        ServerSideTool(["type": "x_search"])
      }

      /// Code execution tool for running Python code.
      public static func codeExecution() -> ServerSideTool {
        ServerSideTool(["type": "code_interpreter"])
      }
    }

    // MARK: - Equatable

    public static func == (lhs: ServerSideTool, rhs: ServerSideTool) -> Bool {
      // Compare by serializing to JSON since [String: any Sendable] isn't directly Equatable
      guard let lhsData = try? JSONSerialization.data(withJSONObject: lhs.definition, options: .sortedKeys),
            let rhsData = try? JSONSerialization.data(withJSONObject: rhs.definition, options: .sortedKeys)
      else {
        return false
      }
      return lhsData == rhsData
    }
  }

  /// Configuration options for Responses API requests.
  public struct Configuration: Sendable {
    /// Reasoning effort level for extended thinking models.
    public var reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel?

    /// Response verbosity level.
    public var verbosityLevel: ResponsesClient.VerbosityLevel?

    /// Server-side tools to enable (web search, code interpreter, etc.).
    public var serverSideTools: [ServerSideTool]

    /// Enable background mode for long-running requests.
    public var backgroundMode: Bool

    /// When true, tool schemas are rewritten for OpenAI strict mode compliance and sent
    /// with `strict: true`. This ensures the model's output exactly matches the schema,
    /// but requires all properties to be in `required` and optional properties to be nullable.
    /// Defaults to true. Disable for tools with optional non-nullable parameters or for
    /// compatible endpoints that don't support strict mode.
    public var enableStrictModeForTools: Bool

    /// Creates a new configuration with the specified options.
    ///
    /// - Parameters:
    ///   - reasoningEffortLevel: Reasoning effort for thinking models.
    ///   - verbosityLevel: Response verbosity level.
    ///   - serverSideTools: Server-side tools to enable.
    ///   - backgroundMode: Enable background mode for long requests.
    ///   - enableStrictModeForTools: Rewrite tool schemas for strict mode compliance.
    public init(
      reasoningEffortLevel: ResponsesClient.ReasoningEffortLevel? = nil,
      verbosityLevel: ResponsesClient.VerbosityLevel? = nil,
      serverSideTools: [ServerSideTool] = [],
      backgroundMode: Bool = false,
      enableStrictModeForTools: Bool = true,
    ) {
      self.reasoningEffortLevel = reasoningEffortLevel
      self.verbosityLevel = verbosityLevel
      self.serverSideTools = serverSideTools
      self.backgroundMode = backgroundMode
      self.enableStrictModeForTools = enableStrictModeForTools
    }
  }

  /// Status of a background response request.
  public enum BackgroundResponseStatus: String, CaseIterable, Sendable {
    /// Request is waiting to be processed.
    case queued
    /// Request is currently being processed.
    case in_progress
    /// Request completed successfully.
    case completed
    /// Request ended early (e.g., max_output_tokens or content filtering) but the response payload is usable.
    case incomplete
    /// Request failed with an error.
    case failed
    /// Request was cancelled.
    case cancelled
  }

  /// Information about a background response request.
  public struct BackgroundResponse: Sendable {
    /// The unique identifier for this response.
    public let id: String
    /// The current status of the response.
    public let status: BackgroundResponseStatus
    /// The generation response if completed, nil otherwise.
    public let response: GenerationResponse?
    /// Error message if the request failed.
    public let error: String?
  }

  // MARK: - Streaming Event Types

  struct StreamEvent: Decodable {
    let type: String?
    let sequenceNumber: Int?
    let delta: String?
    let text: String?
    let refusal: String?
    let itemId: String?
    let outputIndex: Int?
    let contentIndex: Int?
    let summaryIndex: Int?
    let arguments: String?
    let item: ResponseOutputItem?
    let part: ContentItem?
    let response: ResponseObject?
    let error: ErrorObject?

    enum CodingKeys: String, CodingKey {
      case type
      case sequenceNumber = "sequence_number"
      case delta
      case text
      case refusal
      case itemId = "item_id"
      case outputIndex = "output_index"
      case contentIndex = "content_index"
      case summaryIndex = "summary_index"
      case arguments
      case item
      case part
      case response
      case error
    }
  }

  struct SummaryItem: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var text: String? {
      raw["text"]?.stringValue
    }
  }

  struct Usage: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var inputTokens: Int? {
      raw["input_tokens"]?.intValue
    }

    var outputTokens: Int? {
      raw["output_tokens"]?.intValue
    }

    var totalTokens: Int? {
      raw["total_tokens"]?.intValue
    }

    var inputTokensDetails: InputTokensDetails? {
      raw["input_tokens_details"]?.objectValue.map(InputTokensDetails.init(raw:))
    }

    var outputTokensDetails: OutputTokensDetails? {
      raw["output_tokens_details"]?.objectValue.map(OutputTokensDetails.init(raw:))
    }
  }

  struct InputTokensDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var cachedTokens: Int? {
      raw["cached_tokens"]?.intValue
    }
  }

  struct OutputTokensDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var reasoningTokens: Int? {
      raw["reasoning_tokens"]?.intValue
    }
  }

  struct IncompleteDetails: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var reason: String? {
      raw["reason"]?.stringValue
    }
  }

  struct ResponseObject: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var id: String? {
      raw["id"]?.stringValue
    }

    var status: String? {
      raw["status"]?.stringValue
    }

    var model: String? {
      raw["model"]?.stringValue
    }

    var createdAt: Int? {
      raw["created_at"]?.intValue
    }

    var output: [ResponseOutputItem]? {
      raw["output"]?.arrayValue?.compactMap(\.objectValue).map(ResponseOutputItem.init(raw:))
    }

    var outputText: String? {
      raw["output_text"]?.stringValue
    }

    var usage: Usage? {
      raw["usage"]?.objectValue.map(Usage.init(raw:))
    }

    var error: ErrorObject? {
      raw["error"]?.objectValue.map(ErrorObject.init(raw:))
    }

    var incompleteDetails: IncompleteDetails? {
      raw["incomplete_details"]?.objectValue.map(IncompleteDetails.init(raw:))
    }

    /// Converts the response object to a GenerationResponse
    func toGenerationResponse() -> GenerationResponse {
      var content: [Message.Content] = []
      var hasRefusal = false
      var citations: [(label: String, url: String?, fileId: String?)] = []

      if let outputArray = output, !outputArray.isEmpty {
        for item in outputArray {
          guard let itemType = item.type else { continue }

          switch itemType {
            case OutputItemType.message:
              // Insert a boundary marker before each message item's content so that
              // multi-message assistant turns can be split back into separate items
              // during serialization.
              var metadata: [String: String] = [:]
              if let messageId = item.id { metadata["id"] = messageId }
              if let status = item.status { metadata["status"] = status }
              if let phase = item.phase { metadata["phase"] = phase }
              if !metadata.isEmpty,
                 let jsonData = try? JSONSerialization.data(withJSONObject: metadata),
                 let jsonString = String(data: jsonData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "message_metadata",
                  data: jsonString,
                )))
              }
              if let contentArray = item.content {
                for contentItem in contentArray {
                  if contentItem.type == OutputItemType.outputText, let text = contentItem.text, !text.isEmpty {
                    if let annotations = contentItem.annotations, !annotations.isEmpty {
                      // Collect citations for display endnotes
                      for annotation in annotations {
                        switch annotation.type {
                          case "url_citation":
                            if let url = annotation.url {
                              citations.append((label: annotation.title ?? url, url: url, fileId: nil))
                            }
                          case "file_citation", "container_file_citation":
                            if let filename = annotation.filename {
                              citations.append((label: filename, url: nil, fileId: annotation.fileId))
                            }
                          default:
                            break
                        }
                      }
                      // Preserve all annotation fields for lossless round-tripping
                      let annotationsRaw = annotations.map { Value.toSendable($0.raw) }
                      let annotationsJson = (try? JSONSerialization.data(withJSONObject: annotationsRaw))
                        .flatMap { String(data: $0, encoding: .utf8) }
                      content.append(.providerOpaque(OpaqueBlock(
                        provider: "openai-responses",
                        type: "annotated_output_text",
                        content: text,
                        data: annotationsJson,
                        isResponseContent: true,
                      )))
                    } else {
                      content.append(.text(text))
                    }
                  } else if contentItem.type == OutputItemType.refusal, let refusal = contentItem.refusal, !refusal.isEmpty {
                    content.append(.providerOpaque(OpaqueBlock(
                      provider: "openai-responses",
                      type: "refusal",
                      content: refusal,
                      isResponseContent: true,
                    )))
                    hasRefusal = true
                  }
                }
              }
            case OutputItemType.reasoning:
              // Prefer reasoning text from content array (reasoning_text items),
              // fall back to summary text
              let reasoningContentText = item.content?
                .filter { $0.type == "reasoning_text" }
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let summaryText = item.summary?
                .compactMap(\.text)
                .joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
              let reasoningText = (reasoningContentText?.isEmpty == false) ? reasoningContentText : summaryText
              if let reasoningText, !reasoningText.isEmpty {
                content.append(.thinking(text: reasoningText, signature: nil))
              }
              // Preserve the full reasoning item for round-tripping via the Responses API
              if let itemId = item.id,
                 let rawItemData = try? JSONEncoder().encode(item.raw),
                 let rawItemJson = String(data: rawItemData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: "reasoning",
                  content: summaryText,
                  signature: itemId,
                  data: rawItemJson,
                )))
              }
            case OutputItemType.functionCall:
              if let name = item.name,
                 let callId = item.callId,
                 let argumentsString = item.arguments
              {
                var parameters: [String: Value] = [:]
                if let argumentsData = argumentsString.data(using: .utf8),
                   let parsedArgs = try? JSONDecoder().decode([String: Value].self, from: argumentsData)
                {
                  parameters = parsedArgs
                } else if !argumentsString.isEmpty {
                  parameters = ["_parseError": .string("Failed to parse arguments JSON"), "_rawArguments": .string(argumentsString)]
                }
                content.append(.toolCall(ToolCall(
                  name: name,
                  id: callId,
                  parameters: parameters,
                )))
              }
            default:
              // Preserve unknown output item types (web_search_call, code_interpreter_call, etc.)
              // as opaque blocks for multi-turn round-tripping
              let sendable = Value.toSendable(item.raw)
              if let jsonData = try? JSONSerialization.data(withJSONObject: sendable),
                 let jsonString = String(data: jsonData, encoding: .utf8)
              {
                content.append(.providerOpaque(OpaqueBlock(
                  provider: "openai-responses",
                  type: itemType,
                  data: jsonString,
                )))
              }
          }
        }
      } else if let outputText, !outputText.isEmpty {
        content.append(.text(outputText))
      }

      // Format citations as endnotes
      if !citations.isEmpty {
        let uniqueCitations = citations.reduce(into: [(label: String, url: String?, fileId: String?)]()) { result, citation in
          let key = citation.url ?? citation.fileId ?? citation.label
          if !result.contains(where: { ($0.url ?? $0.fileId ?? $0.label) == key }) {
            result.append(citation)
          }
        }
        let endnotes = uniqueCitations.map { citation in
          if let url = citation.url {
            "- [\(citation.label)](\(url))"
          } else {
            "- \(citation.label)"
          }
        }.joined(separator: "\n") + "\n"
        content.append(.endnotes(endnotes))
      }

      // Build metadata from response
      let toolCallCount = content.reduce(into: 0) { count, item in
        if case .toolCall = item {
          count += 1
        }
      }
      let finishReason: GenerationResponse.FinishReason? = if let status {
        switch status {
          case "completed":
            if hasRefusal { .refusal }
            else if toolCallCount > 0 { .toolUse }
            else { .stop }
          case "incomplete":
            // Check the reason for incomplete status
            switch incompleteDetails?.reason {
              case "max_output_tokens": .maxTokens
              case "content_filter": .contentFilter
              default: .other
            }
          case "failed", "cancelled": .other
          default: nil
        }
      } else {
        nil
      }

      var createdAtDate: Date?
      if let createdAt {
        createdAtDate = Date(timeIntervalSince1970: TimeInterval(createdAt))
      }

      let metadata = GenerationResponse.Metadata(
        responseId: id,
        model: model,
        createdAt: createdAtDate,
        finishReason: finishReason,
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
        totalTokens: usage?.totalTokens,
        cacheReadInputTokens: usage?.inputTokensDetails?.cachedTokens,
        reasoningTokens: usage?.outputTokensDetails?.reasoningTokens,
      )

      return GenerationResponse(content: content, metadata: metadata)
    }
  }

  struct ResponseOutputItem: Decodable {
    /// Raw JSON for lossless round-tripping of all output item types.
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var id: String? {
      raw["id"]?.stringValue
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var status: String? {
      raw["status"]?.stringValue
    }

    var name: String? {
      raw["name"]?.stringValue
    }

    var callId: String? {
      raw["call_id"]?.stringValue
    }

    var arguments: String? {
      raw["arguments"]?.stringValue
    }

    var encryptedContent: String? {
      raw["encrypted_content"]?.stringValue
    }

    var phase: String? {
      raw["phase"]?.stringValue
    }

    var content: [ContentItem]? {
      raw["content"]?.arrayValue?.compactMap(\.objectValue).map(ContentItem.init(raw:))
    }

    var summary: [SummaryItem]? {
      raw["summary"]?.arrayValue?.compactMap(\.objectValue).map(SummaryItem.init(raw:))
    }
  }

  struct ContentItem: Decodable {
    /// Raw JSON for lossless round-tripping of all content part types.
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var text: String? {
      raw["text"]?.stringValue
    }

    var refusal: String? {
      raw["refusal"]?.stringValue
    }

    var annotations: [AnnotationItem]? {
      raw["annotations"]?.arrayValue?.compactMap(\.objectValue).map(AnnotationItem.init(raw:))
    }
  }

  struct AnnotationItem: Decodable {
    /// All annotation fields, preserved for lossless round-tripping.
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var type: String? {
      raw["type"]?.stringValue
    }

    var url: String? {
      raw["url"]?.stringValue
    }

    var title: String? {
      raw["title"]?.stringValue
    }

    var filename: String? {
      raw["filename"]?.stringValue
    }

    var fileId: String? {
      raw["file_id"]?.stringValue
    }
  }

  struct ErrorObject: Decodable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    var message: String? {
      raw["message"]?.stringValue
    }

    var code: String? {
      raw["code"]?.stringValue
    }
  }

  /// Checks whether a model supports the reasoning effort parameter.
  /// Defaults to true (latest paradigm), returning false only for known older models.
  static func supportsReasoning(_ modelId: String) -> Bool {
    if modelId.hasPrefix("gpt-3") || modelId.hasPrefix("gpt-4") { return false }
    if modelId.hasPrefix("chatgpt-") { return false }
    if modelId == "o1-mini" || modelId.hasPrefix("o1-mini-") { return false }
    if modelId == "o1-preview" || modelId.hasPrefix("o1-preview-") { return false }
    if modelId.hasPrefix("grok-") {
      return modelId.hasPrefix("grok-3-mini")
    }
    return true
  }
}

/// Response format options for the responses endpoint
enum ResponseFormat {
  case text
  case jsonObject
  case jsonSchema(schema: [String: any Sendable], name: String? = nil, description: String? = nil)
}

private let openAIResponsesLogger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.local-intelligence", category: "ResponsesClient")
