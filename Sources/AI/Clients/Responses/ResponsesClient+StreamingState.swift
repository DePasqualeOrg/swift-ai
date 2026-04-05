// Copyright © Anthony DePasquale

import Foundation

extension ResponsesClient {
  struct ContentKey: Hashable, Comparable {
    let outputIndex: Int
    let contentIndex: Int

    static func < (lhs: ContentKey, rhs: ContentKey) -> Bool {
      if lhs.outputIndex != rhs.outputIndex {
        return lhs.outputIndex < rhs.outputIndex
      }
      return lhs.contentIndex < rhs.contentIndex
    }
  }

  struct StreamingResponseState {
    private static let annotatedOutputTextType = "annotated_output_text"
    var indexedContent: [ContentKey: Message.Content] = [:]
    var fallbackContent: [Message.Content] = []
    var toolCallArgumentBuffers: [String: String] = [:]
    var itemIdToFallbackIndex: [String: Int] = [:]
    var summaryContent: [Int: String] = [:]
    private var nextContentIndex: [Int: Int] = [:]

    private mutating func key(outputIndex: Int, contentIndex: Int?) -> ContentKey {
      if let contentIndex {
        nextContentIndex[outputIndex] = max(nextContentIndex[outputIndex, default: 0], contentIndex + 1)
        return ContentKey(outputIndex: outputIndex, contentIndex: contentIndex)
      }
      return ContentKey(outputIndex: outputIndex, contentIndex: 0)
    }

    var content: [Message.Content] {
      var merged = indexedContent
      for (index, summary) in summaryContent {
        let key = ContentKey(outputIndex: index, contentIndex: 0)
        if merged[key] == nil {
          merged[key] = .thinking(text: summary, signature: nil)
        }
      }
      var orderedContent = merged.keys.sorted().compactMap { merged[$0] } + fallbackContent
      if !orderedContent.contains(where: {
        if case .endnotes = $0 { return true }
        return false
      }), let endnotes = Self.formattedEndnotes(from: orderedContent) {
        orderedContent.append(.endnotes(endnotes))
      }
      return orderedContent
    }

    mutating func appendTextDelta(_ delta: String, outputIndex: Int?, contentIndex: Int?) {
      guard !delta.isEmpty else { return }
      updateTextBlock(outputIndex: outputIndex, contentIndex: contentIndex, createText: delta) { existingText in
        existingText + delta
      }
    }

    mutating func setFinalizedText(_ text: String, outputIndex: Int?, contentIndex: Int?) {
      updateTextBlock(outputIndex: outputIndex, contentIndex: contentIndex, createText: text) { _ in
        text
      }
    }

    mutating func setFinalizedReasoningText(_ text: String, outputIndex: Int?) {
      guard let outputIndex else {
        appendFallback(.thinking(text: text, signature: nil))
        return
      }
      let k = key(outputIndex: outputIndex, contentIndex: nil)
      indexedContent[k] = .thinking(text: text, signature: nil)
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

    mutating func addTextAnnotation(
      _ annotation: [String: Value],
      outputIndex: Int?,
      contentIndex: Int?,
      annotationIndex: Int?,
    ) {
      updateTextBlock(
        outputIndex: outputIndex,
        contentIndex: contentIndex,
        createText: "",
        transformText: { $0 },
        transformOpaque: { existingOpaque in
          var annotations = Self.decodeAnnotationsData(existingOpaque.data)
          if let annotationIndex, annotationIndex <= annotations.count {
            annotations.insert(annotation, at: annotationIndex)
          } else {
            annotations.append(annotation)
          }
          return .providerOpaque(OpaqueBlock(
            provider: "openai-responses",
            type: Self.annotatedOutputTextType,
            content: existingOpaque.content,
            data: Self.encodeAnnotationsData(annotations),
            isResponseContent: true,
          ))
        },
        transformTextBlock: { existingText in
          let annotations = [annotation]
          return .providerOpaque(OpaqueBlock(
            provider: "openai-responses",
            type: Self.annotatedOutputTextType,
            content: existingText,
            data: Self.encodeAnnotationsData(annotations),
            isResponseContent: true,
          ))
        },
      )
    }

    mutating func setToolCall(_ toolCall: ToolCall, outputIndex: Int?, itemId: String? = nil) {
      guard let outputIndex else {
        let index = fallbackContent.count
        appendFallback(.toolCall(toolCall))
        if let itemId {
          itemIdToFallbackIndex[itemId] = index
        }
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
        toolCall.parameters = [
          "_parseError": .string("Failed to parse arguments JSON"),
          "_rawArguments": .string(argumentsString),
        ]
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

    private mutating func updateTextBlock(
      outputIndex: Int?,
      contentIndex: Int?,
      createText: String,
      transformText: (String) -> String,
      transformOpaque: ((OpaqueBlock) -> Message.Content)? = nil,
      transformTextBlock: ((String) -> Message.Content)? = nil,
    ) {
      if let outputIndex {
        let k = key(outputIndex: outputIndex, contentIndex: contentIndex)
        indexedContent[k] = Self.updatedTextContent(
          existing: indexedContent[k],
          createText: createText,
          transformText: transformText,
          transformOpaque: transformOpaque,
          transformTextBlock: transformTextBlock,
        )
        return
      }

      if let fallbackIndex = fallbackContent.lastIndex(where: Self.isTextLikeContent(_:)) {
        fallbackContent[fallbackIndex] = Self.updatedTextContent(
          existing: fallbackContent[fallbackIndex],
          createText: createText,
          transformText: transformText,
          transformOpaque: transformOpaque,
          transformTextBlock: transformTextBlock,
        )
      } else {
        fallbackContent.append(Self.updatedTextContent(
          existing: nil,
          createText: createText,
          transformText: transformText,
          transformOpaque: transformOpaque,
          transformTextBlock: transformTextBlock,
        ))
      }
    }

    private static func updatedTextContent(
      existing: Message.Content?,
      createText: String,
      transformText: (String) -> String,
      transformOpaque: ((OpaqueBlock) -> Message.Content)?,
      transformTextBlock: ((String) -> Message.Content)?,
    ) -> Message.Content {
      switch existing {
        case let .text(existingText)?:
          let updatedText = transformText(existingText)
          if let transformTextBlock {
            return transformTextBlock(updatedText)
          }
          return .text(updatedText)
        case let .providerOpaque(opaque)? where opaque.provider == "openai-responses" && opaque.type == annotatedOutputTextType:
          if let transformOpaque {
            return transformOpaque(opaque)
          }
          return .providerOpaque(OpaqueBlock(
            provider: "openai-responses",
            type: annotatedOutputTextType,
            content: transformText(opaque.content ?? ""),
            data: opaque.data,
            isResponseContent: true,
          ))
        case let .providerOpaque(opaque)? where opaque.content != nil:
          let updatedText = transformText(opaque.content ?? "")
          if let transformTextBlock {
            return transformTextBlock(updatedText)
          }
          return .text(updatedText)
        default:
          if let transformTextBlock {
            return transformTextBlock(createText)
          }
          return .text(createText)
      }
    }

    private static func isTextLikeContent(_ block: Message.Content) -> Bool {
      switch block {
        case .text:
          true
        case let .providerOpaque(opaque):
          opaque.provider == "openai-responses" && opaque.type == annotatedOutputTextType
        default:
          false
      }
    }

    private static func decodeAnnotationsData(_ data: String?) -> [[String: Value]] {
      guard let data,
            let jsonData = data.data(using: .utf8),
            let value = try? Value.fromData(jsonData),
            let annotations = value.arrayValue?.compactMap(\.objectValue)
      else {
        return []
      }
      return annotations
    }

    private static func encodeAnnotationsData(_ annotations: [[String: Value]]) -> String? {
      guard let data = try? JSONSerialization.data(withJSONObject: annotations.map(Value.toSendable)),
            let jsonString = String(data: data, encoding: .utf8)
      else {
        return nil
      }
      return jsonString
    }

    private static func formattedEndnotes(from blocks: [Message.Content]) -> String? {
      let citations = blocks.reduce(into: [(label: String, url: String?, fileId: String?)]()) { result, block in
        guard case let .providerOpaque(opaque) = block,
              opaque.provider == "openai-responses",
              opaque.type == annotatedOutputTextType
        else {
          return
        }

        for annotation in decodeAnnotationsData(opaque.data) {
          switch annotation["type"]?.stringValue {
            case "url_citation":
              if let url = annotation["url"]?.stringValue {
                result.append((label: annotation["title"]?.stringValue ?? url, url: url, fileId: nil))
              }
            case "file_citation", "container_file_citation":
              if let filename = annotation["filename"]?.stringValue {
                result.append((label: filename, url: nil, fileId: annotation["file_id"]?.stringValue))
              }
            default:
              break
          }
        }
      }

      guard !citations.isEmpty else { return nil }

      let uniqueCitations = citations.reduce(into: [(label: String, url: String?, fileId: String?)]()) { result, citation in
        let key = citation.url ?? citation.fileId ?? citation.label
        if !result.contains(where: { ($0.url ?? $0.fileId ?? $0.label) == key }) {
          result.append(citation)
        }
      }

      return uniqueCitations.map { citation in
        if let url = citation.url {
          "- [\(citation.label)](\(url))"
        } else {
          "- \(citation.label)"
        }
      }.joined(separator: "\n") + "\n"
    }
  }

  struct AccumulatedResponseSnapshot {
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
        case StreamEventType.outputTextDone:
          if let text = event.text {
            setContentField(
              text,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              defaultOutputType: OutputItemType.message,
              partType: OutputItemType.outputText,
              fieldName: "text",
            )
          }
        case StreamEventType.outputTextAnnotationAdded:
          if let annotation = event.annotation?.raw {
            addAnnotation(
              annotation,
              outputIndex: event.outputIndex,
              contentIndex: event.contentIndex,
              annotationIndex: event.annotationIndex,
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
        case StreamEventType.reasoningTextDone:
          if let text = event.text {
            setContentField(
              text,
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

    private mutating func addAnnotation(
      _ annotation: [String: Value],
      outputIndex: Int?,
      contentIndex: Int?,
      annotationIndex: Int?,
    ) {
      let resolvedOutputIndex = resolveOutputIndex(explicitIndex: outputIndex, defaultType: OutputItemType.message)
      guard outputs.indices.contains(resolvedOutputIndex) else { return }

      var output = outputs[resolvedOutputIndex]
      var content = output["content"]?.arrayValue ?? []
      let resolvedContentIndex = Self.resolveContentIndex(in: content, requested: contentIndex, preferredType: OutputItemType.outputText)

      while content.count <= resolvedContentIndex {
        content.append(.object(Self.syntheticContentPart(ofType: OutputItemType.outputText)))
      }

      var part = content[resolvedContentIndex].objectValue ?? Self.syntheticContentPart(ofType: OutputItemType.outputText)
      part["type"] = .string(OutputItemType.outputText)

      var annotations = part["annotations"]?.arrayValue ?? []
      if let annotationIndex, annotationIndex <= annotations.count {
        annotations.insert(.object(annotation), at: annotationIndex)
      } else {
        annotations.append(.object(annotation))
      }
      part["annotations"] = .array(annotations)

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

  enum StreamEventType {
    static let outputTextDelta = "response.output_text.delta"
    static let outputTextDone = "response.output_text.done"
    static let outputTextAnnotationAdded = "response.output_text.annotation.added"
    static let reasoningTextDelta = "response.reasoning_text.delta"
    static let reasoningTextDone = "response.reasoning_text.done"
    static let reasoningDelta = "response.reasoning.delta"
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

  enum OutputItemType {
    static let functionCall = "function_call"
    static let reasoning = "reasoning"
    static let codeInterpreterCall = "code_interpreter_call"
    static let webSearchCall = "web_search_call"
    static let message = "message"
    static let outputText = "output_text"
    static let refusal = "refusal"
  }
}
