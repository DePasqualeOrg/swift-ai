// Copyright © Anthony DePasquale

import Foundation

extension AnthropicClient {
  struct MessageParam {
    let role: Role
    let text: String?
    let contentBlocks: [ContentBlockParam]?
    let attachments: [Attachment]?

    init(role: Role, text: String?, contentBlocks: [ContentBlockParam]? = nil, attachments: [Attachment]? = nil) {
      self.role = role
      self.text = text
      self.contentBlocks = contentBlocks
      self.attachments = attachments
    }
  }

  struct ContentBlockParam: Codable {
    let type: ContentBlockType
    let text: String?
    let source: ContentBlockSource?
    let toolUse: ToolUseBlockParam?
    let toolResult: ToolResultBlockParam?
    let codeExecutionToolResult: CodeExecutionToolResultBlockParam?
    let thinking: String?
    let signature: String?
    let data: String?
    var rawValue: Value? = nil

    enum CodingKeys: String, CodingKey {
      case type, text, source, toolUse, toolResult, codeExecutionToolResult, thinking, signature, data
    }

    init(
      type: ContentBlockType,
      text: String? = nil,
      source: ContentBlockSource? = nil,
      toolUse: ToolUseBlockParam? = nil,
      toolResult: ToolResultBlockParam? = nil,
      codeExecutionToolResult: CodeExecutionToolResultBlockParam? = nil,
      thinking: String? = nil,
      signature: String? = nil,
      data: String? = nil,
      rawValue: Value? = nil,
    ) {
      self.type = type
      self.text = text
      self.source = source
      self.toolUse = toolUse
      self.toolResult = toolResult
      self.codeExecutionToolResult = codeExecutionToolResult
      self.thinking = thinking
      self.signature = signature
      self.data = data
      self.rawValue = rawValue
    }
  }

  struct ToolUseBlockParam: Codable {
    let id: String
    let name: String
    let input: Value
  }

  struct ToolResultBlockParam: Codable {
    let toolUseId: String
    let content: ToolResultContent
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
      case isError = "is_error"
    }
  }

  enum ToolResultContent: Codable {
    case text(String)
    case blocks([ToolResultContentBlock])

    func encode(to encoder: Encoder) throws {
      var container = encoder.singleValueContainer()
      switch self {
        case let .text(string):
          try container.encode(string)
        case let .blocks(blocks):
          try container.encode(blocks)
      }
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      if let string = try? container.decode(String.self) {
        self = .text(string)
      } else if let blocks = try? container.decode([ToolResultContentBlock].self) {
        self = .blocks(blocks)
      } else {
        throw DecodingError.typeMismatch(
          ToolResultContent.self,
          DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected String or Array"),
        )
      }
    }
  }

  struct ToolResultContentBlock: Codable {
    let raw: [String: Value]

    init(from decoder: Decoder) throws {
      let container = try decoder.singleValueContainer()
      raw = try container.decode([String: Value].self)
    }

    init(raw: [String: Value]) {
      self.raw = raw
    }

    func encode(to encoder: Encoder) throws {
      var container = encoder.singleValueContainer()
      try container.encode(raw)
    }

    var type: String {
      raw["type"]?.stringValue ?? "text"
    }

    var text: String? {
      raw["text"]?.stringValue
    }

    var source: ContentBlockSource? {
      raw["source"]?.objectValue.map(ContentBlockSource.init(raw:))
    }

    var searchResultSource: String? {
      raw["source"]?.stringValue
    }

    var title: String? {
      raw["title"]?.stringValue
    }

    var contentBlocks: [ToolResultContentBlock]? {
      raw["content"]?.arrayValue?.compactMap(\.objectValue).map(ToolResultContentBlock.init(raw:))
    }

    var toolName: String? {
      raw["tool_name"]?.stringValue
    }

    static func text(_ text: String) -> ToolResultContentBlock {
      ToolResultContentBlock(raw: [
        "type": .string("text"),
        "text": .string(text),
      ])
    }

    static func image(mediaType: String, data: String) -> ToolResultContentBlock {
      ToolResultContentBlock(raw: [
        "type": .string("image"),
        "source": .object([
          "type": .string("base64"),
          "media_type": .string(mediaType),
          "data": .string(data),
        ]),
      ])
    }

    static func document(mediaType: String, data: String, sourceType: String = "base64") -> ToolResultContentBlock {
      ToolResultContentBlock(raw: [
        "type": .string("document"),
        "source": .object([
          "type": .string(sourceType),
          "media_type": .string(mediaType),
          "data": .string(data),
        ]),
      ])
    }
  }

  struct CodeExecutionToolResultBlockParam: Codable {
    let toolUseId: String
    let content: Value

    enum CodingKeys: String, CodingKey {
      case toolUseId = "tool_use_id"
      case content
    }
  }

  struct MessageCreateParams {
    let model: String
    let messages: [MessageParam]
    let maxTokens: Int?
    let system: String?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    var tools: [APITool]?
    var toolChoice: ToolChoice?
    let metadata: [String: String]?
    let thinking: ThinkingConfig?
    let effort: EffortLevel?
    let disableParallelToolUse: Bool?

    init(
      model: String,
      messages: [MessageParam],
      maxTokens: Int? = nil,
      system: String? = nil,
      temperature: Float? = nil,
      topP: Float? = nil,
      topK: Int? = nil,
      tools: [APITool]? = nil,
      toolChoice: ToolChoice? = nil,
      metadata: [String: String]? = nil,
      thinking: ThinkingConfig? = nil,
      effort: EffortLevel? = nil,
      disableParallelToolUse: Bool? = nil,
    ) {
      self.model = model
      self.messages = messages
      self.maxTokens = maxTokens
      self.system = system
      self.temperature = temperature
      self.topP = topP
      self.topK = topK
      self.tools = tools
      self.toolChoice = toolChoice
      self.metadata = metadata
      self.thinking = thinking
      self.effort = effort
      self.disableParallelToolUse = disableParallelToolUse
    }
  }

  struct JSONSchema: Codable {
    let type: String
    let properties: [String: JSONSchemaProperty]?
    let required: [String]?

    init(type: String, properties: [String: JSONSchemaProperty]? = nil, required: [String]? = nil) {
      self.type = type
      self.properties = properties
      self.required = required
    }
  }

  final class JSONSchemaProperty: Codable, Sendable {
    let type: String
    let description: String?
    let enumValues: [String]?
    let items: JSONSchemaProperty?

    enum CodingKeys: String, CodingKey {
      case type
      case description
      case enumValues = "enum"
      case items
    }

    init(type: String, description: String? = nil, enumValues: [String]? = nil, items: JSONSchemaProperty? = nil) {
      self.type = type
      self.description = description
      self.enumValues = enumValues
      self.items = items
    }

    static func from(_ paramType: Tool.ParameterType, description: String, enumValues: [String]? = nil) -> JSONSchemaProperty {
      switch paramType {
        case .string:
          JSONSchemaProperty(type: "string", description: description, enumValues: enumValues)
        case .float, .integer:
          JSONSchemaProperty(type: "number", description: description)
        case .boolean:
          JSONSchemaProperty(type: "boolean", description: description)
        case let .array(itemType):
          JSONSchemaProperty(type: "array", description: description, items: from(itemType, description: ""))
        case .object:
          JSONSchemaProperty(type: "object", description: description)
      }
    }
  }

  enum ToolChoice: Codable {
    case auto
    case any
    case none
    case tool(name: String)

    func encode(to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)

      switch self {
        case .auto:
          try container.encode("auto", forKey: .type)
        case .any:
          try container.encode("any", forKey: .type)
        case .none:
          try container.encode("none", forKey: .type)
        case let .tool(name):
          try container.encode("tool", forKey: .type)
          try container.encode(name, forKey: .name)
      }
    }

    enum CodingKeys: String, CodingKey {
      case type, name
    }

    init(from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      let type = try container.decode(String.self, forKey: .type)

      switch type {
        case "auto":
          self = .auto
        case "any":
          self = .any
        case "none":
          self = .none
        case "tool":
          let name = try container.decode(String.self, forKey: .name)
          self = .tool(name: name)
        default:
          throw DecodingError.dataCorruptedError(
            forKey: .type,
            in: container,
            debugDescription: "Invalid tool choice type: \(type)",
          )
      }
    }
  }

  enum APITool: Codable {
    case custom(name: String, description: String, inputSchema: JSONSchema)
    case rawCustom(name: String, description: String, rawInputSchema: [String: Value])
    case webSearch
    case webFetch
    case codeExecution

    private enum CustomCodingKeys: String, CodingKey {
      case name, description
      case inputSchema = "input_schema"
    }

    private enum WebSearchCodingKeys: String, CodingKey {
      case name, type
    }

    private enum WebFetchCodingKeys: String, CodingKey {
      case name, type
    }

    private enum CodeExecutionCodingKeys: String, CodingKey {
      case name, type
    }

    func encode(to encoder: Encoder) throws {
      switch self {
        case let .custom(name, description, inputSchema):
          var container = encoder.container(keyedBy: CustomCodingKeys.self)
          try container.encode(name, forKey: .name)
          try container.encode(description, forKey: .description)
          try container.encode(inputSchema, forKey: .inputSchema)
        case let .rawCustom(name, description, rawInputSchema):
          var container = encoder.container(keyedBy: CustomCodingKeys.self)
          try container.encode(name, forKey: .name)
          try container.encode(description, forKey: .description)
          try container.encode(Self.transformRawInputSchema(rawInputSchema), forKey: .inputSchema)
        case .webSearch:
          var container = encoder.container(keyedBy: WebSearchCodingKeys.self)
          try container.encode("web_search", forKey: .name)
          try container.encode("web_search_20250305", forKey: .type)
        case .webFetch:
          var container = encoder.container(keyedBy: WebFetchCodingKeys.self)
          try container.encode("web_fetch", forKey: .name)
          try container.encode("web_fetch_20250910", forKey: .type)
        case .codeExecution:
          var container = encoder.container(keyedBy: CodeExecutionCodingKeys.self)
          try container.encode("code_execution", forKey: .name)
          try container.encode("code_execution_20250522", forKey: .type)
      }
    }

    init(from decoder: Decoder) throws {
      let webSearchContainer = try? decoder.container(keyedBy: WebSearchCodingKeys.self)
      if let name = try? webSearchContainer?.decode(String.self, forKey: .name),
         let type = try? webSearchContainer?.decode(String.self, forKey: .type),
         name == "web_search", type.hasPrefix("web_search_")
      {
        self = .webSearch
        return
      }

      let webFetchContainer = try? decoder.container(keyedBy: WebFetchCodingKeys.self)
      if let name = try? webFetchContainer?.decode(String.self, forKey: .name),
         let type = try? webFetchContainer?.decode(String.self, forKey: .type),
         name == "web_fetch", type.hasPrefix("web_fetch_")
      {
        self = .webFetch
        return
      }

      let codeExecutionContainer = try? decoder.container(keyedBy: CodeExecutionCodingKeys.self)
      if let name = try? codeExecutionContainer?.decode(String.self, forKey: .name),
         let type = try? codeExecutionContainer?.decode(String.self, forKey: .type),
         name == "code_execution", type.hasPrefix("code_execution_")
      {
        self = .codeExecution
        return
      }

      let customContainer = try decoder.container(keyedBy: CustomCodingKeys.self)
      let name = try customContainer.decode(String.self, forKey: .name)
      let description = try customContainer.decode(String.self, forKey: .description)
      let inputSchema = try customContainer.decode(JSONSchema.self, forKey: .inputSchema)
      self = .custom(name: name, description: description, inputSchema: inputSchema)
    }

    private static func transformRawInputSchema(_ schema: [String: Value]) -> [String: Value] {
      transformAnthropicSchema(.object(schema)).objectValue ?? schema
    }

    private static func transformAnthropicSchema(_ value: Value) -> Value {
      guard case let .object(schema) = value else { return value }

      if let ref = schema["$ref"] {
        return .object(["$ref": ref])
      }

      var remaining = schema
      var transformed: [String: Value] = [:]

      if let defs = remaining.removeValue(forKey: "$defs") {
        if case let .object(definitions) = defs {
          transformed["$defs"] = .object(definitions.mapValues(transformAnthropicSchema))
        } else {
          transformed["$defs"] = defs
        }
      }

      let type = remaining.removeValue(forKey: "type")
      let anyOf = remaining.removeValue(forKey: "anyOf")
      let oneOf = remaining.removeValue(forKey: "oneOf")
      let allOf = remaining.removeValue(forKey: "allOf")

      if let variants = anyOf?.arrayValue {
        transformed["anyOf"] = .array(variants.map(transformAnthropicSchema))
      } else if let variants = oneOf?.arrayValue {
        transformed["anyOf"] = .array(variants.map(transformAnthropicSchema))
      } else if let variants = allOf?.arrayValue {
        transformed["allOf"] = .array(variants.map(transformAnthropicSchema))
      } else if let type {
        transformed["type"] = type
      }

      if let description = remaining.removeValue(forKey: "description") {
        transformed["description"] = description
      }

      if let title = remaining.removeValue(forKey: "title") {
        transformed["title"] = title
      }

      if type?.stringValue == "object" {
        let properties = remaining.removeValue(forKey: "properties")?.objectValue ?? [:]
        transformed["properties"] = .object(properties.mapValues(transformAnthropicSchema))

        _ = remaining.removeValue(forKey: "additionalProperties")
        transformed["additionalProperties"] = false

        if let required = remaining.removeValue(forKey: "required") {
          transformed["required"] = required
        }
      } else if type?.stringValue == "string" {
        if let format = remaining["format"],
           let formatString = format.stringValue,
           supportedStringFormats.contains(formatString)
        {
          _ = remaining.removeValue(forKey: "format")
          transformed["format"] = .string(formatString)
        }
      } else if type?.stringValue == "array" {
        if let items = remaining.removeValue(forKey: "items") {
          transformed["items"] = transformAnthropicSchema(items)
        }

        if let minItems = remaining["minItems"],
           let count = minItems.intValue,
           count == 0 || count == 1
        {
          _ = remaining.removeValue(forKey: "minItems")
          transformed["minItems"] = .int(count)
        }
      }

      if !remaining.isEmpty {
        let supplemental = remaining
          .map { key, value in "\(key): \(value.stringRepresentationForJSON)" }
          .sorted()
          .joined(separator: ", ")
        let existingDescription = transformed["description"]?.stringValue
        let combinedDescription = ((existingDescription.map { "\($0)\n\n" }) ?? "") + "{\(supplemental)}"
        transformed["description"] = .string(combinedDescription)
      }

      return .object(transformed)
    }

    private static let supportedStringFormats: Set<String> = [
      "date-time", "time", "date", "duration", "email",
      "hostname", "uri", "ipv4", "ipv6", "uuid",
    ]
  }
}
