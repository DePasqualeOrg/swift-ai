// Copyright © Anthony DePasquale

@testable import AI
import AITool
import Foundation
import Testing

private func canonicalJSONString(_ jsonObject: Any) throws -> String {
  let data = try JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys])
  guard let string = String(data: data, encoding: .utf8) else {
    throw CocoaError(.fileReadInapplicableStringEncoding)
  }
  return string
}

private func strictJSONObject(_ schema: [String: Value]) throws -> [String: Any] {
  let normalized = try Value.schemaForStrictMode(schema)
  let data = try JSONSerialization.data(withJSONObject: normalized, options: [.sortedKeys])
  guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
    throw CocoaError(.coderReadCorrupt)
  }
  return object
}

struct ArchitectureInvariantTests {
  @Test
  func `Generation response preserves visible text order across text and opaque response content`() {
    let response = GenerationResponse(content: [
      .text("Alpha"),
      .providerOpaque(OpaqueBlock(
        provider: "openai-responses",
        type: "annotated_output_text",
        content: "Beta",
        isResponseContent: true,
      )),
      .providerOpaque(OpaqueBlock(
        provider: "gemini",
        type: "urlContextMetadata",
        content: "Dropped",
        isResponseContent: false,
      )),
      .text("Gamma"),
    ])

    #expect(response.responseText == "AlphaBetaGamma")
  }

  @Test
  func `Generation response preserves reasoning order across native and opaque thinking content`() {
    let response = GenerationResponse(content: [
      .thinking(text: "First", signature: nil),
      .providerOpaque(OpaqueBlock(
        provider: "gemini",
        type: "thinking",
        content: "Second",
      )),
      .thinking(text: "Third", signature: nil),
    ])

    #expect(response.reasoningText == "FirstSecondThird")
  }

  @Test
  func `Imperative tool strict schema matches macro schema for equivalent optional parameter tool`() throws {
    let imperativeTool = Tool(
      name: "get_weather",
      description: "Get weather for a city",
      parameters: [
        .string("city", description: "City name"),
        .string("unit", description: "Temperature unit", required: false),
      ],
    ) { _ in
      [.text("ok")]
    }

    let imperativeStrict = try canonicalJSONString(Value.schemaForStrictMode(imperativeTool.rawInputSchema))
    let macroStrict = try canonicalJSONString(Value.toSendable(GetWeather.tool.rawInputSchema))

    #expect(imperativeStrict == macroStrict)
  }

  @Test
  func `Imperative tool strict schema matches macro schema for equivalent optional enum tool`() throws {
    let imperativeTool = Tool(
      name: "set_optional_priority",
      description: "Set task priority with an optional override",
      parameters: [
        .string("priority", description: "Priority override", required: false, enum: ["low", "medium", "high"]),
      ],
    ) { _ in
      [.text("ok")]
    }

    let imperativeStrict = try canonicalJSONString(Value.schemaForStrictMode(imperativeTool.rawInputSchema))
    let macroStrict = try canonicalJSONString(Value.toSendable(SetOptionalPriority.tool.rawInputSchema))

    #expect(imperativeStrict == macroStrict)
  }

  @Test
  func `Manual defaulted schema strict normalization matches macro schema`() throws {
    let preStrictSchema: [String: Value] = [
      "type": "object",
      "properties": [
        "query": [
          "type": "string",
          "description": "Search query",
          "minLength": 1,
          "maxLength": 500,
        ],
        "limit": [
          "type": ["integer", "null"],
          "description": "Maximum results",
          "minimum": .double(1),
          "maximum": .double(100),
          "default": 10,
        ],
      ],
      "required": ["query"],
    ]

    let normalized = try canonicalJSONString(Value.schemaForStrictMode(preStrictSchema))
    let macroStrict = try canonicalJSONString(Value.schemaForStrictMode(SearchDocuments.tool.rawInputSchema))

    #expect(normalized == macroStrict)
  }

  @Test
  func `Strict mode rejects optional non nullable field`() {
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "city": ["type": "string"],
        "unit": ["type": "string"],
      ],
      "required": ["city"],
    ]

    #expect(throws: AIError.self) {
      _ = try Value.schemaForStrictMode(schema)
    }
  }

  @Test
  func `Strict mode accepts optional non nullable field with default value`() throws {
    // A property with a default is safe in strict mode — OpenAI fills in the
    // default when the model omits the field. Matches the OpenAI TS SDK's
    // zod-to-json-schema behavior (skips the optional-without-nullable throw
    // when `defaultValue` is present).
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "query": ["type": "string"],
        "limit": [
          "type": "integer",
          "default": 20,
        ],
      ],
      "required": ["query"],
    ]

    let normalized = try strictJSONObject(schema)
    let required = try #require(normalized["required"] as? [String])
    #expect(required.sorted() == ["limit", "query"])

    let properties = try #require(normalized["properties"] as? [String: Any])
    let limit = try #require(properties["limit"] as? [String: Any])
    #expect(limit["default"] as? Int == 20)
    #expect(limit["type"] as? String == "integer")
  }

  @Test
  func `Strict mode collapses single-element allOf into the parent`() throws {
    // Matches the OpenAI TS SDK: when `allOf` has exactly one variant, the
    // variant's keys merge into the parent and `allOf` is dropped. This is
    // how zod-to-json-schema composes refs — a lone `allOf` wrapper is
    // structural noise that strict-mode normalization should remove.
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "value": .object([
          "allOf": .array([
            .object([
              "type": "object",
              "properties": .object(["name": .object(["type": "string"])]),
              "required": .array([.string("name")]),
            ]),
          ]),
        ]),
      ],
      "required": ["value"],
    ]

    let normalized = try strictJSONObject(schema)
    let properties = try #require(normalized["properties"] as? [String: Any])
    let value = try #require(properties["value"] as? [String: Any])
    #expect(value["allOf"] == nil)
    #expect(value["type"] as? String == "object")
    #expect(value["additionalProperties"] as? Bool == false)

    let valueProperties = try #require(value["properties"] as? [String: Any])
    let name = try #require(valueProperties["name"] as? [String: Any])
    #expect(name["type"] as? String == "string")

    let valueRequired = try #require(value["required"] as? [String])
    #expect(valueRequired == ["name"])
  }

  @Test
  func `Strict mode preserves multi-element allOf and normalizes each variant`() throws {
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "value": .object([
          "allOf": .array([
            .object([
              "type": "object",
              "properties": .object(["a": .object(["type": "string"])]),
              "required": .array([.string("a")]),
            ]),
            .object([
              "type": "object",
              "properties": .object(["b": .object(["type": "integer"])]),
              "required": .array([.string("b")]),
            ]),
          ]),
        ]),
      ],
      "required": ["value"],
    ]

    let normalized = try strictJSONObject(schema)
    let properties = try #require(normalized["properties"] as? [String: Any])
    let value = try #require(properties["value"] as? [String: Any])
    let allOf = try #require(value["allOf"] as? [[String: Any]])
    #expect(allOf.count == 2)
    #expect(allOf[0]["additionalProperties"] as? Bool == false)
    #expect(allOf[1]["additionalProperties"] as? Bool == false)
  }

  @Test
  func `Strict mode strips default null from property schemas`() throws {
    // `default: null` carries no information in strict mode — OpenAI's
    // server-side default injection can't distinguish it from "no default",
    // and keeping it around invites confusing validator output.
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "tag": .object([
          "type": .array([.string("string"), .string("null")]),
          "default": .null,
        ]),
      ],
      "required": ["tag"],
    ]

    let normalized = try strictJSONObject(schema)
    let properties = try #require(normalized["properties"] as? [String: Any])
    let tag = try #require(properties["tag"] as? [String: Any])
    #expect(tag["default"] == nil)
    #expect(tag["type"] as? [String] == ["string", "null"])
  }

  @Test
  func `Strict mode makes nullable properties required and adds additionalProperties false recursively`() throws {
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "settings": [
          "type": ["object", "null"],
          "properties": [
            "mode": ["type": "string"],
          ],
          "required": ["mode"],
        ],
      ],
    ]

    let normalized = try strictJSONObject(schema)
    let required = try #require(normalized["required"] as? [String])
    #expect(required == ["settings"])
    #expect(normalized["additionalProperties"] as? Bool == false)

    let properties = try #require(normalized["properties"] as? [String: Any])
    let settings = try #require(properties["settings"] as? [String: Any])
    let settingsRequired = try #require(settings["required"] as? [String])
    #expect(settingsRequired == ["mode"])
    #expect(settings["additionalProperties"] as? Bool == false)
  }

  @Test
  func `Generated strict schema reuses the shared normalizer for nested object parameters`() throws {
    let descriptor = ToolMacroSupport.SchemaParameterDescriptor(
      name: "settings",
      title: nil,
      description: "Settings object",
      jsonSchemaType: "object",
      jsonSchemaProperties: [
        "properties": .object([
          "mode": .object(["type": "string"]),
        ]),
        "required": .array([.string("mode")]),
      ],
      isOptional: false,
      hasDefault: false,
      defaultValue: nil,
      minLength: nil,
      maxLength: nil,
      minimum: nil,
      maximum: nil,
    )

    let strictSchema = try ToolMacroSupport.buildObjectSchema(parameters: [descriptor], strict: true)
    let normalizedFromBase = try ToolSchema.normalizeForStrictMode(
      ToolMacroSupport.buildObjectSchema(parameters: [descriptor], strict: false),
    )

    #expect(strictSchema == normalizedFromBase)

    let settings = try #require(strictSchema["properties"]?.objectValue?["settings"]?.objectValue)
    #expect(settings["additionalProperties"]?.boolValue == false)
    #expect(settings["required"]?.arrayValue == [.string("mode")])
  }

  @Test
  func `Throwing macro schema builder reports invalid strict schema without crashing`() {
    let descriptor = ToolMacroSupport.SchemaParameterDescriptor(
      name: "settings",
      description: "Settings object",
      jsonSchemaType: "object",
      jsonSchemaProperties: [
        "properties": .object([
          "mode": .object(["type": "string"]),
        ]),
      ],
      isOptional: false,
    )

    #expect(throws: AIError.self) {
      _ = try ToolMacroSupport.buildObjectSchema(parameters: [descriptor], strict: true)
    }

    let result = ToolMacroSupport.buildObjectSchemaResult(parameters: [descriptor], strict: true)
    #expect(result.errorMessage != nil)
    #expect(result.schema["type"] == .string("object"))
  }

  @Test
  func `Schema builder rejects duplicate parameter names without crashing`() {
    let first = ToolMacroSupport.SchemaParameterDescriptor(
      name: "query",
      description: "First query",
      jsonSchemaType: "string",
      isOptional: false,
    )
    let second = ToolMacroSupport.SchemaParameterDescriptor(
      name: "query",
      description: "Second query",
      jsonSchemaType: "string",
      isOptional: false,
    )

    #expect(throws: AIError.self) {
      _ = try ToolMacroSupport.buildObjectSchema(parameters: [first, second], strict: false)
    }

    let result = ToolMacroSupport.buildObjectSchemaResult(parameters: [first, second], strict: false)
    #expect(result.errorMessage?.contains("Duplicate parameter name") == true)
    #expect(result.schema["properties"]?.objectValue?.isEmpty == true)
  }

  @Test
  func `Strict mode resolves refs with sibling keywords before normalization`() throws {
    let schema: [String: Value] = [
      "type": "object",
      "properties": [
        "config": [
          "$ref": "#/$defs/Config",
          "description": "Configuration object",
        ],
      ],
      "required": ["config"],
      "$defs": [
        "Config": [
          "type": "object",
          "properties": [
            "mode": ["type": "string"],
          ],
          "required": ["mode"],
        ],
      ],
    ]

    let normalized = try strictJSONObject(schema)
    let properties = try #require(normalized["properties"] as? [String: Any])
    let config = try #require(properties["config"] as? [String: Any])
    let configProperties = try #require(config["properties"] as? [String: Any])

    #expect(config["description"] as? String == "Configuration object")
    #expect(config["type"] as? String == "object")
    #expect(config["additionalProperties"] as? Bool == false)
    #expect(configProperties["mode"] != nil)
  }
}
