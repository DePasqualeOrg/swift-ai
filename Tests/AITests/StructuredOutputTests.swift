// Copyright © Anthony DePasquale

@testable import AI
import AITool
import Foundation
import JSONSchemaBuilder
import Testing

// MARK: - Test types

@Schemable
@StructuredOutput
struct UserInfo {
  let id: String
  let displayName: String?
}

@Schemable
@StructuredOutput
struct WidgetMeta {
  let width: Int
}

@Schemable
@StructuredOutput
struct LipSyncMeta {
  let id: String
}

@Tool
struct GetUser {
  static let name = "get_user"
  static let description = "Look up a user"
  @Parameter(description: "User id") var id: String

  func perform() async throws -> UserInfo {
    UserInfo(id: id, displayName: "Anonymous")
  }
}

@Tool
struct CountItems {
  static let name = "count_items"
  static let description = "Count items"
  @Parameter(description: "Query") var query: String

  func perform() async throws -> Int {
    query.count
  }
}

@Tool
struct GetTags {
  static let name = "get_tags"
  static let description = "Get tags as a map"

  func perform() async throws -> [String: Int] {
    ["alice": 42, "bob": 17]
  }
}

@Tool
struct Ping {
  static let name = "ping"
  static let description = "Touch the server"

  func perform() async throws {}
}

@Schemable
@StructuredOutput
struct ScreenshotMetadata {
  let width: Int
  let height: Int
}

@Tool
struct TakeScreenshot {
  static let name = "take_screenshot"
  static let description = "Capture the current screen with size metadata"

  func perform() async throws -> ImageWithMetadata<ScreenshotMetadata> {
    ImageWithMetadata(
      Data([0x89, 0x50, 0x4E, 0x47]),
      mimeType: "image/png",
      metadata: ScreenshotMetadata(width: 1024, height: 768),
    )
  }
}

// MARK: - Tests

struct StructuredOutputTests {
  // MARK: Primitive wire shapes (§3, §G primitive table)

  @Test
  func `Int primitive populates content text and structuredContent result wrap`() throws {
    let output = try (42 as Int).toToolResult()
    #expect(output.content.count == 1)
    if case let .text(text) = output.content[0] {
      #expect(text == "42")
    } else {
      Issue.record("Expected .text content")
    }
    #expect(output.structuredContent == .object(["result": .int(42)]))
  }

  @Test
  func `String primitive routes through PrimitiveToolOutput with both channels`() throws {
    let output = try "hello".toToolResult()
    #expect(output.content == [.text("hello")])
    #expect(output.structuredContent == .object(["result": .string("hello")]))
  }

  @Test
  func `Optional none populates result null`() throws {
    let value: Int? = nil
    let output = try value.toToolResult()
    #expect(output.content == [.text("null")])
    #expect(output.structuredContent == .object(["result": .null]))
  }

  @Test
  func `Optional some Int renders identical display to bare Int`() throws {
    let bare = try (42 as Int).toToolResult()
    let some: Int? = 42
    let wrapped = try some.toToolResult()
    #expect(bare.content == wrapped.content)
    #expect(bare.structuredContent == wrapped.structuredContent)
  }

  @Test
  func `Dictionary emits unwrapped top-level object`() throws {
    let output = try (["alice": 42, "bob": 17] as [String: Int]).toToolResult()
    // Display text: pretty-printed JSON
    if case let .text(text) = output.content[0] {
      #expect(text.contains("alice"))
      #expect(text.contains("42"))
    } else {
      Issue.record("Expected .text content")
    }
    // Structured channel: raw map, no "result" wrap
    #expect(output.structuredContent == .object(["alice": .int(42), "bob": .int(17)]))
  }

  // MARK: VoidOutput (§3, §C)

  @Test
  func `VoidOutput byte-equals primitive null wrap shape`() throws {
    let output = try VoidOutput().toToolResult()
    #expect(output.content == [.text("null")])
    #expect(output.structuredContent == .object(["result": .null]))
  }

  @Test
  func `VoidOutput outputJSONSchema matches primitive null wrap`() {
    // Pin the schema so a future refactor that re-routes Void through the
    // primitive-wrap path can't silently change the wire schema (§C, §10).
    let expected: Value = .object([
      "type": .string("object"),
      "properties": .object(["result": .object(["type": .string("null")])]),
      "required": .array([.string("result")]),
      "additionalProperties": .bool(false),
    ])
    #expect(VoidOutput.outputJSONSchema == expected)
  }

  // MARK: AISchema dispatcher (§3, §C)

  @Test
  func `AISchema dispatches StructuredOutput to outputJSONSchema unwrapped`() {
    let schema = AISchema.outputSchema(for: UserInfo.self)
    #expect(schema == UserInfo.outputJSONSchema)
  }

  @Test
  func `AISchema dispatches PrimitiveToolOutput to wrapped result schema`() {
    let schema = AISchema.outputSchema(for: Int.self)
    #expect(schema == .object([
      "type": .string("object"),
      "properties": .object(["result": .object(["type": .string("integer")])]),
      "required": .array([.string("result")]),
      "additionalProperties": .bool(false),
    ]))
  }

  @Test
  func `AISchema dispatches Dictionary to unwrapped valueSchema`() {
    let schema = AISchema.outputSchema(for: [String: Int].self)
    #expect(schema == .object([
      "type": .string("object"),
      "additionalProperties": .object(["type": .string("integer")]),
    ]))
  }

  @Test
  func `AISchema returns nil for opaque ToolOutput conformer`() {
    #expect(AISchema.outputSchema(for: MultiContent.self) == nil)
  }

  // MARK: @StructuredOutput macro round-trip (§2)

  @Test
  func `StructuredOutput struct emits content text and structured channel`() throws {
    let info = UserInfo(id: "u-1", displayName: nil)
    let output = try info.toToolResult()
    #expect(output.content.count == 1)
    if case let .text(text) = output.content[0] {
      #expect(text.contains("\"id\":\"u-1\""))
      // Optional emits as explicit null (stable-shape contract)
      #expect(text.contains("\"displayName\":null"))
    } else {
      Issue.record("Expected .text content")
    }
    #expect(output.structuredContent == .object([
      "id": .string("u-1"),
      "displayName": .null,
    ]))
  }

  @Test
  func `StructuredOutput required list includes optional fields`() {
    // The structuredOutputSchemaDictionary post-processes @Schemable's output
    // so every property appears in `required`, including optionals. This
    // matches the wire contract where optionals emit as explicit null.
    if case let .object(schema) = UserInfo.outputJSONSchema,
       case let .array(required) = schema["required"]
    {
      let names = required.compactMap { $0.stringValue }.sorted()
      #expect(names == ["displayName", "id"])
    } else {
      Issue.record("Expected outputJSONSchema with required array")
    }
  }

  // MARK: @Tool outputSchema population (§6)

  @Test
  func `Tool returning StructuredOutput publishes outputSchema`() {
    #expect(GetUser.tool.outputSchema == UserInfo.outputJSONSchema)
  }

  @Test
  func `Tool returning Int publishes wrapped result schema`() {
    #expect(CountItems.tool.outputSchema == .object([
      "type": .string("object"),
      "properties": .object(["result": .object(["type": .string("integer")])]),
      "required": .array([.string("result")]),
      "additionalProperties": .bool(false),
    ]))
  }

  @Test
  func `Tool returning Dictionary publishes unwrapped object schema`() {
    #expect(GetTags.tool.outputSchema == .object([
      "type": .string("object"),
      "additionalProperties": .object(["type": .string("integer")]),
    ]))
  }

  @Test
  func `Tool returning Void publishes VoidOutput schema`() {
    #expect(Ping.tool.outputSchema == VoidOutput.outputJSONSchema)
  }

  // MARK: StructuredMetadataCarrier dispatcher rung (§3 rung 4 — end-to-end)

  @Test
  func `Tool returning ImageWithMetadata publishes metadata schema and image result types`() async throws {
    // Pins the §3 4-rung dispatcher's `StructuredMetadataCarrier` arm: a
    // `@Tool` returning `ImageWithMetadata<T>` must surface `T.outputJSONSchema`
    // (the typed metadata schema) as `outputSchema`, and `[.text, .json, .image]`
    // as `resultTypes`. Without this the capability filter under-reports and
    // `outputSchema` falls back to nil — both regressions a unit test on
    // `AISchema.outputSchema(for:)` alone wouldn't catch.
    #expect(TakeScreenshot.tool.outputSchema == ScreenshotMetadata.outputJSONSchema)
    #expect(TakeScreenshot.tool.resultTypes == [.text, .json, .image])

    // And the runtime tool result must carry the same structured payload the
    // dispatcher's schema describes — `text(json) + image` in `content[]` and
    // the typed metadata in `structuredContent`.
    let result = try await TakeScreenshot().perform().toToolResult()
    #expect(result.content.count == 2)
    if case let .text(text) = result.content[0] {
      #expect(text.contains("\"width\":1024"))
      #expect(text.contains("\"height\":768"))
    } else {
      Issue.record("Expected first content to be JSON text")
    }
    if case let .image(_, mime) = result.content[1] {
      #expect(mime == "image/png")
    } else {
      Issue.record("Expected second content to be image bytes")
    }
    #expect(result.structuredContent == .object([
      "width": .int(1024),
      "height": .int(768),
    ]))
  }

  // MARK: Output validation in Tools.call() (§B)

  @Test
  func `Tools.call passes valid structured output through unchanged`() async {
    let tools = Tools([CountItems.tool])
    let result = await tools.call(ToolCall(name: "count_items", id: "1", parameters: ["query": "abc"]))
    #expect(result.isError != true)
    #expect(result.result.structuredContent == .object(["result": .int(3)]))
  }

  @Test
  func `Tools.call surfaces missing structured payload as isError`() async {
    // Hand-build a tool that publishes outputSchema but skips structuredContent.
    let outputSchema: Value = .object([
      "type": .string("object"),
      "properties": .object(["x": .object(["type": .string("integer")])]),
      "required": .array([.string("x")]),
    ])
    let tool = Tool(
      name: "broken",
      description: "Returns content only",
      outputSchema: outputSchema,
    ) { _ in
      ToolOutputResult(content: [.text("oops")])
    }
    let tools = Tools([tool])
    let result = await tools.call(ToolCall(name: "broken", id: "1", parameters: [:]))
    #expect(result.isError == true)
    if case let .text(text) = result.content[0] {
      #expect(text.hasPrefix("Output validation error:"))
    } else {
      Issue.record("Expected text content describing the validation error")
    }
  }

  @Test
  func `Tools.call surfaces schema mismatch as isError`() async {
    let outputSchema: Value = .object([
      "type": .string("object"),
      "properties": .object(["x": .object(["type": .string("integer")])]),
      "required": .array([.string("x")]),
    ])
    let tool = Tool(
      name: "type_mismatch",
      description: "Returns wrong type",
      outputSchema: outputSchema,
    ) { _ in
      ToolOutputResult(
        content: [.text("oops")],
        structuredContent: .object(["x": .string("not an integer")]),
      )
    }
    let tools = Tools([tool])
    let result = await tools.call(ToolCall(name: "type_mismatch", id: "1", parameters: [:]))
    #expect(result.isError == true)
    if case let .text(text) = result.content[0] {
      #expect(text.hasPrefix("Output validation error:"))
    } else {
      Issue.record("Expected text content describing the validation error")
    }
  }

  @Test
  func `Tools.call skips validation when outputSchema is nil`() async {
    let tool = Tool(
      name: "untyped",
      description: "No schema",
    ) { _ in
      ToolOutputResult(content: [.text("ok")])
    }
    let tools = Tools([tool])
    let result = await tools.call(ToolCall(name: "untyped", id: "1", parameters: [:]))
    #expect(result.isError != true)
  }

  // MARK: Batched call preserves siblings on validation failure (§B)

  @Test
  func `Batched Tools.call preserves sibling results when one fails output validation`() async {
    let outputSchema: Value = .object([
      "type": .string("object"),
      "properties": .object(["x": .object(["type": .string("integer")])]),
      "required": .array([.string("x")]),
    ])
    let goodA = Tool(name: "good_a", description: "valid", outputSchema: outputSchema) { _ in
      ToolOutputResult(content: [.text("a")], structuredContent: .object(["x": .int(1)]))
    }
    let bad = Tool(name: "bad", description: "invalid", outputSchema: outputSchema) { _ in
      ToolOutputResult(content: [.text("oops")])
    }
    let goodC = Tool(name: "good_c", description: "valid", outputSchema: outputSchema) { _ in
      ToolOutputResult(content: [.text("c")], structuredContent: .object(["x": .int(3)]))
    }
    let tools = Tools([goodA, bad, goodC])
    let calls = [
      ToolCall(name: "good_a", id: "1", parameters: [:]),
      ToolCall(name: "bad", id: "2", parameters: [:]),
      ToolCall(name: "good_c", id: "3", parameters: [:]),
    ]
    let results = await tools.call(calls)
    #expect(results.count == 3)
    #expect(results[0].isError != true)
    #expect(results[1].isError == true)
    #expect(results[2].isError != true)
  }

  // MARK: Compatibility filter (§D)

  @Test
  func `ImageWithMetadata is compatible with image-supporting providers`() {
    let types = ImageWithMetadata<WidgetMeta>.resultTypes
    #expect(types == [.text, .json, .image])
    // [.text, .json, .image] ⊆ Anthropic's [.text, .json, .image, .file, .resource]
    #expect(types?.isSubset(of: AnthropicClient.supportedResultTypes) == true)
    // [.text, .json, .image] ⊆ Responses' [.text, .json, .image, .file, .resource]
    #expect(types?.isSubset(of: ResponsesClient.supportedResultTypes) == true)
    // [.text, .json, .image] ⊄ ChatCompletions' [.text, .json] (no .image)
    #expect(types?.isSubset(of: ChatCompletionsClient.supportedResultTypes) == false)
  }

  @Test
  func `MediaWithMetadata mixed-modality not compatible with audio-incompatible providers`() {
    let types = MediaWithMetadata<LipSyncMeta>.resultTypes
    #expect(types == [.text, .json, .image, .audio])
    // [.text, .json, .image, .audio] ⊄ Anthropic's set (no .audio)
    #expect(types?.isSubset(of: AnthropicClient.supportedResultTypes) == false)
    // ⊆ Gemini's set (which has .audio)
    #expect(types?.isSubset(of: GeminiClient.supportedResultTypes) == true)
  }

  // MARK: Optional<String> display vs Optional<Int> display (§C ambiguity case)

  @Test
  func `Optional String some renders quoted while bare String renders verbatim`() throws {
    let bare = try "hello".toToolResult()
    let some: String? = "hello"
    let wrapped = try some.toToolResult()
    // Display channels diverge by source type — this is the ambiguity case the
    // dual-channel design exists to resolve. Bare String passes through; the
    // Optional path renders the JSON form, which is the quoted string.
    #expect(bare.content == [.text("hello")])
    #expect(wrapped.content == [.text("\"hello\"")])
    // Structured channels are identical — wire shape is the same.
    #expect(bare.structuredContent == wrapped.structuredContent)
    #expect(bare.structuredContent == .object(["result": .string("hello")]))
  }

  // MARK: Validation only fires inside Tools.call (§B "Scope")

  @Test
  func `Direct tool execute call bypasses output validation`() async throws {
    // Per spec §B, validation is a Tools.call() guarantee, not a tool.execute
    // guarantee. Calling execute directly returns whatever the tool produces,
    // even when it would fail validation. Pinning this so a future refactor
    // doesn't slip validation into execute (which would change the public
    // contract authors and conversion paths rely on).
    let outputSchema: Value = .object([
      "type": .string("object"),
      "properties": .object(["x": .object(["type": .string("integer")])]),
      "required": .array([.string("x")]),
    ])
    let tool = Tool(
      name: "broken_direct",
      description: "Returns no structured content",
      outputSchema: outputSchema,
    ) { _ in
      ToolOutputResult(content: [.text("oops")])
    }
    let raw = try await tool.execute([:])
    // Raw return passes through unchanged — execute does not synthesize an
    // isError result, even though Tools.call() would.
    #expect(raw.content == [.text("oops")])
    #expect(raw.structuredContent == nil)
  }

  // MARK: ToolError-thrown errors bypass output validation (§B + §F)

  @Test
  func `Thrown ToolError bypasses output validation and preserves rich content`() async {
    struct RichFailure: AI.ToolError {
      var content: [ToolResult.Content] {
        [.text("operation failed"), .json(.object(["code": .int(42)]))]
      }

      var structuredContent: Value? {
        .object(["code": .int(42), "message": .string("operation failed")])
      }
    }

    let outputSchema: Value = .object([
      "type": .string("object"),
      "properties": .object(["x": .object(["type": .string("integer")])]),
      "required": .array([.string("x")]),
    ])
    let tool = Tool(
      name: "rich_error",
      description: "Throws a rich error",
      outputSchema: outputSchema,
    ) { _ in
      throw RichFailure()
    }
    let result = await Tools([tool]).call(ToolCall(name: "rich_error", id: "1", parameters: [:]))
    #expect(result.isError == true)
    // Author-thrown content survives — validation does not inspect throw paths,
    // even when outputSchema is set and the structuredContent doesn't match.
    #expect(result.content.count == 2)
    if case let .text(text) = result.content[0] {
      #expect(text == "operation failed")
    } else {
      Issue.record("Expected first block to be .text(\"operation failed\")")
    }
    if case let .json(value) = result.content[1] {
      #expect(value == .object(["code": .int(42)]))
    } else {
      Issue.record("Expected second block to be .json(...)")
    }
    // Structured channel survives end-to-end despite not matching outputSchema.
    #expect(result.result.structuredContent == .object([
      "code": .int(42),
      "message": .string("operation failed"),
    ]))
  }

  // MARK: Date ISO8601 channel parity (Low #14)

  @Test
  func `Date asJSONValue and asDisplayText produce equivalent ISO8601 strings`() throws {
    // The two paths use different formatter constructions (encoder.iso8601
    // strategy vs ISO8601DateFormatter). Pinning the equivalence so a change
    // to either path can't silently desync the structured channel from the
    // display channel.
    let cases: [Date] = [
      Date(timeIntervalSince1970: 0),
      Date(timeIntervalSince1970: 1_700_000_000),
      Date(timeIntervalSince1970: 1_900_000_000.5),
    ]
    for date in cases {
      let jsonValue = try date.asJSONValue()
      guard case let .string(jsonString) = jsonValue else {
        Issue.record("Expected Date asJSONValue to produce .string, got \(jsonValue)")
        continue
      }
      let displayString = try date.asDisplayText()
      #expect(jsonString == displayString)
    }
  }
}
