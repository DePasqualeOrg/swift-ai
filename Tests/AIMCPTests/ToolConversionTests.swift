// Copyright © Anthony DePasquale

import AIMCP
import Testing

struct ToolConversionTests {
  @Test
  func `AI.Tool to MCP.Tool conversion`() throws {
    let aiTool = AI.Tool(
      name: "get_weather",
      description: "Get weather for a location",
      title: "Get Weather",
      parameters: [
        AI.Tool.Parameter(
          name: "city",
          title: "City",
          type: .string,
          description: "The city name",
          required: true,
        ),
        AI.Tool.Parameter(
          name: "units",
          title: "Units",
          type: .string,
          description: "Temperature units",
          required: false,
        ),
      ],
      execute: { _ in AI.ToolOutputResult(content: [.text("Sunny")]) },
    )

    let tool = try MCP.Tool(from: aiTool)

    #expect(tool.name == "get_weather")
    #expect(tool.description == "Get weather for a location")
    #expect(tool.title == "Get Weather")

    // Check input schema structure
    guard case let .object(schema) = tool.inputSchema else {
      Issue.record("Expected object schema")
      return
    }

    #expect(schema["type"]?.stringValue == "object")

    guard case let .object(properties) = schema["properties"] else {
      Issue.record("Expected properties object")
      return
    }

    #expect(properties.count == 2)
    #expect(properties["city"] != nil)
    #expect(properties["units"] != nil)

    // Check required array
    guard case let .array(required) = schema["required"] else {
      Issue.record("Expected required array")
      return
    }

    #expect(required.count == 1)
    #expect(required[0].stringValue == "city")
  }

  @Test
  func `MCP.Tool to AI.Tool conversion`() throws {
    let inputSchema: MCP.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "query": .object([
          "type": .string("string"),
          "description": .string("Search query"),
        ]),
        "limit": .object([
          "type": .string("integer"),
          "description": .string("Max results"),
        ]),
      ]),
      "required": .array([.string("query")]),
    ])

    let tool = MCP.Tool(
      name: "search",
      title: "Search",
      description: "Search for items",
      inputSchema: inputSchema,
    )

    let aiTool = try AI.Tool(from: tool) { params in
      AI.ToolOutputResult(content: [.text("Results for: \(params["query"]?.stringRepresentation ?? "")")])
    }

    #expect(aiTool.name == "search")
    #expect(aiTool.description == "Search for items")
    #expect(aiTool.title == "Search")
    #expect(aiTool.parameters.count == 2)

    // Find the query parameter
    let queryParam = aiTool.parameters.first { $0.name == "query" }
    #expect(queryParam?.type == .string)
    #expect(queryParam?.required == true)
    #expect(queryParam?.description == "Search query")

    // Find the limit parameter
    let limitParam = aiTool.parameters.first { $0.name == "limit" }
    #expect(limitParam?.type == .integer)
    #expect(limitParam?.required == false)
  }

  @Test
  func `Parameter type mapping`() throws {
    let aiTool = AI.Tool(
      name: "test",
      description: "Test tool",
      title: "Test",
      parameters: [
        AI.Tool.Parameter(name: "str", title: "Str", type: .string, description: "", required: true),
        AI.Tool.Parameter(name: "num", title: "Num", type: .float, description: "", required: true),
        AI.Tool.Parameter(name: "int", title: "Int", type: .integer, description: "", required: true),
        AI.Tool.Parameter(name: "bool", title: "Bool", type: .boolean, description: "", required: true),
      ],
      execute: { _ in AI.ToolOutputResult(content: [.text("ok")]) },
    )

    let tool = try MCP.Tool(from: aiTool)

    guard case let .object(schema) = tool.inputSchema,
          case let .object(properties) = schema["properties"]
    else {
      Issue.record("Invalid schema structure")
      return
    }

    // Check type mappings
    if case let .object(strProp) = properties["str"] {
      #expect(strProp["type"]?.stringValue == "string")
    }
    if case let .object(numProp) = properties["num"] {
      #expect(numProp["type"]?.stringValue == "number")
    }
    if case let .object(intProp) = properties["int"] {
      #expect(intProp["type"]?.stringValue == "integer")
    }
    if case let .object(boolProp) = properties["bool"] {
      #expect(boolProp["type"]?.stringValue == "boolean")
    }
  }

  @Test
  func `Batch conversion of AI.Tools to MCP.Tools`() throws {
    let aiTools = [
      AI.Tool(name: "func1", description: "First", title: "Func 1", parameters: [], execute: { _ in AI.ToolOutputResult(content: [.text("1")]) }),
      AI.Tool(name: "func2", description: "Second", title: "Func 2", parameters: [], execute: { _ in AI.ToolOutputResult(content: [.text("2")]) }),
    ]

    let tools = try aiTools.mcpTools()
    #expect(tools.count == 2)
    #expect(tools[0].name == "func1")
    #expect(tools[1].name == "func2")
  }

  @Test
  func `AI.Tool to MCP.Tool conversion throws for invalid imperative schema`() {
    let aiTool = AI.Tool(
      name: "invalid_tool",
      description: "Duplicate parameters",
      parameters: [
        AI.Tool.Parameter(name: "query", title: "Query", type: .string, description: "", required: true),
        AI.Tool.Parameter(name: "query", title: "Query Again", type: .string, description: "", required: true),
      ],
      execute: { _ in AI.ToolOutputResult(content: [.text("ok")]) },
    )

    #expect(throws: AIError.self) {
      _ = try MCP.Tool(from: aiTool)
    }
  }

  // MARK: - outputSchema round-trip across the MCP boundary (§H)

  @Test
  func `AI.Tool to MCP.Tool carries outputSchema across the boundary`() throws {
    // Schemas are JSON-shaped and never contain MCP-only .data, so the
    // round-trip is byte-equal. Pinning so a future refactor that drops
    // outputSchema from the conversion path can't slip through CI.
    let outputSchema: AI.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "score": .object(["type": .string("integer")]),
        "label": .object(["type": .string("string")]),
      ]),
      "required": .array([.string("score"), .string("label")]),
    ])
    let aiTool = AI.Tool(
      name: "rate",
      description: "Rate something",
      parameters: [],
      outputSchema: outputSchema,
      execute: { _ in AI.ToolOutputResult(content: [.text("ok")]) },
    )

    let mcpTool = try MCP.Tool(from: aiTool)
    #expect(mcpTool.outputSchema != nil)
    #expect(mcpTool.outputSchema == outputSchema.mcpValue)
  }

  @Test
  func `MCP.Tool to AI.Tool carries outputSchema across the boundary`() throws {
    let mcpOutputSchema: MCP.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "uri": .object(["type": .string("string")]),
        "size": .object(["type": .string("integer")]),
      ]),
      "required": .array([.string("uri")]),
    ])
    let mcpTool = MCP.Tool(
      name: "fetch",
      description: "Fetch a resource",
      inputSchema: .object([
        "type": .string("object"),
        "properties": .object([:]),
      ]),
      outputSchema: mcpOutputSchema,
    )

    let aiTool = try AI.Tool(from: mcpTool) { _ in
      AI.ToolOutputResult(content: [.text("ok")])
    }
    #expect(aiTool.outputSchema != nil)
    #expect(aiTool.outputSchema == mcpOutputSchema.aiValue)
  }

  @Test
  func `outputSchema survives an AI to MCP to AI round-trip byte-equal`() throws {
    // Spec §10 calls for the byte-equal round-trip pin: schemas are JSON-shaped
    // and never carry MCP's binary-only `.data`, so a full AI → MCP → AI loop
    // must recover the original schema verbatim. Locking it in so a future
    // converter change that lossily collapses (e.g.) integer/number can't
    // silently degrade tool-result validation.
    let outputSchema: AI.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "score": .object(["type": .string("integer"), "minimum": .int(0)]),
        "label": .object(["type": .string("string")]),
        "tags": .object([
          "type": .string("array"),
          "items": .object(["type": .string("string")]),
        ]),
        "nullable": .object(["type": .array([.string("string"), .string("null")])]),
      ]),
      "required": .array([.string("score"), .string("label")]),
      "additionalProperties": .bool(false),
    ])
    let aiTool = AI.Tool(
      name: "rate",
      description: "Rate something",
      parameters: [],
      outputSchema: outputSchema,
      execute: { _ in AI.ToolOutputResult(content: [.text("ok")]) },
    )
    let mcpTool = try MCP.Tool(from: aiTool)
    let roundTripped = try AI.Tool(from: mcpTool) { _ in
      AI.ToolOutputResult(content: [.text("ok")])
    }
    #expect(roundTripped.outputSchema == outputSchema)
  }

  @Test
  func `outputSchema survives an MCP to AI to MCP round-trip byte-equal`() throws {
    let mcpOutputSchema: MCP.Value = .object([
      "type": .string("object"),
      "properties": .object([
        "uri": .object(["type": .string("string"), "format": .string("uri")]),
        "size": .object(["type": .string("integer")]),
        "tags": .object([
          "type": .string("array"),
          "items": .object(["type": .string("string")]),
        ]),
      ]),
      "required": .array([.string("uri")]),
      "additionalProperties": .bool(false),
    ])
    let mcpTool = MCP.Tool(
      name: "fetch",
      description: "Fetch a resource",
      inputSchema: .object([
        "type": .string("object"),
        "properties": .object([:]),
      ]),
      outputSchema: mcpOutputSchema,
    )
    let aiTool = try AI.Tool(from: mcpTool) { _ in
      AI.ToolOutputResult(content: [.text("ok")])
    }
    let roundTripped = try MCP.Tool(from: aiTool)
    #expect(roundTripped.outputSchema == mcpOutputSchema)
  }
}
