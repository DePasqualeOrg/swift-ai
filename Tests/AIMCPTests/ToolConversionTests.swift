// Copyright © Anthony DePasquale

import AIMCP
import Testing

@Suite("Tool Conversions")
struct ToolConversionTests {
  @Test("AI.Tool to MCP.Tool conversion")
  func aiToolToMCPTool() {
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
          required: true
        ),
        AI.Tool.Parameter(
          name: "units",
          title: "Units",
          type: .string,
          description: "Temperature units",
          required: false
        ),
      ],
      execute: { _ in [.text("Sunny")] }
    )

    let tool = MCP.Tool(from: aiTool)

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

  @Test("MCP.Tool to AI.Tool conversion")
  func mcpToolToAITool() throws {
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
      inputSchema: inputSchema
    )

    let aiTool = try AI.Tool(from: tool) { params in
      [.text("Results for: \(params["query"]?.stringRepresentation ?? "")")]
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

  @Test("Parameter type mapping")
  func parameterTypeMapping() {
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
      execute: { _ in [.text("ok")] }
    )

    let tool = MCP.Tool(from: aiTool)

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

  @Test("Batch conversion of AI.Tools to MCP.Tools")
  func batchAIToolsToMCPTools() {
    let aiTools = [
      AI.Tool(name: "func1", description: "First", title: "Func 1", parameters: [], execute: { _ in [.text("1")] }),
      AI.Tool(name: "func2", description: "Second", title: "Func 2", parameters: [], execute: { _ in [.text("2")] }),
    ]

    let tools = aiTools.mcpTools
    #expect(tools.count == 2)
    #expect(tools[0].name == "func1")
    #expect(tools[1].name == "func2")
  }
}
