// Copyright © Anthony DePasquale

@testable import AI
import AITool
import Foundation
import Testing

// MARK: - Test Tools

@Tool
struct GetWeather {
  static let name = "get_weather"
  static let description = "Get weather for a city"

  @Parameter(description: "City name")
  var city: String

  @Parameter(description: "Temperature unit")
  var unit: String?

  func perform() async throws -> String {
    let u = unit ?? "celsius"
    return "Weather in \(city): Sunny, 22°\(u == "celsius" ? "C" : "F")"
  }
}

@Tool
struct SearchDocuments {
  static let name = "search_documents"
  static let description = "Search documents by query"

  @Parameter(description: "Search query", minLength: 1, maxLength: 500)
  var query: String

  @Parameter(description: "Maximum results", minimum: 1, maximum: 100)
  var limit: Int = 10

  func perform() async throws -> String {
    "Found \(limit) results for: \(query)"
  }
}

@Tool
struct GetServerTime {
  static let name = "get_server_time"
  static let description = "Returns current server time"

  func perform() async throws -> String {
    Date().ISO8601Format()
  }
}

@Tool
struct GenerateImage {
  static let name = "generate_image"
  static let description = "Generate an image"

  @Parameter(description: "Prompt for image generation")
  var prompt: String

  func perform() async throws -> ImageResult {
    ImageResult(pngData: Data([0x89, 0x50, 0x4E, 0x47]))
  }
}

@Tool
struct MultiContentTool {
  static let name = "multi_content"
  static let description = "Returns multiple content types"

  func perform() async throws -> MultiContent {
    MultiContent([
      .text("Analysis complete"),
      .image(Data([0x89, 0x50, 0x4E, 0x47]), mimeType: "image/png"),
    ])
  }
}

@Tool
struct ToolWithTitle {
  static let name = "tool_with_title"
  static let description = "A tool with a custom title"
  static let title = "My Custom Tool"

  func perform() async throws -> String {
    "Done"
  }
}

enum Priority: String, ToolEnum, CaseIterable {
  case low, medium, high
}

@Tool
struct ProcessNestedData {
  static let name = "process_nested_data"
  static let description = "Process nested data structures"

  @Parameter(description: "Array of dictionaries")
  var records: [[String: String]]

  @Parameter(description: "Dictionary with array values")
  var grouped: [String: [Int]]?

  @Parameter(description: "Nested arrays")
  var matrix: [[Int]]?

  func perform() async throws -> String {
    "Processed \(records.count) records"
  }
}

@Tool
struct SetPriority {
  static let name = "set_priority"
  static let description = "Set task priority"

  @Parameter(description: "Priority level")
  var priority: Priority

  func perform() async throws -> String {
    "Priority set to \(priority.rawValue)"
  }
}

@Tool
struct ToolWithCustomKey {
  static let name = "tool_with_custom_key"
  static let description = "Tool using custom JSON keys"

  @Parameter(key: "start_date", description: "Start date")
  var startDate: String

  @Parameter(key: "end_date", description: "End date")
  var endDate: String?

  func perform() async throws -> String {
    "Range: \(startDate) to \(endDate ?? "now")"
  }
}

@Tool
struct ToolWithParameterTitles {
  static let name = "tool_with_param_titles"
  static let description = "Tool with parameter titles"

  @Parameter(title: "Location", description: "City name")
  var city: String

  @Parameter(title: "Temperature Units", description: "Temperature unit")
  var unit: String?

  @Parameter(description: "No title specified")
  var other: String?

  func perform() async throws -> String {
    "Weather in \(city)"
  }
}

@Tool
struct ToolWithDateAndData {
  static let name = "tool_with_date_data"
  static let description = "Tool with Date and Data parameters"

  @Parameter(description: "Event date")
  var eventDate: Date

  @Parameter(description: "Binary payload")
  var payload: Data?

  func perform() async throws -> String {
    let payloadSize = payload?.count ?? 0
    return "Event at \(eventDate.ISO8601Format()), payload: \(payloadSize) bytes"
  }
}

@Tool
struct ToolWithAllOutputTypes {
  static let name = "multi_output"
  static let description = "Returns different output types"

  @Parameter(description: "Output type to return")
  var outputType: String

  func perform() async throws -> MultiContent {
    switch outputType {
      case "audio":
        MultiContent([
          .text("Audio generated"),
          .audio(Data([0x00, 0x01]), mimeType: "audio/wav"),
        ])
      case "file":
        MultiContent([
          .file(Data([0x25, 0x50, 0x44, 0x46]), mimeType: "application/pdf", filename: "report.pdf"),
        ])
      default:
        MultiContent([.text("Default output")])
    }
  }
}

@Tool
struct StrictSchemaTool {
  static let name = "strict_tool"
  static let description = "A tool with strict schema validation"
  static let strictSchema = true

  @Parameter(description: "Input value")
  var input: String

  func perform() async throws -> String {
    "Received: \(input)"
  }
}

// MARK: - Tests

@Suite("Tool Macro Integration Tests")
struct ToolMacroIntegrationTests {
  @Test("Basic tool definition is generated correctly")
  func basicToolDefinition() {
    let tool = GetWeather.tool

    #expect(tool.name == "get_weather")
    #expect(tool.description == "Get weather for a city")
    #expect(tool.title == "get_weather") // Defaults to name

    // Check schema
    let schema = tool.rawInputSchema
    #expect(schema["type"]?.stringValue == "object")

    let properties = schema["properties"]?.objectValue
    #expect(properties?["city"] != nil)
    #expect(properties?["unit"] != nil)

    let required = schema["required"]?.arrayValue?.compactMap { $0.stringValue }
    #expect(required?.contains("city") == true)
    #expect(required?.contains("unit") == false) // Optional parameter
  }

  @Test("Tool with default parameter value")
  func toolWithDefaultValue() {
    let tool = SearchDocuments.tool

    let schema = tool.rawInputSchema
    let properties = schema["properties"]?.objectValue

    // Check that limit has default value in schema
    let limitProp = properties?["limit"]?.objectValue
    #expect(limitProp?["default"]?.intValue == 10)

    // Required should not include limit (has default)
    let required = schema["required"]?.arrayValue?.compactMap { $0.stringValue }
    #expect(required?.contains("query") == true)
    #expect(required?.contains("limit") == false)
  }

  @Test("Tool with no parameters")
  func toolWithNoParameters() {
    let tool = GetServerTime.tool

    let schema = tool.rawInputSchema
    let properties = schema["properties"]?.objectValue
    #expect(properties?.isEmpty == true)

    let required = schema["required"]?.arrayValue
    #expect(required?.isEmpty == true)
  }

  @Test("Tool with custom title")
  func toolWithCustomTitle() {
    let tool = ToolWithTitle.tool

    #expect(tool.name == "tool_with_title")
    #expect(tool.title == "My Custom Tool")
  }

  @Test("Tool with strictSchema includes additionalProperties false")
  func toolWithStrictSchema() {
    let tool = StrictSchemaTool.tool
    let schema = tool.rawInputSchema

    // Strict tool should have additionalProperties: false
    #expect(schema["additionalProperties"]?.boolValue == false)

    // Non-strict tool should not have additionalProperties
    let nonStrictTool = GetWeather.tool
    let nonStrictSchema = nonStrictTool.rawInputSchema
    #expect(nonStrictSchema["additionalProperties"] == nil)
  }

  @Test("Tool with enum parameter")
  func toolWithEnumParameter() {
    let tool = SetPriority.tool

    let schema = tool.rawInputSchema
    let properties = schema["properties"]?.objectValue
    let priorityProp = properties?["priority"]?.objectValue

    let enumValues = priorityProp?["enum"]?.arrayValue?.compactMap { $0.stringValue }
    #expect(enumValues?.contains("low") == true)
    #expect(enumValues?.contains("medium") == true)
    #expect(enumValues?.contains("high") == true)
  }

  @Test("Parse and execute tool")
  func parseAndExecute() async throws {
    let arguments: [String: Value] = [
      "city": "Paris",
      "unit": "fahrenheit",
    ]

    let instance = try GetWeather.parse(from: arguments)
    #expect(instance.city == "Paris")
    #expect(instance.unit == "fahrenheit")

    let result = try await instance.perform()
    #expect(result.contains("Paris"))
    #expect(result.contains("F"))
  }

  @Test("Parse with optional parameter omitted")
  func parseWithOptionalOmitted() async throws {
    let arguments: [String: Value] = [
      "city": "Tokyo",
    ]

    let instance = try GetWeather.parse(from: arguments)
    #expect(instance.city == "Tokyo")
    #expect(instance.unit == nil)

    let result = try await instance.perform()
    #expect(result.contains("C")) // Default unit
  }

  @Test("Parse with default value")
  func parseWithDefaultValue() throws {
    let arguments: [String: Value] = [
      "query": "swift concurrency",
    ]

    let instance = try SearchDocuments.parse(from: arguments)
    #expect(instance.query == "swift concurrency")
    #expect(instance.limit == 10) // Default value
  }

  @Test("Parse with default value overridden")
  func parseWithDefaultValueOverridden() throws {
    let arguments: [String: Value] = [
      "query": "swift concurrency",
      "limit": 50,
    ]

    let instance = try SearchDocuments.parse(from: arguments)
    #expect(instance.query == "swift concurrency")
    #expect(instance.limit == 50)
  }

  @Test("Parse throws error when default parameter has wrong type")
  func parseThrowsForWrongTypeOnDefault() throws {
    let arguments: [String: Value] = [
      "query": "swift concurrency",
      "limit": "not a number", // Wrong type - should throw, not silently use default
    ]

    #expect(throws: ToolError.self) {
      _ = try SearchDocuments.parse(from: arguments)
    }
  }

  @Test("Parse enum parameter")
  func parseEnumParameter() throws {
    let arguments: [String: Value] = [
      "priority": "high",
    ]

    let instance = try SetPriority.parse(from: arguments)
    #expect(instance.priority == .high)
  }

  @Test("Tool returns image result")
  func toolReturnsImageResult() async throws {
    let arguments: [String: Value] = [
      "prompt": "A sunset",
    ]

    let instance = try GenerateImage.parse(from: arguments)
    let result = try await instance.perform()
    let content = result.toToolResult()

    #expect(content.count == 1)
    if case let .image(data, mimeType) = content[0] {
      #expect(mimeType == "image/png")
      #expect(!data.isEmpty)
    } else {
      Issue.record("Expected image content")
    }
  }

  // MARK: - Result Types Derivation Tests

  @Test("String return type derives text resultTypes")
  func stringResultTypesDerivation() {
    // GetWeather returns String, so resultTypes should be [.text]
    let tool = GetWeather.tool
    #expect(tool.resultTypes == [.text])
  }

  @Test("ImageResult return type derives image resultTypes")
  func imageResultTypesDerivation() {
    // GenerateImage returns ImageResult, so resultTypes should be [.image]
    let tool = GenerateImage.tool
    #expect(tool.resultTypes == [.image])
  }

  @Test("MultiContent return type derives nil resultTypes")
  func multiContentResultTypesDerivation() {
    // MultiContentTool returns MultiContent, so resultTypes should be nil
    let tool = MultiContentTool.tool
    #expect(tool.resultTypes == nil)
  }

  @Test("Tools compatible filtering works with derived resultTypes")
  func toolsCompatibleFiltering() {
    let tools = [GetWeather.tool, GenerateImage.tool, GetServerTime.tool]

    // ChatCompletions only supports text
    let compatibleWithChatCompletions = tools.compatible(with: ChatCompletionsClient.self)
    #expect(compatibleWithChatCompletions.count == 2)
    #expect(compatibleWithChatCompletions.contains { $0.name == "get_weather" })
    #expect(compatibleWithChatCompletions.contains { $0.name == "get_server_time" })
    #expect(!compatibleWithChatCompletions.contains { $0.name == "generate_image" })

    // Anthropic supports text and image
    let compatibleWithAnthropic = tools.compatible(with: AnthropicClient.self)
    #expect(compatibleWithAnthropic.count == 3)
  }

  @Test("Tools collection executes tool calls")
  func toolsCollectionExecution() async {
    let tools = Tools([
      GetWeather.tool,
      SearchDocuments.tool,
      GetServerTime.tool,
    ])

    let toolCall = GenerationResponse.ToolCall(
      name: "get_weather",
      id: "call_1",
      parameters: ["city": "London"]
    )

    let result = await tools.call(toolCall)

    #expect(result.name == "get_weather")
    #expect(result.id == "call_1")
    #expect(result.isError == nil)
    #expect(result.content.count == 1)

    if case let .text(text) = result.content[0] {
      #expect(text.contains("London"))
    } else {
      Issue.record("Expected text content")
    }
  }

  @Test("Tools collection handles unknown tool")
  func toolsCollectionUnknownTool() async {
    let tools = Tools([GetWeather.tool])

    let toolCall = GenerationResponse.ToolCall(
      name: "unknown_tool",
      id: "call_1",
      parameters: [:]
    )

    let result = await tools.call(toolCall)

    #expect(result.isError == true)
    if case let .text(text) = result.content[0] {
      #expect(text.contains("Unknown tool"))
    }
  }

  @Test("Tool with nested types generates correct schema")
  func toolWithNestedTypes() {
    let tool = ProcessNestedData.tool

    let schema = tool.rawInputSchema
    let properties = schema["properties"]?.objectValue

    // Check [[String: String]] - array of dictionaries
    let recordsProp = properties?["records"]?.objectValue
    #expect(recordsProp?["type"]?.stringValue == "array")
    let recordsItems = recordsProp?["items"]?.objectValue
    #expect(recordsItems?["type"]?.stringValue == "object")
    let recordsAdditionalProps = recordsItems?["additionalProperties"]?.objectValue
    #expect(recordsAdditionalProps?["type"]?.stringValue == "string")

    // Check [String: [Int]] - dictionary with array values
    let groupedProp = properties?["grouped"]?.objectValue
    #expect(groupedProp?["type"]?.stringValue == "object")
    let groupedAdditionalProps = groupedProp?["additionalProperties"]?.objectValue
    #expect(groupedAdditionalProps?["type"]?.stringValue == "array")
    let groupedItems = groupedAdditionalProps?["items"]?.objectValue
    #expect(groupedItems?["type"]?.stringValue == "integer")

    // Check [[Int]] - nested arrays
    let matrixProp = properties?["matrix"]?.objectValue
    #expect(matrixProp?["type"]?.stringValue == "array")
    let matrixItems = matrixProp?["items"]?.objectValue
    #expect(matrixItems?["type"]?.stringValue == "array")
    let matrixInnerItems = matrixItems?["items"]?.objectValue
    #expect(matrixInnerItems?["type"]?.stringValue == "integer")
  }

  @Test("Parse nested types")
  func parseNestedTypes() throws {
    let arguments: [String: Value] = [
      "records": .array([
        .object(["name": "Alice", "role": "admin"]),
        .object(["name": "Bob", "role": "user"]),
      ]),
      "grouped": .object([
        "scores": .array([1, 2, 3]),
        "counts": .array([10, 20]),
      ]),
      "matrix": .array([
        .array([1, 2]),
        .array([3, 4]),
      ]),
    ]

    let instance = try ProcessNestedData.parse(from: arguments)
    #expect(instance.records.count == 2)
    #expect(instance.records[0]["name"] == "Alice")
    #expect(instance.grouped?["scores"] == [1, 2, 3])
    #expect(instance.matrix == [[1, 2], [3, 4]])
  }

  // MARK: - Custom Key Tests

  @Test("Custom key appears in schema instead of property name")
  func customKeyInSchema() {
    let tool = ToolWithCustomKey.tool
    let properties = tool.rawInputSchema["properties"]?.objectValue

    // Schema should use custom keys, not Swift property names
    #expect(properties?["start_date"] != nil)
    #expect(properties?["end_date"] != nil)
    #expect(properties?["startDate"] == nil)
    #expect(properties?["endDate"] == nil)

    // Required should use custom key
    let required = tool.rawInputSchema["required"]?.arrayValue?.compactMap { $0.stringValue }
    #expect(required?.contains("start_date") == true)
    #expect(required?.contains("end_date") == false)
  }

  @Test("Parameter title appears in schema")
  func parameterTitleInSchema() {
    let tool = ToolWithParameterTitles.tool
    let properties = tool.rawInputSchema["properties"]?.objectValue

    // Parameters with title should have it in schema
    let cityProp = properties?["city"]?.objectValue
    #expect(cityProp?["title"]?.stringValue == "Location")

    let unitProp = properties?["unit"]?.objectValue
    #expect(unitProp?["title"]?.stringValue == "Temperature Units")

    // Parameter without title should not have title in schema
    let otherProp = properties?["other"]?.objectValue
    #expect(otherProp?["title"] == nil)
  }

  @Test("Parse uses custom keys from arguments")
  func parseWithCustomKeys() throws {
    let arguments: [String: Value] = [
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
    ]

    let instance = try ToolWithCustomKey.parse(from: arguments)
    #expect(instance.startDate == "2024-01-01")
    #expect(instance.endDate == "2024-12-31")
  }

  // MARK: - Validation Constraint Tests

  @Test("Validation constraints appear in generated schema")
  func validationConstraintsInSchema() {
    let tool = SearchDocuments.tool
    let properties = tool.rawInputSchema["properties"]?.objectValue

    // Check string constraints
    let queryProp = properties?["query"]?.objectValue
    #expect(queryProp?["minLength"]?.intValue == 1)
    #expect(queryProp?["maxLength"]?.intValue == 500)

    // Check numeric constraints
    let limitProp = properties?["limit"]?.objectValue
    #expect(limitProp?["minimum"]?.doubleValue == 1)
    #expect(limitProp?["maximum"]?.doubleValue == 100)
  }

  @Test("Tools.call rejects input that violates minLength constraint")
  func validationRejectsShortString() async {
    let tools = Tools([SearchDocuments.tool])

    let toolCall = GenerationResponse.ToolCall(
      name: "search_documents",
      id: "call_1",
      parameters: ["query": ""] // Empty string violates minLength: 1
    )

    let result = await tools.call(toolCall)

    #expect(result.isError == true)
    if case let .text(text) = result.content[0] {
      #expect(text.contains("validation") || text.contains("minLength"))
    }
  }

  @Test("Tools.call rejects input that violates maximum constraint")
  func validationRejectsOutOfRangeNumber() async {
    let tools = Tools([SearchDocuments.tool])

    let toolCall = GenerationResponse.ToolCall(
      name: "search_documents",
      id: "call_1",
      parameters: [
        "query": "valid query",
        "limit": 999, // Violates maximum: 100
      ]
    )

    let result = await tools.call(toolCall)

    #expect(result.isError == true)
    if case let .text(text) = result.content[0] {
      #expect(text.contains("validation") || text.contains("maximum"))
    }
  }

  @Test("Tools.call accepts valid input within constraints")
  func validationAcceptsValidInput() async {
    let tools = Tools([SearchDocuments.tool])

    let toolCall = GenerationResponse.ToolCall(
      name: "search_documents",
      id: "call_1",
      parameters: [
        "query": "valid query",
        "limit": 50,
      ]
    )

    let result = await tools.call(toolCall)

    #expect(result.isError == nil)
  }

  // MARK: - Parse Error Handling Tests

  @Test("Parse throws error for missing required parameter")
  func parseMissingRequiredParameter() {
    let arguments: [String: Value] = [:] // Missing required "city"

    #expect(throws: ToolError.self) {
      _ = try GetWeather.parse(from: arguments)
    }
  }

  @Test("Parse throws error for wrong parameter type")
  func parseWrongParameterType() {
    let arguments: [String: Value] = [
      "city": .int(123), // Should be string, not int
    ]

    #expect(throws: ToolError.self) {
      _ = try GetWeather.parse(from: arguments)
    }
  }

  @Test("Parse error contains parameter name")
  func parseErrorContainsParameterName() {
    let arguments: [String: Value] = [:]

    do {
      _ = try GetWeather.parse(from: arguments)
      Issue.record("Expected error to be thrown")
    } catch let error as ToolError {
      let description = error.errorDescription ?? ""
      #expect(description.contains("city"))
    } catch {
      Issue.record("Expected ToolError, got \(type(of: error))")
    }
  }

  // MARK: - Date and Data Parameter Tests

  @Test("Date parameter schema has date-time format")
  func dateParameterSchemaFormat() {
    let tool = ToolWithDateAndData.tool
    let properties = tool.rawInputSchema["properties"]?.objectValue
    let dateProp = properties?["eventDate"]?.objectValue

    #expect(dateProp?["type"]?.stringValue == "string")
    #expect(dateProp?["format"]?.stringValue == "date-time")
  }

  @Test("Data parameter schema has base64 encoding")
  func dataParameterSchemaEncoding() {
    let tool = ToolWithDateAndData.tool
    let properties = tool.rawInputSchema["properties"]?.objectValue
    let dataProp = properties?["payload"]?.objectValue

    #expect(dataProp?["type"]?.stringValue == "string")
    #expect(dataProp?["contentEncoding"]?.stringValue == "base64")
  }

  @Test("Parse Date from ISO 8601 string")
  func parseDateParameter() throws {
    let dateString = "2024-06-15T10:30:00Z"
    let arguments: [String: Value] = [
      "eventDate": .string(dateString),
    ]

    let instance = try ToolWithDateAndData.parse(from: arguments)

    let formatter = ISO8601DateFormatter()
    let expectedDate = try #require(formatter.date(from: dateString))
    #expect(instance.eventDate == expectedDate)
  }

  @Test("Parse Date with fractional seconds")
  func parseDateWithFractionalSeconds() throws {
    let dateString = "2024-06-15T10:30:00.123Z"
    let arguments: [String: Value] = [
      "eventDate": .string(dateString),
    ]

    let instance = try ToolWithDateAndData.parse(from: arguments)

    // Should parse successfully - date will be close to expected
    let calendar = Calendar(identifier: .gregorian)
    let components = try calendar.dateComponents(in: #require(TimeZone(identifier: "UTC")), from: instance.eventDate)
    #expect(components.year == 2024)
    #expect(components.month == 6)
    #expect(components.day == 15)
  }

  @Test("Parse Data from base64 string")
  func parseDataParameter() throws {
    let originalData = Data([0x48, 0x65, 0x6C, 0x6C, 0x6F]) // "Hello"
    let base64String = originalData.base64EncodedString()

    let arguments: [String: Value] = [
      "eventDate": "2024-01-01T00:00:00Z",
      "payload": .string(base64String),
    ]

    let instance = try ToolWithDateAndData.parse(from: arguments)

    #expect(instance.payload == originalData)
  }

  @Test("Invalid base64 returns nil for Data parameter")
  func parseInvalidBase64() throws {
    let arguments: [String: Value] = [
      "eventDate": "2024-01-01T00:00:00Z",
      "payload": .string("not valid base64!!!"),
    ]

    let instance = try ToolWithDateAndData.parse(from: arguments)

    // Invalid base64 should result in nil for optional Data
    #expect(instance.payload == nil)
  }

  // MARK: - Multiple Output Types Tests

  @Test("MultiContent output with audio")
  func multiContentWithAudio() async throws {
    let arguments: [String: Value] = ["outputType": "audio"]
    let instance = try ToolWithAllOutputTypes.parse(from: arguments)
    let result = try await instance.perform()
    let content = result.toToolResult()

    #expect(content.count == 2)

    if case let .text(text) = content[0] {
      #expect(text == "Audio generated")
    } else {
      Issue.record("Expected text content first")
    }

    if case let .audio(data, mimeType) = content[1] {
      #expect(mimeType == "audio/wav")
      #expect(!data.isEmpty)
    } else {
      Issue.record("Expected audio content second")
    }
  }

  @Test("MultiContent output with file")
  func multiContentWithFile() async throws {
    let arguments: [String: Value] = ["outputType": "file"]
    let instance = try ToolWithAllOutputTypes.parse(from: arguments)
    let result = try await instance.perform()
    let content = result.toToolResult()

    #expect(content.count == 1)

    if case let .file(data, mimeType, filename) = content[0] {
      #expect(mimeType == "application/pdf")
      #expect(filename == "report.pdf")
      #expect(!data.isEmpty)
    } else {
      Issue.record("Expected file content")
    }
  }

  // MARK: - Dynamic Tool API Tests

  @Test("Tool.Parameter factory methods generate correct schema")
  func parameterFactoryMethods() {
    let tool = Tool(
      name: "dynamic_tool",
      description: "Tool created with factory methods",
      parameters: [
        .string("query", description: "Search query", minLength: 1, maxLength: 100),
        .integer("count", description: "Item count", required: false, minimum: 0, maximum: 1000),
        .number("threshold", description: "Score threshold", minimum: 0.0, maximum: 1.0),
        .boolean("verbose", description: "Enable verbose output", required: false),
        .array("tags", description: "Filter tags", items: .string),
      ]
    ) { _ in [.text("Result")] }

    let properties = tool.rawInputSchema["properties"]?.objectValue

    // String with constraints
    let queryProp = properties?["query"]?.objectValue
    #expect(queryProp?["type"]?.stringValue == "string")
    #expect(queryProp?["minLength"]?.intValue == 1)
    #expect(queryProp?["maxLength"]?.intValue == 100)

    // Integer with constraints
    let countProp = properties?["count"]?.objectValue
    #expect(countProp?["type"]?.stringValue == "integer")
    #expect(countProp?["minimum"]?.doubleValue == 0)
    #expect(countProp?["maximum"]?.doubleValue == 1000)

    // Number (float) with constraints
    let thresholdProp = properties?["threshold"]?.objectValue
    #expect(thresholdProp?["type"]?.stringValue == "number")
    #expect(thresholdProp?["minimum"]?.doubleValue == 0.0)
    #expect(thresholdProp?["maximum"]?.doubleValue == 1.0)

    // Boolean
    let verboseProp = properties?["verbose"]?.objectValue
    #expect(verboseProp?["type"]?.stringValue == "boolean")

    // Array with items type
    let tagsProp = properties?["tags"]?.objectValue
    #expect(tagsProp?["type"]?.stringValue == "array")
    #expect(tagsProp?["items"]?.objectValue?["type"]?.stringValue == "string")

    // Required array should only include required params
    let required = tool.rawInputSchema["required"]?.arrayValue?.compactMap { $0.stringValue } ?? []
    #expect(required.contains("query"))
    #expect(required.contains("threshold"))
    #expect(required.contains("tags"))
    #expect(!required.contains("count"))
    #expect(!required.contains("verbose"))
  }

  @Test("Tool.Parameter factory methods include title in schema")
  func parameterFactoryMethodsWithTitle() {
    let tool = Tool(
      name: "titled_params_tool",
      description: "Tool with titled parameters",
      parameters: [
        .string("city", title: "City Name", description: "The city to search"),
        .integer("limit", title: "Result Limit", description: "Max results"),
        .number("threshold", description: "Without explicit title"),
      ]
    ) { _ in [.text("Result")] }

    let properties = tool.rawInputSchema["properties"]?.objectValue

    // Parameters with explicit title should have it in schema
    let cityProp = properties?["city"]?.objectValue
    #expect(cityProp?["title"]?.stringValue == "City Name")

    let limitProp = properties?["limit"]?.objectValue
    #expect(limitProp?["title"]?.stringValue == "Result Limit")

    // Parameter without explicit title should not have title in schema
    // (title defaults to name, so it's omitted to avoid redundancy)
    let thresholdProp = properties?["threshold"]?.objectValue
    #expect(thresholdProp?["title"] == nil)
  }

  @Test("Tool with inputSchema initializer uses raw schema directly")
  func toolWithRawInputSchema() {
    let customSchema: [String: Value] = [
      "type": .string("object"),
      "properties": .object([
        "custom_field": .object([
          "type": .string("string"),
          "pattern": .string("^[A-Z]{3}$"),
        ]),
      ]),
      "required": .array([.string("custom_field")]),
    ]

    let tool = Tool(
      name: "custom_schema_tool",
      description: "Tool with custom schema",
      inputSchema: customSchema
    ) { _ in [.text("OK")] }

    // Schema should be used exactly as provided
    #expect(tool.rawInputSchema["type"]?.stringValue == "object")
    let properties = tool.rawInputSchema["properties"]?.objectValue
    let customField = properties?["custom_field"]?.objectValue
    #expect(customField?["pattern"]?.stringValue == "^[A-Z]{3}$")
  }
}
