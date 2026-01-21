// Copyright © Anthony DePasquale

import AIMCP
import Testing

@Suite("Value Conversions")
struct ValueConversionTests {
  @Test("AI.Value to MCP.Value - primitives")
  func aiValueToMCPValuePrimitives() {
    // String
    let stringValue = AI.Value.string("hello")
    #expect(stringValue.mcpValue == MCP.Value.string("hello"))

    // Bool
    let boolValue = AI.Value.bool(true)
    #expect(boolValue.mcpValue == MCP.Value.bool(true))

    // Null
    let nullValue = AI.Value.null
    #expect(nullValue.mcpValue == MCP.Value.null)
  }

  @Test("AI.Value to MCP.Value - numbers")
  func aiValueToMCPValueNumbers() {
    // Int stays as int
    let intValue = AI.Value.int(42)
    #expect(intValue.mcpValue == MCP.Value.int(42))

    // Double stays as double
    let doubleValue = AI.Value.double(3.14)
    #expect(doubleValue.mcpValue == MCP.Value.double(3.14))
  }

  @Test("AI.Value to MCP.Value - nested structures")
  func aiValueToMCPValueNested() {
    let aiArray = AI.Value.array([.string("a"), .int(1)])
    let mcpArray = aiArray.mcpValue
    #expect(mcpArray == MCP.Value.array([.string("a"), .int(1)]))

    let aiObject = AI.Value.object(["key": .string("value")])
    let mcpObject = aiObject.mcpValue
    #expect(mcpObject == MCP.Value.object(["key": .string("value")]))
  }

  @Test("MCP.Value to AI.Value - primitives")
  func mcpValueToAIValuePrimitives() {
    #expect(MCP.Value.string("test").aiValue == AI.Value.string("test"))
    #expect(MCP.Value.bool(false).aiValue == AI.Value.bool(false))
    #expect(MCP.Value.null.aiValue == AI.Value.null)
  }

  @Test("MCP.Value to AI.Value - numbers")
  func mcpValueToAIValueNumbers() {
    // Int stays as int
    #expect(MCP.Value.int(42).aiValue == AI.Value.int(42))

    // Double stays as double
    #expect(MCP.Value.double(3.14).aiValue == AI.Value.double(3.14))
  }

  @Test("Round-trip conversion preserves values")
  func roundTripConversion() {
    let original = AI.Value.object([
      "name": .string("test"),
      "count": .int(10),
      "price": .double(19.99),
      "enabled": .bool(true),
      "tags": .array([.string("a"), .string("b")]),
    ])

    let converted = original.mcpValue.aiValue
    #expect(converted == original)
  }

  @Test("Dictionary conversion helpers")
  func dictionaryConversions() {
    let aiDict: [String: AI.Value] = [
      "a": .string("hello"),
      "b": .int(42),
    ]

    let mcpDict = aiDict.mcpValues
    #expect(mcpDict["a"] == MCP.Value.string("hello"))
    #expect(mcpDict["b"] == MCP.Value.int(42))

    let backToAI = mcpDict.aiValues
    #expect(backToAI == aiDict)
  }

  @Test("MCP.Value data case converts to string")
  func mcpDataConvertsToString() {
    // MCP's .data case should convert to a data URL string in AI.Value
    let data = "Hello".data(using: .utf8)!
    let mcpData = MCP.Value.data(mimeType: "text/plain", data)
    let aiValue = mcpData.aiValue

    // Should be a string with data URL format
    if case let .string(s) = aiValue {
      #expect(s.hasPrefix("data:text/plain;base64,"))
    } else {
      Issue.record("Expected string value for data conversion")
    }
  }
}
