// Copyright © Anthony DePasquale

@testable @_spi(ToolMacroSupport) import AI
import Foundation
import JSONSchemaBuilder
import Testing

@Schemable
struct SchemableAdapterTestsSearchQuery {
  let text: String
  let limit: Int
}

@Schemable
enum SchemableAdapterTestsPriority {
  case low
  case medium
  case high
}

@Schemable
enum SchemableAdapterTestsLineEdit {
  case insert(line: Int, lines: [String])
  case delete(startLine: Int, endLine: Int)
  case replace(startLine: Int, endLine: Int, lines: [String])
}

struct SchemableAdapterTests {
  @Test
  func `Schemable struct round-trips into Value dictionary`() throws {
    let dict = try SchemableAdapter.valueDictionary(from: SchemableAdapterTestsSearchQuery.schema)

    #expect(dict["type"] == .string("object"))
    let properties = try #require(dict["properties"]?.objectValue)
    #expect(properties["text"]?.objectValue?["type"] == .string("string"))
    #expect(properties["limit"]?.objectValue?["type"] == .string("integer"))

    let required = try #require(dict["required"]?.arrayValue)
    #expect(Set(required.compactMap(\.stringValue)) == ["text", "limit"])
  }

  @Test
  func `Schemable plain enum produces string schema with enum values`() throws {
    let dict = try SchemableAdapter.valueDictionary(from: SchemableAdapterTestsPriority.schema)

    #expect(dict["type"] == .string("string"))
    let enumValues = try #require(dict["enum"]?.arrayValue)
    #expect(Set(enumValues.compactMap(\.stringValue)) == ["low", "medium", "high"])
  }

  @Test
  func `Schemable associated-value enum produces oneOf composition`() throws {
    let dict = try SchemableAdapter.valueDictionary(from: SchemableAdapterTestsLineEdit.schema)

    let oneOf = try #require(dict["oneOf"]?.arrayValue)
    #expect(oneOf.count == 3)

    let caseKeys = oneOf.compactMap { variant -> String? in
      guard let props = variant.objectValue?["properties"]?.objectValue else { return nil }
      return props.keys.first
    }
    #expect(Set(caseKeys) == ["insert", "delete", "replace"])
  }

  @Test
  func `Primitive Schemable conformances produce expected schemas`() throws {
    #expect(try SchemableAdapter.valueDictionary(from: String.schema)["type"] == .string("string"))
    #expect(try SchemableAdapter.valueDictionary(from: Int.schema)["type"] == .string("integer"))
    #expect(try SchemableAdapter.valueDictionary(from: Double.schema)["type"] == .string("number"))
    #expect(try SchemableAdapter.valueDictionary(from: Bool.schema)["type"] == .string("boolean"))
  }

  @Test
  func `Array of primitives produces array schema with items`() throws {
    let dict = try SchemableAdapter.valueDictionary(from: [String].schema)
    #expect(dict["type"] == .string("array"))
    let items = try #require(dict["items"]?.objectValue)
    #expect(items["type"] == .string("string"))
  }

  @Test
  func `Primitive parse returns the underlying Swift value`() throws {
    let text: String = try SchemableAdapter.parse(String.schema, from: .string("hello"), parameterName: "x")
    #expect(text == "hello")

    let n: Int = try SchemableAdapter.parse(Int.schema, from: .int(42), parameterName: "x")
    #expect(n == 42)

    let arr: [String] = try SchemableAdapter.parse([String].schema, from: .array([.string("a"), .string("b")]), parameterName: "x")
    #expect(arr == ["a", "b"])
  }

  @Test
  func `Parse surfaces ParseIssue as human-readable error`() {
    #expect(throws: ToolError.self) {
      _ = try SchemableAdapter.parse(String.schema, from: .int(42), parameterName: "name")
    }
  }
}
