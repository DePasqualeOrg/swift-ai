// Copyright © Anthony DePasquale

import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

#if canImport(AIMacros)
import AIMacros

final class ToolMacroTests: XCTestCase {
  let testMacros: [String: Macro.Type] = [
    "Tool": ToolMacro.self,
  ]

  // MARK: - Compile-Time Validation Tests

  func testMissingNameError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let description = "Missing name"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let description = "Missing name"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'static let name: String' property", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testMissingDescriptionError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'static let description: String' property", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testNotAStructError() {
    assertMacroExpansion(
      """
      @Tool
      class BadClass {
          static let name = "bad"
          static let description = "Bad"
      }
      """,
      expandedSource: """
      class BadClass {
          static let name = "bad"
          static let description = "Bad"
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool can only be applied to structs", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testInvalidToolNameWithSpaces() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "invalid tool name"
          static let description = "Has spaces"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "invalid tool name"
          static let description = "Has spaces"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "Invalid tool name: Tool name contains invalid characters: ' ', ' '. Only A-Z, a-z, 0-9, _, -, . are allowed", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testInvalidToolNameTooLong() {
    let longName = String(repeating: "a", count: 129)
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "\(longName)"
          static let description = "Name too long"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "\(longName)"
          static let description = "Name too long"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "Invalid tool name: Tool name exceeds maximum length of 128 characters (got 129)", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testNonLiteralDefaultValueError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal default"

          @Parameter(description: "Start date")
          var startDate: Date = Date()

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal default"

          @Parameter(description: "Start date")
          var startDate: Date = Date()

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "Parameter 'startDate' has a non-literal default value. Only literal values (numbers, strings, booleans) are supported. For complex defaults, make the parameter optional and handle the default in perform().", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testInterpolatedStringDefaultValueError() {
    assertMacroExpansion(
      #"""
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Interpolated string default"

          @Parameter(description: "Greeting")
          var greeting: String = "Hello, \(name)"

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      expandedSource: #"""
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Interpolated string default"

          @Parameter(description: "Greeting")
          var greeting: String = "Hello, \(name)"

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      diagnostics: [
        DiagnosticSpec(message: "Parameter 'greeting' has a non-literal default value. Only literal values (numbers, strings, booleans) are supported. For complex defaults, make the parameter optional and handle the default in perform().", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testDuplicateParameterKeyError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Duplicate keys"

          @Parameter(key: "query", description: "First query")
          var first: String

          @Parameter(key: "query", description: "Second query")
          var second: String

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Duplicate keys"

          @Parameter(key: "query", description: "First query")
          var first: String

          @Parameter(key: "query", description: "Second query")
          var second: String

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "Duplicate @Parameter key 'query'. Each @Parameter key must be unique.", line: 6, column: 5),
        DiagnosticSpec(message: "Duplicate @Parameter key 'query'. Each @Parameter key must be unique.", line: 9, column: 5),
      ],
      macros: testMacros,
    )
  }

  func testPerformMissingAsyncError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing async"

          func perform() throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing async"

          func perform() throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'perform()' to be marked 'async'", line: 6, column: 10),
      ],
      macros: testMacros,
    )
  }

  func testPerformMissingThrowsError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing throws"

          func perform() async -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing throws"

          func perform() async -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'perform()' to be marked 'throws'", line: 6, column: 10),
      ],
      macros: testMacros,
    )
  }

  func testPerformHasUnexpectedParametersError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Unexpected parameters"

          func perform(input: String) async throws -> String {
              input
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Unexpected parameters"

          func perform(input: String) async throws -> String {
              input
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'perform()' to take no arguments. Use '@Parameter' properties on the struct to declare inputs.", line: 6, column: 17),
      ],
      macros: testMacros,
    )
  }

  func testPerformWithVoidReturnNormalizesToVoidOutput() {
    // Void-returning perform is normalized to AI.VoidOutput so the wire shape
    // matches Optional<T>.none — see §C "VoidOutput for Void-returning tools".
    assertMacroExpansion(
      """
      @Tool
      struct PingTool {
          static let name = "ping"
          static let description = "Touch the server"

          func perform() async throws {
          }
      }
      """,
      expandedSource: """
      struct PingTool {
          static let name = "ping"
          static let description = "Touch the server"

          func perform() async throws {
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> AI.VoidOutput {
              try await perform()
              return AI.VoidOutput()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: AI.VoidOutput.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: AI.VoidOutput.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension PingTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testResultTypesOverrideEmittedIntoGeneratedTool() {
    // Per spec §A "grouped types" mitigation: a tool whose return type spans
    // multiple wire kinds (e.g. `Media` covers `.image + .audio`) can narrow
    // its declared types via `static let resultTypes`. The macro must emit
    // that override into the generated `Tool` initializer in place of the
    // type-level default; otherwise capability filtering (§D) over-reports
    // the tool's modalities.
    assertMacroExpansion(
      """
      @Tool
      struct NarrowScreenshot {
          static let name = "narrow_screenshot"
          static let description = "Take a screenshot"
          static let resultTypes: Set<AI.ToolResult.ValueType>? = [.image]

          func perform() async throws -> AI.Media {
              AI.Media(.image(Data(), mime: "image/png"))
          }
      }
      """,
      expandedSource: """
      struct NarrowScreenshot {
          static let name = "narrow_screenshot"
          static let description = "Take a screenshot"
          static let resultTypes: Set<AI.ToolResult.ValueType>? = [.image]

          func perform() async throws -> AI.Media {
              AI.Media(.image(Data(), mime: "image/png"))
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> AI.Media {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: [.image],
                  outputSchema: AI.AISchema.outputSchema(for: AI.Media.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension NarrowScreenshot: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testParameterOnStaticPropertyError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Static parameter"

          @Parameter(description: "A static property")
          static var shared: String = "oops"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Static parameter"

          @Parameter(description: "A static property")
          static var shared: String = "oops"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Parameter cannot be applied to static property 'shared'. Tool parameters must be instance properties.", line: 6, column: 5),
      ],
      macros: testMacros,
    )
  }

  func testPerformAccessLevelWarningWhenMoreRestrictive() {
    assertMacroExpansion(
      """
      @Tool
      public struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with restrictive perform"

          private func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      public struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with restrictive perform"

          private func perform() async throws -> String {
              "Result"
          }

          public init() {
          }

          public static var title: String {
              name
          }

          public func _perform() async throws -> String {
              try await perform()
          }

          public static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          public static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "'perform()' has more restrictive access (private) than the enclosing struct (public). If the return type is similarly restricted, the generated '_perform()' bridge will fail to compile.",
          line: 6,
          column: 5,
          severity: .warning,
        ),
      ],
      macros: testMacros,
    )
  }

  func testStaticPerformMethodError() {
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Static perform"

          static func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Static perform"

          static func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "@Tool requires 'perform()' to be an instance method, not static.", line: 6, column: 17),
      ],
      macros: testMacros,
    )
  }

  // MARK: - Access Level Propagation Tests

  func testPackageStructGeneratesPackageMembers() {
    assertMacroExpansion(
      """
      @Tool
      package struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      package struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }

          package init() {
          }

          package static var title: String {
              name
          }

          package func _perform() async throws -> String {
              try await perform()
          }

          package static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          package static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testDefaultAccessStructGeneratesUnmodifiedMembers() {
    assertMacroExpansion(
      """
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testPublicStructGeneratesPublicMembers() {
    assertMacroExpansion(
      """
      @Tool
      public struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      public struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }

          public init() {
          }

          public static var title: String {
              name
          }

          public func _perform() async throws -> String {
              try await perform()
          }

          public static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          public static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testInternalStructGeneratesInternalMembers() {
    assertMacroExpansion(
      """
      @Tool
      internal struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      internal struct MyTool {
          static let name = "my_tool"
          static let description = "A tool"

          func perform() async throws -> String {
              "done"
          }

          internal init() {
          }

          internal static var title: String {
              name
          }

          internal func _perform() async throws -> String {
              try await perform()
          }

          internal static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          internal static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testUnmarkedPerformInPublicStructDoesNotWarn() {
    // Regression test: the canonical `public struct` + unmarked `func perform()`
    // pattern must NOT trigger the access-level warning. That warning only fires
    // for explicit more-restrictive modifiers.
    assertMacroExpansion(
      """
      @Tool
      public struct MyTool {
          static let name = "my_tool"
          static let description = "Canonical public tool"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      public struct MyTool {
          static let name = "my_tool"
          static let description = "Canonical public tool"

          func perform() async throws -> String {
              "Result"
          }

          public init() {
          }

          public static var title: String {
              name
          }

          public func _perform() async throws -> String {
              try await perform()
          }

          public static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: []
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          public static func parse(from arguments: [String: AI.Value]) throws -> Self {
              Self()
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testParameterMissingTypeAnnotationError() {
    // `@Parameter var count = 1` relies on type inference from the default value.
    // The macro needs an explicit annotation to pick the schema and parse code,
    // so it must reject this up front — otherwise it would emit String.schema
    // and the user would see a downstream "cannot assign String to Int" error.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing type annotation"

          @Parameter(description: "Count")
          var count = 1

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing type annotation"

          @Parameter(description: "Count")
          var count = 1

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Parameter property 'count' requires an explicit type annotation (e.g. 'var count: Int').",
          line: 7,
          column: 9,
        ),
      ],
      macros: testMacros,
    )
  }

  func testNegativeLiteralDefault() {
    // Covers PrefixOperatorExpr('-') in both isLiteralExpression (accept) and
    // convertToValueLiteral (map to .int/.double). If the two drift, the second
    // now traps via preconditionFailure rather than silently emitting .null.
    assertMacroExpansion(
      """
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with negative default"

          @Parameter(description: "Count")
          var count: Int = -5

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with negative default"

          @Parameter(description: "Count")
          var count: Int = -5

          func perform() async throws -> String {
              "done"
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: [
                          AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
            name: "count",
            title: nil,
            description: "Count",
            schema: Int.schema,
            isOptional: false,
            hasDefault: true,
            defaultValue: AI.Value.int(-5),
            minLength: nil,
            maxLength: nil,
            minimum: nil,
            maximum: nil
                          )
                      ]
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              var _instance = Self()
              let _args = arguments
              if let _value = _args["count"], !_value.isNull {
                  _instance.count = try AITool.ToolMacroSupport.parseParameter(Int.schema, from: _value, parameterName: "count")
              }
              return _instance
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testExplicitOptionalParameterType() {
    // Covers `unwrapExplicitOptional` — the non-sugared `Optional<T>` form
    // must be treated the same as `T?` (isOptional: true, typeName: T).
    assertMacroExpansion(
      """
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with explicit Optional"

          @Parameter(description: "Limit")
          var limit: Optional<Int>

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with explicit Optional"

          @Parameter(description: "Limit")
          var limit: Optional<Int>

          func perform() async throws -> String {
              "done"
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: [
                          AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
            name: "limit",
            title: nil,
            description: "Limit",
            schema: Int.schema,
            isOptional: true,
            hasDefault: false,
            defaultValue: nil,
            minLength: nil,
            maxLength: nil,
            minimum: nil,
            maximum: nil
                          )
                      ]
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              var _instance = Self()
              let _args = arguments
              if let _value = _args["limit"], !_value.isNull {
                  _instance.limit = try AITool.ToolMacroSupport.parseParameter(Int.schema, from: _value, parameterName: "limit")
              }
              return _instance
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testParameterWithDidSetObserverAccepted() {
    // Stored properties with observers remain assignable by the generated
    // parser, so they must not be rejected as computed.
    assertMacroExpansion(
      """
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with observer-backed parameter"

          @Parameter(description: "City")
          var city: String = "" {
              didSet {}
          }

          func perform() async throws -> String {
              city
          }
      }
      """,
      expandedSource: """
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with observer-backed parameter"

          @Parameter(description: "City")
          var city: String = "" {
              didSet {}
          }

          func perform() async throws -> String {
              city
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: [
                          AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
            name: "city",
            title: nil,
            description: "City",
            schema: String.schema,
            isOptional: false,
            hasDefault: true,
            defaultValue: AI.Value.string(""),
            minLength: nil,
            maxLength: nil,
            minimum: nil,
            maximum: nil
                          )
                      ]
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              var _instance = Self()
              let _args = arguments
              if let _value = _args["city"], !_value.isNull {
                  _instance.city = try AITool.ToolMacroSupport.parseParameter(String.schema, from: _value, parameterName: "city")
              }
              return _instance
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testComputedParameterPropertyError() {
    // Getter-backed properties cannot be assigned by the generated parser.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Computed parameter"

          @Parameter(description: "City")
          var city: String {
              "Paris"
          }

          func perform() async throws -> String {
              city
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Computed parameter"

          @Parameter(description: "City")
          var city: String {
              "Paris"
          }

          func perform() async throws -> String {
              city
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Parameter property 'city' must be a stored property, not computed (the macro-generated parser assigns to it).",
          line: 7,
          column: 9,
        ),
      ],
      macros: testMacros,
    )
  }

  func testSpecialCharsInParameterDescription() {
    // Covers swiftStringLiteral's escape path for backslash + quote. Without the
    // re-escaping, the generated source would contain a raw `"` that closes the
    // enclosing string literal and fails to compile.
    assertMacroExpansion(
      #"""
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool"

          @Parameter(description: "She said \"hi\" and left")
          var message: String

          func perform() async throws -> String {
              "done"
          }
      }
      """#,
      expandedSource: #"""
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool"

          @Parameter(description: "She said \"hi\" and left")
          var message: String

          func perform() async throws -> String {
              "done"
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: [
                          AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
            name: "message",
            title: nil,
            description: "She said \\\"hi\\\" and left",
            schema: String.schema,
            isOptional: false,
            hasDefault: false,
            defaultValue: nil,
            minLength: nil,
            maxLength: nil,
            minimum: nil,
            maximum: nil
                          )
                      ]
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              var _instance = Self()
              let _args = arguments
              guard let _messageValue = _args["message"] else {
                  throw ToolDispatchError.missingRequiredParameter("message")
              }
              _instance.message = try AITool.ToolMacroSupport.parseParameter(String.schema, from: _messageValue, parameterName: "message")
              return _instance
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """#,
      macros: testMacros,
    )
  }

  func testQualifiedAIParameterAttribute() {
    // Covers isParameterAttribute's qualified-attribute branch: `@AI.Parameter`
    // must be recognized the same as the bare `@Parameter`. Used when a file
    // imports both MCP and AI and needs to disambiguate.
    assertMacroExpansion(
      """
      @Tool
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with qualified parameter"

          @AI.Parameter(description: "Query string")
          var query: String

          func perform() async throws -> String {
              "done"
          }
      }
      """,
      expandedSource: """
      struct MyTool {
          static let name = "my_tool"
          static let description = "Tool with qualified parameter"

          @AI.Parameter(description: "Query string")
          var query: String

          func perform() async throws -> String {
              "done"
          }

          init() {
          }

          static var title: String {
              name
          }

          func _perform() async throws -> String {
              try await perform()
          }

          static var tool: AI.Tool {
              let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
                  parameters: [
                          AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
            name: "query",
            title: nil,
            description: "Query string",
            schema: String.schema,
            isOptional: false,
            hasDefault: false,
            defaultValue: nil,
            minLength: nil,
            maxLength: nil,
            minimum: nil,
            maximum: nil
                          )
                      ]
              )
              return AI.Tool(
                  name: name,
                  description: description,
                  title: title,
                  inputSchema: _schemaBuild.schema,
                  resultTypes: String.resultTypes,
                  outputSchema: AI.AISchema.outputSchema(for: String.self),
                  schemaBuildErrorMessage: _schemaBuild.errorMessage,
                  execute: { parameters in
                      let instance = try Self.parse(from: parameters)
                      let output = try await instance._perform()
                      return try output.toToolResult()
                  }
              )
          }

          static func parse(from arguments: [String: AI.Value]) throws -> Self {
              var _instance = Self()
              let _args = arguments
              guard let _queryValue = _args["query"] else {
                  throw ToolDispatchError.missingRequiredParameter("query")
              }
              _instance.query = try AITool.ToolMacroSupport.parseParameter(String.schema, from: _queryValue, parameterName: "query")
              return _instance
          }
      }

      extension MyTool: AI.ToolSpec, Sendable {
      }
      """,
      macros: testMacros,
    )
  }

  func testMissingPerformDoesNotCascadeConformanceError() {
    // Regression test: when perform() is missing, the ExtensionMacro must NOT add the
    // ToolSpec conformance — otherwise the user sees both the attribute-level
    // missingPerformMethod error AND a generated "does not conform to ToolSpec" error.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing perform"
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Missing perform"
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Tool requires a 'perform' method (e.g., 'func perform() async throws -> String')",
          line: 1,
          column: 1,
        ),
      ],
      macros: testMacros,
    )
  }

  func testNegativeNonNumericLiteralDefaultRejected() {
    // `isLiteralExpression` used to accept any `PrefixOperatorExpr('-')` wrapping a
    // literal (including Bool/String/nil), but `convertToValueLiteral` only knows how
    // to emit `.int`/`.double` for negative numerics — anything else hit
    // `preconditionFailure` and crashed the macro plugin. The validator now restricts
    // the prefix-minus branch to numeric literals so nonsense defaults like `-true`
    // get the normal "non-literal default" diagnostic instead of a plugin crash.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Negative non-numeric default"

          @Parameter(description: "Flag")
          var flag: Bool = -true

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Negative non-numeric default"

          @Parameter(description: "Flag")
          var flag: Bool = -true

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(message: "Parameter 'flag' has a non-literal default value. Only literal values (numbers, strings, booleans) are supported. For complex defaults, make the parameter optional and handle the default in perform().", line: 1, column: 1),
      ],
      macros: testMacros,
    )
  }

  func testNonLiteralParameterDescriptionRejected() {
    // `@Parameter(key:/title:/description:)` values are baked into generated Swift
    // source, so the macro used to silently drop non-literal expressions and fall
    // back to nil metadata — producing a silent divergence between the declared tool
    // interface and the generated schema. The shared helper now emits a node-level
    // diagnostic so the mismatch surfaces at compile time.
    assertMacroExpansion(
      #"""
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal parameter metadata"

          @Parameter(description: sharedDescription)
          var city: String

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      expandedSource: #"""
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal parameter metadata"

          @Parameter(description: sharedDescription)
          var city: String

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      diagnostics: [
        DiagnosticSpec(
          message: "@Parameter 'description' on 'city' must be a string literal.",
          line: 6,
          column: 29,
        ),
      ],
      macros: testMacros,
    )
  }

  func testNonLiteralToolNameDoesNotAddConformance() {
    // When `static let name` isn't a plain string literal the member macro refuses to
    // generate the required members. The diagnostic points at the offending
    // expression (matching the `@Parameter` treatment), not at the attribute — a
    // generic "property is missing" error would mislead the user, since the property
    // is present but has a non-literal value.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = someComputedName
          static let description = "Non-literal name"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = someComputedName
          static let description = "Non-literal name"

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Tool 'name' must be a string literal.",
          line: 3,
          column: 23,
        ),
      ],
      macros: testMacros,
    )
  }

  func testNonLiteralStrictSchemaDiagnostic() {
    // `strictSchema` is read at macro expansion time to decide whether to insert
    // the strict-mode validation call. A non-literal initializer would have been
    // silently treated as `false` and disabled the assertion without warning.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal strictSchema"
          static let strictSchema = sharedFlag

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-literal strictSchema"
          static let strictSchema = sharedFlag

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Tool 'strictSchema' must be a boolean literal (true or false).",
          line: 5,
          column: 31,
        ),
      ],
      macros: testMacros,
    )
  }

  func testNonStringLiteralTitleDiagnostic() {
    // `title` is referenced at runtime as a `String`. A non-string-literal stored
    // initializer (e.g. `static let title = 1`) used to bypass the macro and only
    // fail later as a "cannot convert Int to String" error in macro-generated code.
    // Computed properties (no initializer) are still allowed and left to the
    // type checker.
    assertMacroExpansion(
      """
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-string title"
          static let title = 1

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      expandedSource: """
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Non-string title"
          static let title = 1

          func perform() async throws -> String {
              "Result"
          }
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@Tool 'title' must be a string literal.",
          line: 5,
          column: 24,
        ),
      ],
      macros: testMacros,
    )
  }

  func testInterpolatedToolDescriptionDiagnostic() {
    // Interpolated `static let description = "prefix \\(suffix)"` used to silently
    // fall through to the generic `missingDescription` error. It now gets a targeted
    // diagnostic explaining that interpolation isn't supported.
    assertMacroExpansion(
      #"""
      @Tool
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Prefix \(suffix)"

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      expandedSource: #"""
      struct BadTool {
          static let name = "bad_tool"
          static let description = "Prefix \(suffix)"

          func perform() async throws -> String {
              "Result"
          }
      }
      """#,
      diagnostics: [
        DiagnosticSpec(
          message: "@Tool 'description' must be a plain string literal without interpolation.",
          line: 4,
          column: 30,
        ),
      ],
      macros: testMacros,
    )
  }
}
#endif
