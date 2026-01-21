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

  func testMissingNameError() throws {
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
      macros: testMacros
    )
  }

  func testMissingDescriptionError() throws {
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
      macros: testMacros
    )
  }

  func testNotAStructError() throws {
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
      macros: testMacros
    )
  }

  func testInvalidToolNameWithSpaces() throws {
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
        DiagnosticSpec(message: "Invalid tool name: Tool name contains invalid characters: '  '. Only A-Z, a-z, 0-9, _, -, . are allowed", line: 1, column: 1),
      ],
      macros: testMacros
    )
  }

  func testInvalidToolNameTooLong() throws {
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
      macros: testMacros
    )
  }

  func testNonLiteralDefaultValueError() throws {
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
      macros: testMacros
    )
  }
}
#endif
