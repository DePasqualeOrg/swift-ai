// Copyright © Anthony DePasquale

import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

#if canImport(AIMacros)
import AIMacros

final class StructuredOutputMacroTests: XCTestCase {
  let testMacros: [String: Macro.Type] = [
    "StructuredOutput": StructuredOutputMacro.self,
    "ManualEncoding": ManualEncodingMacro.self,
  ]

  // MARK: - Diagnostic paths

  func testRequiresStruct() {
    assertMacroExpansion(
      """
      @StructuredOutput
      class Bad {
          let id: String = ""
      }
      """,
      expandedSource: """
      class Bad {
          let id: String = ""
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@StructuredOutput can only be applied to structs.",
          line: 1, column: 1,
        ),
      ],
      macros: testMacros,
    )
  }

  func testRejectsGenericStruct() {
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      struct Container<T: Sendable>: Sendable {
          let value: T
      }
      """,
      expandedSource: """
      @Schemable
      struct Container<T: Sendable>: Sendable {
          let value: T
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@StructuredOutput doesn't support generic structs. The synthesized 'outputJSONSchema' is a static property that requires a concrete type. Declare a non-generic wrapper struct (e.g. 'struct MyResult { let container: Container<Int> }') and attach '@StructuredOutput' to the wrapper.",
          line: 3, column: 17,
        ),
      ],
      macros: testMacros,
    )
  }

  func testRequiresSchemableAttribute() {
    assertMacroExpansion(
      """
      @StructuredOutput
      struct UserInfo: Sendable {
          let id: String
      }
      """,
      expandedSource: """
      struct UserInfo: Sendable {
          let id: String
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@StructuredOutput requires @Schemable. Add '@Schemable' to 'UserInfo' so the schema can be generated.",
          line: 1, column: 1,
        ),
      ],
      macros: testMacros,
    )
  }

  func testRejectsUserEncodeWithoutManualEncodingMarker() {
    // The MemberMacro path bails out (no encode/CodingKeys synthesized), but the
    // ExtensionMacro path still adds the protocol conformance — pinning that
    // split so the diagnostic doesn't accidentally start suppressing both.
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      struct UserInfo: Sendable {
          let id: String

          func encode(to encoder: Encoder) throws {
          }
      }
      """,
      expandedSource: """
      @Schemable
      struct UserInfo: Sendable {
          let id: String

          func encode(to encoder: Encoder) throws {
          }
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@StructuredOutput synthesizes 'encode(to:)' to guarantee a stable wire shape (every optional emits as 'null'). Remove this custom 'encode(to:)' and let the macro synthesize one, or mark the struct '@ManualEncoding' to opt out.",
          line: 6, column: 10,
        ),
      ],
      macros: testMacros,
    )
  }

  func testManualEncodingOptsOutOfSynthesis() {
    // With @ManualEncoding present and a hand-rolled encode, the macro emits
    // no encode/CodingKeys members but still adds the protocol conformance
    // through the ExtensionMacro path. (The @ManualEncoding attribute is
    // stripped from the expansion output because it's a registered peer macro
    // — same as how @StructuredOutput itself disappears.)
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      @ManualEncoding
      struct UserInfo: Sendable {
          let id: String

          func encode(to encoder: Encoder) throws {
          }
      }
      """,
      expandedSource: """
      @Schemable
      struct UserInfo: Sendable {
          let id: String

          func encode(to encoder: Encoder) throws {
          }
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      macros: testMacros,
    )
  }

  func testManualEncodingWithoutEncodeEmitsWarning() {
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      @ManualEncoding
      struct UserInfo: Sendable {
          let id: String
      }
      """,
      expandedSource: """
      @Schemable
      struct UserInfo: Sendable {
          let id: String
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "@ManualEncoding opts out of @StructuredOutput's encoder synthesis, but no 'encode(to:)' was found in the struct body. If your encoder lives in an extension, ignore this warning. Otherwise add a hand-rolled 'encode(to:)' or remove '@ManualEncoding'.",
          line: 3, column: 1,
          severity: .warning,
        ),
      ],
      macros: testMacros,
    )
  }

  func testUserCodingKeysMissingCaseDiagnostic() {
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      struct UserInfo: Sendable {
          let id: String
          let displayName: String?

          enum CodingKeys: String, CodingKey {
              case id
          }
      }
      """,
      expandedSource: """
      @Schemable
      struct UserInfo: Sendable {
          let id: String
          let displayName: String?

          enum CodingKeys: String, CodingKey {
              case id
          }
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      diagnostics: [
        DiagnosticSpec(
          message: "CodingKeys is missing case(s) for stored property 'displayName'. Add `case displayName` to CodingKeys, or mark the struct '@ManualEncoding' if you intentionally want to exclude properties from the wire shape.",
          line: 7, column: 10,
        ),
      ],
      macros: testMacros,
    )
  }

  // MARK: - Canonical expansion snapshot

  func testCanonicalExpansionEmitsEncodeAndCodingKeysAndExtension() {
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      struct UserInfo: Sendable {
          let id: String
          let displayName: String?
      }
      """,
      expandedSource: """
      @Schemable
      struct UserInfo: Sendable {
          let id: String
          let displayName: String?

          func encode(to encoder: Encoder) throws {
              var container = encoder.container(keyedBy: CodingKeys.self)
              try container.encode(self.id, forKey: .id)
              try container.encode(self.displayName, forKey: .displayName)
          }

          enum CodingKeys: String, CodingKey {
              case id, displayName
          }
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      macros: testMacros,
    )
  }

  func testPublicAccessLevelPropagatesToSynthesizedMembers() {
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      public struct UserInfo: Sendable {
          public let id: String
      }
      """,
      expandedSource: """
      @Schemable
      public struct UserInfo: Sendable {
          public let id: String

          public func encode(to encoder: Encoder) throws {
              var container = encoder.container(keyedBy: CodingKeys.self)
              try container.encode(self.id, forKey: .id)
          }

          public enum CodingKeys: String, CodingKey {
              case id
          }
      }

      extension UserInfo: AI.StructuredOutput, AI.WrappableValue {
          public static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      macros: testMacros,
    )
  }

  func testEmptyStructEmitsExtensionOnlyNoEncode() {
    // Empty structs still need protocol conformance (the @Schemable component
    // produces an empty-object schema), but there's nothing to encode and no
    // CodingKeys needed.
    assertMacroExpansion(
      """
      @Schemable
      @StructuredOutput
      struct Empty: Sendable {
      }
      """,
      expandedSource: """
      @Schemable
      struct Empty: Sendable {
      }

      extension Empty: AI.StructuredOutput, AI.WrappableValue {
          static var outputJSONSchema: AI.Value {
              _structuredOutputSchema
          }
          private static let _structuredOutputSchema: AI.Value = {
              do {
                  return .object(try AITool.ToolMacroSupport.structuredOutputSchemaDictionary(from: Self.schema))
              } catch {
                  fatalError("@StructuredOutput failed to derive outputJSONSchema for \\(Self.self) — the @Schemable component returned a non-object schema or the schema → Value round-trip threw: \\(error)")
              }
          }()
      }
      """,
      macros: testMacros,
    )
  }
}

#endif
