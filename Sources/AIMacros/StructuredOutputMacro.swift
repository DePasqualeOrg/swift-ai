// Copyright © Anthony DePasquale

import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

/// The `@StructuredOutput` macro pairs with JSONSchemaBuilder's `@Schemable`
/// to give a struct a stable wire encoding and `StructuredOutput` protocol
/// conformance.
///
/// Each attribute owns one concern:
/// - `@Schemable` (JSONSchemaBuilder) — generates the schema.
/// - `@StructuredOutput` (this macro) — synthesizes a stable `encode(to:)`
///   that calls `container.encode` for every stored property so optionals
///   emit as `null` rather than being absent, adds `StructuredOutput` and
///   `WrappableValue` conformances, and bridges `outputJSONSchema` to the
///   Schemable component through `SchemableAdapter`.
///
/// Mirrors swift-mcp's `@StructuredOutput` macro verbatim modulo the AI/MCP
/// module-name swap.
public struct StructuredOutputMacro: MemberMacro, ExtensionMacro {
  // MARK: - MemberMacro

  public static func expansion(
    of node: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo _: [TypeSyntax],
    in context: some MacroExpansionContext,
  ) throws -> [DeclSyntax] {
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      context.diagnose(Diagnostic(
        node: Syntax(node),
        message: StructuredOutputMacroDiagnostic.error(
          "@StructuredOutput can only be applied to structs.",
        ),
      ))
      return []
    }

    if let genericClause = structDecl.genericParameterClause {
      context.diagnose(Diagnostic(
        node: Syntax(genericClause),
        message: StructuredOutputMacroDiagnostic.error(
          "@StructuredOutput doesn't support generic structs. The synthesized 'outputJSONSchema' is a static property that requires a concrete type. Declare a non-generic wrapper struct (e.g. 'struct MyResult { let container: Container<Int> }') and attach '@StructuredOutput' to the wrapper.",
        ),
      ))
      return []
    }

    if !hasSchemableAttribute(structDecl.attributes) {
      context.diagnose(Diagnostic(
        node: Syntax(node),
        message: StructuredOutputMacroDiagnostic.error(
          "@StructuredOutput requires @Schemable. Add '@Schemable' to '\(structDecl.name.text)' so the schema can be generated.",
        ),
      ))
      return []
    }

    let userEncodeDecl = findUserEncodeToDecl(in: structDecl)
    let hasManualEncoding = hasManualEncodingAttribute(structDecl.attributes)

    if let userEncodeDecl, !hasManualEncoding {
      context.diagnose(Diagnostic(
        node: Syntax(userEncodeDecl.name),
        message: StructuredOutputMacroDiagnostic.error(
          "@StructuredOutput synthesizes 'encode(to:)' to guarantee a stable wire shape (every optional emits as 'null'). Remove this custom 'encode(to:)' and let the macro synthesize one, or mark the struct '@ManualEncoding' to opt out.",
        ),
      ))
      return []
    }

    if hasManualEncoding {
      if userEncodeDecl == nil {
        let anchor = manualEncodingAttribute(structDecl.attributes).map(Syntax.init) ?? Syntax(node)
        context.diagnose(Diagnostic(
          node: anchor,
          message: StructuredOutputMacroDiagnostic.warning(
            "@ManualEncoding opts out of @StructuredOutput's encoder synthesis, but no 'encode(to:)' was found in the struct body. If your encoder lives in an extension, ignore this warning. Otherwise add a hand-rolled 'encode(to:)' or remove '@ManualEncoding'.",
          ),
        ))
      }
      return []
    }

    let properties = storedInstanceProperties(in: structDecl)

    if let codingKeysEnum = findUserCodingKeysDecl(in: structDecl) {
      let cases = codingKeyCaseNames(in: codingKeysEnum)
      let missing = properties.filter { !cases.contains($0) }
      if !missing.isEmpty {
        let missingList = missing.map { "'\($0)'" }.joined(separator: ", ")
        let propertyWord = missing.count == 1 ? "property" : "properties"
        let caseClause = "case \(missing.joined(separator: ", "))"
        context.diagnose(Diagnostic(
          node: Syntax(codingKeysEnum.name),
          message: StructuredOutputMacroDiagnostic.error(
            "CodingKeys is missing case(s) for stored \(propertyWord) \(missingList). Add `\(caseClause)` to CodingKeys, or mark the struct '@ManualEncoding' if you intentionally want to exclude properties from the wire shape.",
          ),
        ))
        return []
      }
    }

    let accessPrefix = accessLevelPrefix(of: structDecl.modifiers)

    if properties.isEmpty {
      return []
    }

    var encodeLines: [String] = []
    encodeLines.append("    var container = encoder.container(keyedBy: CodingKeys.self)")
    for prop in properties {
      encodeLines.append("    try container.encode(self.\(prop), forKey: .\(prop))")
    }
    let encodeBody = encodeLines.joined(separator: "\n")

    let encodeSource = """
    \(accessPrefix)func encode(to encoder: Encoder) throws {
    \(encodeBody)
    }
    """
    let encodeDecl = DeclSyntax(stringLiteral: encodeSource)

    var members: [DeclSyntax] = [encodeDecl]

    if !hasUserCodingKeys(in: structDecl) {
      let caseList = properties.joined(separator: ", ")
      let codingKeysSource = """
      \(accessPrefix)enum CodingKeys: String, CodingKey {
          case \(caseList)
      }
      """
      members.append(DeclSyntax(stringLiteral: codingKeysSource))
    }

    return members
  }

  // MARK: - ExtensionMacro

  public static func expansion(
    of _: AttributeSyntax,
    attachedTo declaration: some DeclGroupSyntax,
    providingExtensionsOf type: some TypeSyntaxProtocol,
    conformingTo _: [TypeSyntax],
    in _: some MacroExpansionContext,
  ) throws -> [ExtensionDeclSyntax] {
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      return []
    }

    if structDecl.genericParameterClause != nil {
      return []
    }

    if !hasSchemableAttribute(structDecl.attributes) {
      return []
    }

    let accessPrefix = accessLevelPrefix(of: structDecl.modifiers)
    let extensionSource = """
    extension \(type.trimmedDescription): AI.StructuredOutput, AI.WrappableValue {
        \(accessPrefix)static var outputJSONSchema: AI.Value {
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
    """

    guard let ext = DeclSyntax(stringLiteral: extensionSource).as(ExtensionDeclSyntax.self) else {
      return []
    }
    return [ext]
  }

  // MARK: - Helpers

  private static func storedInstanceProperties(in structDecl: StructDeclSyntax) -> [String] {
    var names: [String] = []
    for member in structDecl.memberBlock.members {
      guard let varDecl = member.decl.as(VariableDeclSyntax.self),
            !varDecl.modifiers.contains(where: { $0.name.text == "static" })
      else { continue }

      for binding in varDecl.bindings {
        guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self) else {
          continue
        }
        if let accessorBlock = binding.accessorBlock {
          switch accessorBlock.accessors {
            case .getter:
              continue
            case let .accessors(accessors):
              let isComputed = accessors.contains { accessor in
                switch accessor.accessorSpecifier.text {
                  case "willSet", "didSet":
                    false
                  default:
                    true
                }
              }
              if isComputed {
                continue
              }
          }
        }
        names.append(identifier.identifier.text)
      }
    }
    return names
  }

  private static func findUserEncodeToDecl(in structDecl: StructDeclSyntax) -> FunctionDeclSyntax? {
    for member in structDecl.memberBlock.members {
      guard let funcDecl = member.decl.as(FunctionDeclSyntax.self),
            funcDecl.name.text == "encode",
            !funcDecl.modifiers.contains(where: { $0.name.text == "static" })
      else { continue }

      let params = funcDecl.signature.parameterClause.parameters
      guard params.count == 1,
            let param = params.first,
            param.firstName.text == "to"
      else { continue }

      let typeText = param.type.trimmedDescription
      guard typeText == "Encoder" || typeText == "any Encoder" else { continue }

      return funcDecl
    }
    return nil
  }

  private static func hasUserCodingKeys(in structDecl: StructDeclSyntax) -> Bool {
    findUserCodingKeysDecl(in: structDecl) != nil
  }

  private static func findUserCodingKeysDecl(in structDecl: StructDeclSyntax) -> EnumDeclSyntax? {
    for member in structDecl.memberBlock.members {
      guard let enumDecl = member.decl.as(EnumDeclSyntax.self),
            enumDecl.name.text == "CodingKeys"
      else { continue }
      return enumDecl
    }
    return nil
  }

  private static func codingKeyCaseNames(in enumDecl: EnumDeclSyntax) -> Set<String> {
    var names: Set<String> = []
    for member in enumDecl.memberBlock.members {
      guard let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) else { continue }
      for element in caseDecl.elements {
        names.insert(element.name.text)
      }
    }
    return names
  }

  private static func hasSchemableAttribute(_ attributes: AttributeListSyntax) -> Bool {
    attributes.contains { element in
      guard case let .attribute(attribute) = element else { return false }
      if let identifier = attribute.attributeName.as(IdentifierTypeSyntax.self) {
        return identifier.name.text == "Schemable"
      }
      if let memberType = attribute.attributeName.as(MemberTypeSyntax.self) {
        return memberType.name.text == "Schemable"
      }
      return false
    }
  }

  private static func hasManualEncodingAttribute(_ attributes: AttributeListSyntax) -> Bool {
    manualEncodingAttribute(attributes) != nil
  }

  private static func manualEncodingAttribute(_ attributes: AttributeListSyntax) -> AttributeSyntax? {
    for element in attributes {
      guard case let .attribute(attribute) = element else { continue }
      if let identifier = attribute.attributeName.as(IdentifierTypeSyntax.self),
         identifier.name.text == "ManualEncoding"
      {
        return attribute
      }
      if let memberType = attribute.attributeName.as(MemberTypeSyntax.self),
         memberType.name.text == "ManualEncoding"
      {
        return attribute
      }
    }
    return nil
  }

  private static func accessLevelPrefix(of modifiers: DeclModifierListSyntax) -> String {
    for modifier in modifiers {
      switch modifier.name.text {
        case "public", "package", "internal", "fileprivate", "private":
          return "\(modifier.name.text) "
        default:
          continue
      }
    }
    return ""
  }
}

/// `@ManualEncoding` is a marker attribute that opts out of `@StructuredOutput`'s
/// `encode(to:)` synthesis.
public struct ManualEncodingMacro: PeerMacro {
  public static func expansion(
    of _: AttributeSyntax,
    providingPeersOf _: some DeclSyntaxProtocol,
    in _: some MacroExpansionContext,
  ) throws -> [DeclSyntax] {
    []
  }
}

// MARK: - Diagnostics

struct StructuredOutputMacroDiagnostic: DiagnosticMessage {
  let message: String
  let diagnosticID: MessageID
  let severity: DiagnosticSeverity

  static func error(_ message: String) -> StructuredOutputMacroDiagnostic {
    StructuredOutputMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "StructuredOutputMacro", id: "error"),
      severity: .error,
    )
  }

  static func warning(_ message: String) -> StructuredOutputMacroDiagnostic {
    StructuredOutputMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "StructuredOutputMacro", id: "warning"),
      severity: .warning,
    )
  }
}
