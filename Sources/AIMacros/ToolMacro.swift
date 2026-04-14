// Copyright © Anthony DePasquale

import Foundation
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

/// The `@Tool` macro generates `ToolSpec` protocol conformance.
///
/// It inspects the struct to find:
/// - `static let name: String` — The tool name
/// - `static let description: String` — The tool description
/// - `static let title: String` (optional) — User-facing title
/// - `static let strictSchema: Bool` (optional) — Opt-in assertion that the tool's
///   schema is strict JSON Schema-compatible. Defaults to `false`. When set to
///   `true`, the generated `tool` accessor traps at first access (typically when
///   the tool is registered) if the schema is not strict-compatible. This is a
///   declaration-site self-check by the tool author, independent of any per-request
///   strict-mode flag configured on a client.
/// - Properties with `@Parameter` attribute — Tool parameters
/// - `func perform()` — The execution method
///
/// It generates:
/// - `static var tool: Tool` — The tool definition with JSON Schema
/// - `static func parse(from:)` — Argument parsing
/// - `init()` — Empty initializer
/// - `_perform()` — Bridges to the user's `perform()` method
/// - `static var title: String` — Defaults to `name` (only if not declared on the struct)
public struct ToolMacro: MemberMacro, ExtensionMacro {
  // MARK: - MemberMacro

  public static func expansion(
    of _: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo _: [TypeSyntax],
    in context: some MacroExpansionContext,
  ) throws -> [DeclSyntax] {
    // Ensure we're applied to a struct
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      throw ToolMacroError.notAStruct
    }

    // Determine access level from the struct declaration
    let accessLevel = structDecl.modifiers.first(where: {
      $0.name.text == "public" || $0.name.text == "package" || $0.name.text == "internal"
    })?.name.text
    let accessPrefix = accessLevel.map { "\($0) " } ?? ""

    // Extract tool metadata
    let toolInfo: ToolInfo
    do {
      toolInfo = try extractToolInfo(from: structDecl, context: context)
    } catch is AbortMacroExpansion {
      // Diagnostic has already been emitted at a specific node; skip member generation.
      return []
    }

    // Generate members
    var members: [DeclSyntax] = []

    // Generate init()
    members.append("""
    \(raw: accessPrefix)init() {}
    """)

    // Generate title if not provided by user
    if !toolInfo.hasTitle {
      members.append("""
      \(raw: accessPrefix)static var title: String { name }
      """)
    }

    // Generate _perform() bridging to the user's perform()
    members.append("""
    \(raw: accessPrefix)func _perform() async throws -> \(raw: toolInfo.outputType) {
        try await perform()
    }
    """)

    // Generate tool property
    let toolDecl = generateToolProperty(
      toolInfo: toolInfo,
      accessPrefix: accessPrefix,
    )
    members.append(toolDecl)

    // Generate parse(from:)
    let parseDecl = generateParseMethod(toolInfo: toolInfo, accessPrefix: accessPrefix)
    members.append(parseDecl)

    return members
  }

  // MARK: - ExtensionMacro

  public static func expansion(
    of _: AttributeSyntax,
    attachedTo declaration: some DeclGroupSyntax,
    providingExtensionsOf type: some TypeSyntaxProtocol,
    conformingTo _: [TypeSyntax],
    in context: some MacroExpansionContext,
  ) throws -> [ExtensionDeclSyntax] {
    // Validate before adding conformance
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      // Let member macro report the error
      return []
    }

    // Validation checks below must mirror those in `extractToolInfo`. If one side
    // rejects a tool but the other generates code for it, the user sees a cascade
    // of "type does not conform to ToolSpec" errors on top of the real problem.
    var hasName = false
    var hasDescription = false
    var toolName: String?
    var parameterInfos: [ParameterInfo] = []
    var performDecl: FunctionDeclSyntax?

    for member in structDecl.memberBlock.members {
      if let varDecl = member.decl.as(VariableDeclSyntax.self),
         varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
        // @Parameter on static properties is an error
        if hasParameterAttribute(varDecl) {
          return []
        }
        for binding in varDecl.bindings {
          if let identifier = binding.pattern.as(IdentifierPatternSyntax.self) {
            let propName = identifier.identifier.text
            if propName == "name" {
              hasName = true
              // Extract name value for validation
              if let initializer = binding.initializer,
                 let stringLiteral = initializer.value.as(StringLiteralExprSyntax.self),
                 let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
              {
                toolName = segment.content.text
              }
            }
            if propName == "description" { hasDescription = true }
          }
        }
      }

      // Check for @Parameter properties with non-literal defaults
      if let varDecl = member.decl.as(VariableDeclSyntax.self),
         !varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
        if hasParameterAttribute(varDecl) {
          for binding in varDecl.bindings {
            if let initializer = binding.initializer,
               !isLiteralExpression(initializer.value)
            {
              // Non-literal default - don't add conformance
              return []
            }
            if let parameterInfo = try? extractParameterInfo(from: varDecl, binding: binding, context: context) {
              parameterInfos.append(parameterInfo)
            }
          }
        }
      }

      // Capture perform method for signature validation
      // Must mirror the checks in `extractToolInfo`, or the conformance extension
      // will be generated even when the member macro refuses to produce `_perform`.
      if let funcDecl = member.decl.as(FunctionDeclSyntax.self),
         funcDecl.name.text == "perform"
      {
        if funcDecl.modifiers.contains(where: { $0.name.text == "static" }) {
          return []
        }
        performDecl = funcDecl
      }
    }

    // Don't add conformance if basic validation fails
    guard hasName, hasDescription else {
      return []
    }

    // Validate tool name
    if let name = toolName, validateToolName(name) != nil {
      return []
    }

    if !duplicateParameterKeys(in: parameterInfos).isEmpty {
      return []
    }

    // Reject if perform() is missing — otherwise the extension would claim conformance
    // while the MemberMacro failed to generate `_perform`, producing a second error.
    guard let performDecl else {
      return []
    }

    // Don't add conformance if perform() has an invalid signature
    let params = performDecl.signature.parameterClause.parameters
    let hasAsync = performDecl.signature.effectSpecifiers?.asyncSpecifier != nil
    let hasThrows = performDecl.signature.effectSpecifiers?.throwsClause != nil
    let hasReturn = performDecl.signature.returnClause != nil
    if !params.isEmpty || !hasAsync || !hasThrows || !hasReturn {
      return []
    }

    // Add ToolSpec and Sendable conformance (fully qualified for compatibility with MCP imports)
    let extensionDecl: DeclSyntax = """
    extension \(type): AI.ToolSpec, Sendable {}
    """

    guard let ext = extensionDecl.as(ExtensionDeclSyntax.self) else {
      return []
    }

    return [ext]
  }

  // MARK: - Tool Info Extraction

  private struct ToolInfo {
    var name: String
    var description: String
    var parameters: [ParameterInfo]
    var outputType: String
    var hasTitle: Bool
    var strictSchema: Bool
  }

  private struct ParameterInfo {
    var propertyName: String
    var jsonKey: String
    var typeName: String
    var isOptional: Bool
    var hasDefault: Bool
    var defaultValue: String?
    var title: String?
    var description: String?
    var minLength: String?
    var maxLength: String?
    var minimum: String?
    var maximum: String?
    var declSyntax: VariableDeclSyntax? // For pointing diagnostics at the offending @Parameter
  }

  /// Emits a node-level error diagnostic and throws `AbortMacroExpansion` so the
  /// outer expansion returns empty results without producing a second attribute-level error.
  private static func diagnoseAndAbort(
    message: String,
    node: some SyntaxProtocol,
    in context: some MacroExpansionContext,
  ) throws -> Never {
    context.diagnose(Diagnostic(
      node: Syntax(node),
      message: ToolMacroDiagnostic.error(message),
    ))
    throw AbortMacroExpansion()
  }

  private static func extractToolInfo(
    from structDecl: StructDeclSyntax,
    context: some MacroExpansionContext,
  ) throws -> ToolInfo {
    var name: String?
    var nameSyntax: SyntaxProtocol?
    var description: String?
    var parameters: [ParameterInfo] = []
    var outputType = "String"
    var hasTitle = false
    var strictSchema = false
    var performDecl: FunctionDeclSyntax?

    for member in structDecl.memberBlock.members {
      let decl = member.decl

      // Look for static let name/description/title
      if let varDecl = decl.as(VariableDeclSyntax.self),
         varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
        // Reject @Parameter on static properties
        if hasParameterAttribute(varDecl) {
          let propertyName = varDecl.bindings.first?.pattern.as(IdentifierPatternSyntax.self)?.identifier.text ?? "?"
          try diagnoseAndAbort(
            message: "@Parameter cannot be applied to static property '\(propertyName)'. Tool parameters must be instance properties.",
            node: varDecl,
            in: context,
          )
        }

        for binding in varDecl.bindings {
          guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self) else {
            continue
          }

          let propertyName = identifier.identifier.text

          if propertyName == "name",
             let initializer = binding.initializer,
             let stringLiteral = initializer.value.as(StringLiteralExprSyntax.self),
             let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
          {
            name = segment.content.text
            nameSyntax = stringLiteral
          }

          if propertyName == "description",
             let initializer = binding.initializer,
             let stringLiteral = initializer.value.as(StringLiteralExprSyntax.self),
             let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
          {
            description = segment.content.text
          }

          if propertyName == "title" {
            hasTitle = true
          }

          if propertyName == "strictSchema",
             let initializer = binding.initializer,
             let boolLiteral = initializer.value.as(BooleanLiteralExprSyntax.self)
          {
            strictSchema = boolLiteral.literal.text == "true"
          }
        }
      }

      // Look for @Parameter properties
      if let varDecl = decl.as(VariableDeclSyntax.self),
         !varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
        if hasParameterAttribute(varDecl) {
          for binding in varDecl.bindings {
            if let paramInfo = try extractParameterInfo(from: varDecl, binding: binding, context: context) {
              parameters.append(paramInfo)
            }
          }
        }
      }

      // Look for perform method to get output type
      if let funcDecl = decl.as(FunctionDeclSyntax.self),
         funcDecl.name.text == "perform"
      {
        if funcDecl.modifiers.contains(where: { $0.name.text == "static" }) {
          try diagnoseAndAbort(
            message: "@Tool requires 'perform()' to be an instance method, not static.",
            node: funcDecl.name,
            in: context,
          )
        }
        performDecl = funcDecl
        if let returnClause = funcDecl.signature.returnClause {
          outputType = returnClause.type.trimmedDescription
        }
      }
    }

    guard let toolName = name else {
      throw ToolMacroError.missingName
    }

    guard let toolDescription = description else {
      throw ToolMacroError.missingDescription
    }

    guard let performDecl else {
      throw ToolMacroError.missingPerformMethod
    }

    // Validate perform() signature
    let performParams = performDecl.signature.parameterClause.parameters
    if !performParams.isEmpty {
      try diagnoseAndAbort(
        message: "@Tool requires 'perform()' to take no arguments. Use '@Parameter' properties on the struct to declare inputs.",
        node: performDecl.signature.parameterClause,
        in: context,
      )
    }
    if performDecl.signature.effectSpecifiers?.asyncSpecifier == nil {
      try diagnoseAndAbort(
        message: "@Tool requires 'perform()' to be marked 'async'",
        node: performDecl.name,
        in: context,
      )
    }
    if performDecl.signature.effectSpecifiers?.throwsClause == nil {
      try diagnoseAndAbort(
        message: "@Tool requires 'perform()' to be marked 'throws'",
        node: performDecl.name,
        in: context,
      )
    }
    if performDecl.signature.returnClause == nil {
      try diagnoseAndAbort(
        message: "@Tool requires 'perform()' to return a value conforming to 'ToolOutput'",
        node: performDecl.name,
        in: context,
      )
    }

    // Warn if perform() has an explicit access modifier more restrictive than the struct.
    // Only flag explicit modifiers; an unmarked `perform()` is the canonical case this
    // macro is designed for, so it must not produce a diagnostic.
    let structAccess = accessLevelRank(of: structDecl.modifiers)
    if let performAccess = explicitAccessLevelRank(of: performDecl.modifiers),
       performAccess < structAccess
    {
      context.diagnose(Diagnostic(
        node: Syntax(performDecl),
        message: ToolMacroDiagnostic.warning(
          "'perform()' has more restrictive access (\(accessLevelName(performAccess))) than the enclosing struct (\(accessLevelName(structAccess))). If the return type is similarly restricted, the generated '_perform()' bridge will fail to compile.",
        ),
      ))
    }

    // Validate tool name
    if let validationError = validateToolName(toolName) {
      throw ToolMacroError.invalidToolName(validationError)
    }

    // Reject duplicate @Parameter keys (silent overwrite in the schema otherwise).
    // Emit one diagnostic per offending property so the user can locate each one.
    let duplicates = Set(duplicateParameterKeys(in: parameters))
    if !duplicates.isEmpty {
      for param in parameters where duplicates.contains(param.jsonKey) {
        let node: any SyntaxProtocol = param.declSyntax ?? structDecl.name
        context.diagnose(Diagnostic(
          node: Syntax(node),
          message: ToolMacroDiagnostic.error(
            "Duplicate @Parameter key '\(param.jsonKey)'. Each @Parameter key must be unique.",
          ),
        ))
      }
      throw AbortMacroExpansion()
    }

    // Warn about tool name style issues
    if let styleWarning = toolNameStyleWarning(toolName),
       let syntax = nameSyntax
    {
      context.diagnose(Diagnostic(
        node: Syntax(syntax),
        message: ToolMacroDiagnostic.warning(styleWarning),
      ))
    }

    return ToolInfo(
      name: toolName,
      description: toolDescription,
      parameters: parameters,
      outputType: outputType,
      hasTitle: hasTitle,
      strictSchema: strictSchema,
    )
  }

  private static func extractParameterInfo(
    from varDecl: VariableDeclSyntax,
    binding: PatternBindingSyntax,
    context _: some MacroExpansionContext,
  ) throws -> ParameterInfo? {
    guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self) else {
      return nil
    }

    let propertyName = identifier.identifier.text
    var jsonKey = propertyName
    var typeName = "String"
    var isOptional = false
    var hasDefault = false
    var defaultValue: String?
    var paramTitle: String?
    var paramDescription: String?
    var minLength: String?
    var maxLength: String?
    var minimum: String?
    var maximum: String?

    // Get type annotation
    if let typeAnnotation = binding.typeAnnotation {
      let typeString = typeAnnotation.type.trimmedDescription
      typeName = typeString

      // Check if optional
      if typeString.hasSuffix("?") {
        isOptional = true
        typeName = String(typeString.dropLast())
      } else if typeString.hasPrefix("Optional<") {
        isOptional = true
        typeName = String(typeString.dropFirst(9).dropLast())
      }
    }

    // Check for default value
    if let initializer = binding.initializer {
      hasDefault = true
      defaultValue = initializer.value.trimmedDescription

      // Validate that default value is a literal
      if !isLiteralExpression(initializer.value) {
        throw ToolMacroError.nonLiteralDefaultValue(propertyName)
      }
    }

    // Extract @Parameter arguments
    for attr in varDecl.attributes {
      if case let .attribute(attrSyntax) = attr,
         isParameterAttribute(attr),
         let arguments = attrSyntax.arguments?.as(LabeledExprListSyntax.self)
      {
        for arg in arguments {
          let label = arg.label?.text

          switch label {
            case "key":
              if let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self),
                 let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
              {
                jsonKey = segment.content.text
              }
            case "title":
              if let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self),
                 let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
              {
                paramTitle = segment.content.text
              }
            case "description":
              if let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self),
                 let segment = stringLiteral.segments.first?.as(StringSegmentSyntax.self)
              {
                paramDescription = segment.content.text
              }
            case "minLength":
              minLength = arg.expression.trimmedDescription
            case "maxLength":
              maxLength = arg.expression.trimmedDescription
            case "minimum":
              minimum = arg.expression.trimmedDescription
            case "maximum":
              maximum = arg.expression.trimmedDescription
            default:
              break
          }
        }
      }
    }

    return ParameterInfo(
      propertyName: propertyName,
      jsonKey: jsonKey,
      typeName: typeName,
      isOptional: isOptional,
      hasDefault: hasDefault,
      defaultValue: defaultValue,
      title: paramTitle,
      description: paramDescription,
      minLength: minLength,
      maxLength: maxLength,
      minimum: minimum,
      maximum: maximum,
      declSyntax: varDecl,
    )
  }

  // MARK: - Code Generation

  private static func generateToolProperty(
    toolInfo: ToolInfo,
    accessPrefix: String,
  ) -> DeclSyntax {
    let descriptorEntries = toolInfo.parameters.map { param in
      let defaultValueLiteral = if let defaultVal = param.defaultValue {
        convertToValueLiteral(defaultVal, type: param.typeName)
      } else {
        "nil"
      }
      let titleLiteral = param.title.map { "\"\($0)\"" } ?? "nil"
      let descriptionLiteral = param.description.map { "\"\($0)\"" } ?? "nil"
      let minLengthLiteral = param.minLength ?? "nil"
      let maxLengthLiteral = param.maxLength ?? "nil"
      let minimumLiteral = param.minimum ?? "nil"
      let maximumLiteral = param.maximum ?? "nil"

      return """
      AITool.ToolMacroSupport.makeSchemaParameterDescriptor(
        name: "\(param.jsonKey)",
        title: \(titleLiteral),
        description: \(descriptionLiteral),
        schema: \(param.typeName).schema,
        isOptional: \(param.isOptional),
        hasDefault: \(param.hasDefault),
        defaultValue: \(defaultValueLiteral),
        minLength: \(minLengthLiteral),
        maxLength: \(maxLengthLiteral),
        minimum: \(minimumLiteral),
        maximum: \(maximumLiteral)
      )
      """
    }.joined(separator: ",\n                ")

    let descriptorsLiteral = descriptorEntries.isEmpty ? "[]" : "[\n                \(descriptorEntries)\n            ]"

    let strictValidationStmt = toolInfo.strictSchema
      ? "try! AITool.ToolMacroSupport.validateStrictCompatibility(_schemaBuild.schema, toolName: name)\n        "
      : ""
    return """
    \(raw: accessPrefix)static var tool: AI.Tool {
        let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
            parameters: \(raw: descriptorsLiteral)
        )
        \(raw: strictValidationStmt)return AI.Tool(
            name: name,
            description: description,
            title: \(raw: toolInfo.hasTitle ? "title" : "name"),
            inputSchema: _schemaBuild.schema,
            resultTypes: \(raw: toolInfo.outputType).resultTypes,
            schemaBuildErrorMessage: _schemaBuild.errorMessage,
            execute: { parameters in
                let instance = try Self.parse(from: parameters)
                let output = try await instance._perform()
                return output.toToolResult()
            }
        )
    }
    """
  }

  private static func generateParseMethod(toolInfo: ToolInfo, accessPrefix: String) -> DeclSyntax {
    // For tools with no parameters, generate a simple parse method
    if toolInfo.parameters.isEmpty {
      return """
      \(raw: accessPrefix)static func parse(from arguments: [String: AI.Value]) throws -> Self {
          Self()
      }
      """
    }

    var parseStatements: [String] = []

    for param in toolInfo.parameters {
      let key = param.jsonKey
      let prop = param.propertyName
      let type = param.typeName

      if param.isOptional {
        // Optional: parse only if present and non-null; leave as nil otherwise.
        parseStatements.append(
          "if let _value = _args[\"\(key)\"], !_value.isNull { _instance.\(prop) = try AITool.ToolMacroSupport.parseParameter(\(type).schema, from: _value, parameterName: \"\(key)\") }",
        )
      } else if param.hasDefault {
        // Has default: only parse if key is present and non-null; otherwise keep the struct's default.
        parseStatements.append(
          "if let _value = _args[\"\(key)\"], !_value.isNull { _instance.\(prop) = try AITool.ToolMacroSupport.parseParameter(\(type).schema, from: _value, parameterName: \"\(key)\") }",
        )
      } else {
        // Required: must be present; parse throws with detail if the value shape is wrong.
        parseStatements.append(
          "guard let _\(prop)Value = _args[\"\(key)\"] else { throw ToolError.invalidParameterType(parameter: \"\(key)\", expected: \"\(type)\", got: \"nil\") }",
        )
        parseStatements.append(
          "_instance.\(prop) = try AITool.ToolMacroSupport.parseParameter(\(type).schema, from: _\(prop)Value, parameterName: \"\(key)\")",
        )
      }
    }

    let statements = parseStatements.joined(separator: "\n    ")

    return """
    \(raw: accessPrefix)static func parse(from arguments: [String: AI.Value]) throws -> Self {
        var _instance = Self()
        let _args = arguments
        \(raw: statements)
        return _instance
    }
    """
  }

  // MARK: - Type Mapping Helpers

  private static func convertToValueLiteral(_ value: String, type: String) -> String {
    if value == "nil" {
      return "AI.Value.null"
    }

    return switch type {
      case "String":
        "AI.Value.string(\(value))"
      case "Int":
        "AI.Value.int(\(value))"
      case "Double":
        "AI.Value.double(\(value))"
      case "Bool":
        "AI.Value.bool(\(value))"
      default:
        if value == "true" || value == "false" {
          "AI.Value.bool(\(value))"
        } else if value.contains(".") {
          "AI.Value.double(\(value))"
        } else if let _ = Int(value) {
          "AI.Value.int(\(value))"
        } else {
          "AI.Value.string(\(value))"
        }
    }
  }
}

// MARK: - Errors

enum ToolMacroError: Error, CustomStringConvertible {
  case notAStruct
  case missingName
  case missingDescription
  case missingPerformMethod
  case invalidToolName(String)
  case nonLiteralDefaultValue(String)

  var description: String {
    switch self {
      case .notAStruct:
        "@Tool can only be applied to structs"
      case .missingName:
        "@Tool requires 'static let name: String' property"
      case .missingDescription:
        "@Tool requires 'static let description: String' property"
      case .missingPerformMethod:
        "@Tool requires a 'perform' method (e.g., 'func perform() async throws -> String')"
      case let .invalidToolName(reason):
        "Invalid tool name: \(reason)"
      case let .nonLiteralDefaultValue(param):
        "Parameter '\(param)' has a non-literal default value. Only literal values (numbers, strings, booleans) are supported. For complex defaults, make the parameter optional and handle the default in perform()."
    }
  }
}

/// Thrown after emitting a node-level diagnostic to silently abort macro expansion
/// without a second attribute-level error. Caught by the outer `expansion` function.
private struct AbortMacroExpansion: Error {}

// MARK: - Diagnostics

struct ToolMacroDiagnostic: DiagnosticMessage {
  let message: String
  let diagnosticID: MessageID
  let severity: DiagnosticSeverity

  static func warning(_ message: String) -> ToolMacroDiagnostic {
    ToolMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "ToolMacro", id: "warning"),
      severity: .warning,
    )
  }

  static func error(_ message: String) -> ToolMacroDiagnostic {
    ToolMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "ToolMacro", id: "error"),
      severity: .error,
    )
  }
}

// MARK: - Access Level Helpers

extension ToolMacro {
  /// Ranks Swift access levels from most to least restrictive.
  /// Missing modifier defaults to internal.
  static func accessLevelRank(of modifiers: DeclModifierListSyntax) -> Int {
    explicitAccessLevelRank(of: modifiers) ?? 2
  }

  /// Returns the explicit access level rank, or nil if no access modifier is present.
  static func explicitAccessLevelRank(of modifiers: DeclModifierListSyntax) -> Int? {
    for modifier in modifiers {
      switch modifier.name.text {
        case "private": return 0
        case "fileprivate": return 1
        case "internal": return 2
        case "package": return 3
        case "public": return 4
        case "open": return 5
        default: continue
      }
    }
    return nil
  }

  static func accessLevelName(_ rank: Int) -> String {
    switch rank {
      case 0: "private"
      case 1: "fileprivate"
      case 2: "internal"
      case 3: "package"
      case 4: "public"
      case 5: "open"
      default: "internal"
    }
  }
}

// MARK: - Duplicate Parameter Key Detection

extension ToolMacro {
  private static func duplicateParameterKeys(in parameters: [ParameterInfo]) -> [String] {
    var counts: [String: Int] = [:]
    for parameter in parameters {
      counts[parameter.jsonKey, default: 0] += 1
    }
    return counts
      .filter { $0.value > 1 }
      .map(\.key)
      .sorted()
  }
}

// MARK: - Tool Name Validation

extension ToolMacro {
  /// Valid characters for tool names: A-Z, a-z, 0-9, _, -, .
  private static let validToolNameCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.",
  )

  static func validateToolName(_ name: String) -> String? {
    if name.isEmpty {
      return "Tool name cannot be empty"
    }
    if name.count > 128 {
      return "Tool name exceeds maximum length of 128 characters (got \(name.count))"
    }

    let nameCharSet = CharacterSet(charactersIn: name)
    if !nameCharSet.isSubset(of: validToolNameCharacters) {
      let invalidChars = name.unicodeScalars.filter { !validToolNameCharacters.contains($0) }
      let invalidStr = String(String.UnicodeScalarView(invalidChars))
      return "Tool name contains invalid characters: '\(invalidStr)'. Only A-Z, a-z, 0-9, _, -, . are allowed"
    }

    return nil
  }

  static func toolNameStyleWarning(_ name: String) -> String? {
    if name.hasPrefix("-") || name.hasPrefix(".") {
      return "Tool name '\(name)' starts with '\(name.first!)' which may cause compatibility issues"
    }
    if name.hasSuffix("-") || name.hasSuffix(".") {
      return "Tool name '\(name)' ends with '\(name.last!)' which may cause compatibility issues"
    }
    return nil
  }
}

// MARK: - Attribute Matching

extension ToolMacro {
  /// Checks if an attribute is the `@Parameter` attribute.
  /// Recognizes both `@Parameter` and `@AI.Parameter` forms for compatibility
  /// when AI module is imported alongside other frameworks that also define Parameter.
  static func isParameterAttribute(_ attr: AttributeListSyntax.Element) -> Bool {
    guard case let .attribute(attrSyntax) = attr else { return false }

    // Check for simple `@Parameter`
    if let identifier = attrSyntax.attributeName.as(IdentifierTypeSyntax.self) {
      return identifier.name.text == "Parameter"
    }

    // Check for qualified `@AI.Parameter`
    if let memberType = attrSyntax.attributeName.as(MemberTypeSyntax.self),
       let baseIdentifier = memberType.baseType.as(IdentifierTypeSyntax.self)
    {
      return baseIdentifier.name.text == "AI" && memberType.name.text == "Parameter"
    }

    return false
  }

  /// Checks if a variable declaration has the `@Parameter` attribute.
  static func hasParameterAttribute(_ varDecl: VariableDeclSyntax) -> Bool {
    varDecl.attributes.contains { isParameterAttribute($0) }
  }
}

// MARK: - Default Value Validation

extension ToolMacro {
  static func isLiteralExpression(_ expr: ExprSyntax) -> Bool {
    if expr.is(IntegerLiteralExprSyntax.self) {
      return true
    }
    if expr.is(FloatLiteralExprSyntax.self) {
      return true
    }
    if expr.is(StringLiteralExprSyntax.self) {
      return true
    }
    if expr.is(BooleanLiteralExprSyntax.self) {
      return true
    }
    if expr.is(NilLiteralExprSyntax.self) {
      return true
    }
    if let prefixExpr = expr.as(PrefixOperatorExprSyntax.self),
       prefixExpr.operator.text == "-"
    {
      return isLiteralExpression(prefixExpr.expression)
    }
    return false
  }
}
