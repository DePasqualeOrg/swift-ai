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

    // Generate _perform() bridging to the user's perform().
    // For Void returns, run the handler, discard the empty tuple, and emit the
    // VoidOutput sentinel so the wire shape matches Optional<T>.none.
    if toolInfo.returnsVoid {
      members.append("""
      \(raw: accessPrefix)func _perform() async throws -> \(raw: toolInfo.outputType) {
          try await perform()
          return AI.VoidOutput()
      }
      """)
    } else {
      members.append("""
      \(raw: accessPrefix)func _perform() async throws -> \(raw: toolInfo.outputType) {
          try await perform()
      }
      """)
    }

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
    var performDecls: [FunctionDeclSyntax] = []

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
            // `hasName` / `hasDescription` must mirror `extractToolInfo`'s
            // plain-string-literal requirement. Flipping them on for any
            // initializer would let the extension add ToolSpec conformance
            // when the member macro refuses to generate the required members,
            // producing a secondary "does not conform" error.
            if propName == "name",
               let initializer = binding.initializer,
               let stringLiteral = initializer.value.as(StringLiteralExprSyntax.self),
               let content = plainLiteralStringContent(stringLiteral)
            {
              hasName = true
              toolName = content
            }
            if propName == "description",
               let initializer = binding.initializer,
               let stringLiteral = initializer.value.as(StringLiteralExprSyntax.self),
               plainLiteralStringContent(stringLiteral) != nil
            {
              hasDescription = true
            }
            // Mirror the MemberMacro's "title must be a string literal (when
            // initialized)" check — bail silently so the MemberMacro is the sole
            // source of the node-level diagnostic. Computed `title` (no initializer)
            // is left to the type checker, matching the MemberMacro behavior.
            if propName == "title", let initializer = binding.initializer {
              let isValidLiteral = initializer.value.as(StringLiteralExprSyntax.self)
                .flatMap(plainLiteralStringContent(_:)) != nil
              if !isValidLiteral {
                return []
              }
            }
            // Mirror the MemberMacro's "strictSchema must be a boolean literal" check.
            if propName == "strictSchema",
               let initializer = binding.initializer,
               !initializer.value.is(BooleanLiteralExprSyntax.self)
            {
              return []
            }
          }
        }
      }

      // Check for @Parameter properties with non-literal defaults
      if let varDecl = member.decl.as(VariableDeclSyntax.self),
         !varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
        if hasParameterAttribute(varDecl) {
          // Mirror the MemberMacro's "@Parameter key/title/description must be a
          // single-segment string literal" check — bail silently so the MemberMacro
          // is the sole source of the node-level diagnostic.
          if hasNonLiteralParameterMetadata(varDecl) {
            return []
          }
          // Mirror the MemberMacro's "@Parameter must be a mutable stored var" check.
          if varDecl.bindingSpecifier.text != "var" {
            return []
          }
          for binding in varDecl.bindings {
            // Mirror the MemberMacro's "explicit type annotation required"
            // check — bail silently so the MemberMacro is the sole source
            // of the node-level diagnostic.
            if binding.typeAnnotation == nil {
              return []
            }
            // Mirror the MemberMacro's "no computed properties" check.
            if hasNonStoredAccessorBlock(binding) {
              return []
            }
            if let initializer = binding.initializer,
               !isLiteralExpression(initializer.value)
            {
              // Non-literal default - don't add conformance
              return []
            }
            // Catches only `ToolMacroError` (parameter-shape problems the
            // MemberMacro will report). `AbortMacroExpansion` is re-thrown so
            // an abort from upstream still cancels extension generation, and
            // any other error propagates rather than being silently swallowed
            // — a bare `try?` would hide both.
            do {
              if let parameterInfo = try extractParameterInfo(from: varDecl, binding: binding, context: context) {
                parameterInfos.append(parameterInfo)
              }
            } catch is ToolMacroError {
              // MemberMacro path will diagnose; skip this parameter and continue.
            }
          }
        }
      }

      // Collect perform methods (must mirror the checks in `extractToolInfo`).
      // We collect all of them so the duplicate-overload check below can bail.
      if let funcDecl = member.decl.as(FunctionDeclSyntax.self),
         funcDecl.name.text == "perform"
      {
        if funcDecl.modifiers.contains(where: { $0.name.text == "static" }) {
          return []
        }
        performDecls.append(funcDecl)
      }
    }

    // Mirror the MemberMacro's "exactly one perform" check — bail silently so
    // the MemberMacro's diagnostic is the only one the user sees.
    if performDecls.count > 1 {
      return []
    }
    let performDecl = performDecls.first

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

    // Reject if perform() signature is invalid. Uses the same validator as the
    // MemberMacro so the two paths can't drift.
    if case .invalid = validatePerformSignature(performDecl) {
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
    var returnsVoid: Bool
    var resultTypesOverride: ExprSyntax?
    var hasTitle: Bool
    var strictSchema: Bool
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
    var performDecls: [FunctionDeclSyntax] = []

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

          if propertyName == "name", let initializer = binding.initializer {
            name = try requireStaticStringLiteralProperty(
              initializerValue: initializer.value,
              propertyName: "name",
              context: context,
            )
            nameSyntax = initializer.value
          }

          if propertyName == "description", let initializer = binding.initializer {
            description = try requireStaticStringLiteralProperty(
              initializerValue: initializer.value,
              propertyName: "description",
              context: context,
            )
          }

          if propertyName == "title" {
            hasTitle = true
            // If the user provided a stored-property initializer, validate it's a
            // string literal up front so they get a targeted diagnostic instead of
            // a confusing downstream error like "cannot convert Int to String" in
            // macro-generated code that references `Self.title: String`. Computed
            // properties (no initializer) are left to Swift's type checker.
            if let initializer = binding.initializer {
              _ = try requireStaticStringLiteralProperty(
                initializerValue: initializer.value,
                propertyName: "title",
                context: context,
              )
            }
          }

          if propertyName == "strictSchema", let initializer = binding.initializer {
            strictSchema = try requireStaticBooleanLiteralProperty(
              initializerValue: initializer.value,
              propertyName: "strictSchema",
              context: context,
            )
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

      // Collect all `perform` declarations. We pick after the loop so we can
      // explicitly diagnose multiple overloads instead of silently letting the
      // last declaration win, which would make the generated bridge depend on
      // source order.
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
        performDecls.append(funcDecl)
      }
    }

    guard let toolName = name else {
      throw ToolMacroError.missingName
    }

    guard let toolDescription = description else {
      throw ToolMacroError.missingDescription
    }

    // Reject multiple `perform` declarations explicitly. Helper methods that
    // happen to share the name (e.g. `perform(verbose:)`) need to be renamed
    // — the macro picks one to bridge into `_perform()` and would otherwise
    // pick whichever appears last in source order.
    if performDecls.count > 1 {
      try diagnoseAndAbort(
        message: "@Tool requires exactly one 'perform' method, but found \(performDecls.count). Rename helper methods to avoid the 'perform' name.",
        node: performDecls[1].name,
        in: context,
      )
    }

    guard let performDecl = performDecls.first else {
      throw ToolMacroError.missingPerformMethod
    }

    let returnsVoid = performReturnsVoid(performDecl)
    if returnsVoid {
      outputType = "AI.VoidOutput"
    } else if let returnClause = performDecl.signature.returnClause {
      outputType = returnClause.type.trimmedDescription
    }
    let resultTypesOverride = findResultTypesOverride(in: structDecl)

    // Validate perform() signature using the shared validator so MemberMacro and
    // ExtensionMacro can never disagree on what's accepted.
    if case let .invalid(message, blameNode) = validatePerformSignature(performDecl) {
      try diagnoseAndAbort(message: message, node: blameNode, in: context)
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
      returnsVoid: returnsVoid,
      resultTypesOverride: resultTypesOverride,
      hasTitle: hasTitle,
      strictSchema: strictSchema,
    )
  }

  // MARK: - Code Generation

  private static func generateToolProperty(
    toolInfo: ToolInfo,
    accessPrefix: String,
  ) -> DeclSyntax {
    let descriptorEntries = toolInfo.parameters.map { param in
      let defaultValueLiteral = if let defaultExpr = param.defaultValueExpr {
        convertToValueLiteral(defaultExpr)
      } else {
        "nil"
      }
      let titleLiteral = swiftStringLiteral(param.title)
      let descriptionLiteral = swiftStringLiteral(param.description)
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

    // A strict-schema mismatch is a programmer error: surface it with a tool-named
    // precondition rather than a bare `try!` trap so the crash log identifies the tool.
    let strictValidationStmt = toolInfo.strictSchema
      ? """
      do {
                try AITool.ToolMacroSupport.validateStrictCompatibility(_schemaBuild.schema, toolName: name)
            } catch {
                preconditionFailure("Strict schema validation failed for tool '\\(name)': \\(error)")
            }

      """
      : ""
    let resultTypesExpr = toolInfo.resultTypesOverride.map(\.trimmedDescription)
      ?? "\(toolInfo.outputType).resultTypes"

    return """
    \(raw: accessPrefix)static var tool: AI.Tool {
        let _schemaBuild = AITool.ToolMacroSupport.buildObjectSchemaResult(
            parameters: \(raw: descriptorsLiteral)
        )
        \(raw: strictValidationStmt)return AI.Tool(
            name: name,
            description: description,
            title: title,
            inputSchema: _schemaBuild.schema,
            resultTypes: \(raw: resultTypesExpr),
            outputSchema: AI.AISchema.outputSchema(for: \(raw: toolInfo.outputType).self),
            schemaBuildErrorMessage: _schemaBuild.errorMessage,
            execute: { parameters in
                let instance = try Self.parse(from: parameters)
                let output = try await instance._perform()
                return try output.toToolResult()
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
        // Use `missingRequiredParameter` for absent keys — `invalidParameterType` is
        // reserved for keys that are present but the wrong shape.
        parseStatements.append(
          "guard let _\(prop)Value = _args[\"\(key)\"] else { throw ToolDispatchError.missingRequiredParameter(\"\(key)\") }",
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

  /// Converts a literal ExprSyntax default value into the matching `AI.Value` case.
  ///
  /// Dispatch is by the literal node's syntax type, not by scanning its string form —
  /// a parameter like `var version: Version = "1.0"` (`ExpressibleByStringLiteral`) is
  /// a string literal in the source and must stay a `.string`, even though the text
  /// happens to parse as a double.
  private static func convertToValueLiteral(_ expr: ExprSyntax) -> String {
    if expr.is(NilLiteralExprSyntax.self) {
      return "AI.Value.null"
    }
    if expr.is(BooleanLiteralExprSyntax.self) {
      return "AI.Value.bool(\(expr.trimmedDescription))"
    }
    if expr.is(IntegerLiteralExprSyntax.self) {
      return "AI.Value.int(\(expr.trimmedDescription))"
    }
    if expr.is(FloatLiteralExprSyntax.self) {
      return "AI.Value.double(\(expr.trimmedDescription))"
    }
    if expr.is(StringLiteralExprSyntax.self) {
      // Preserve the literal verbatim — it's already a well-formed Swift string literal.
      return "AI.Value.string(\(expr.trimmedDescription))"
    }
    if let prefix = expr.as(PrefixOperatorExprSyntax.self), prefix.operator.text == "-" {
      if prefix.expression.is(IntegerLiteralExprSyntax.self) {
        return "AI.Value.int(\(expr.trimmedDescription))"
      }
      if prefix.expression.is(FloatLiteralExprSyntax.self) {
        return "AI.Value.double(\(expr.trimmedDescription))"
      }
    }
    // Contract: `isLiteralExpression` gates every call to this function, so the two
    // must accept exactly the same set of syntax kinds. Trapping here means a drift
    // between the two surfaces as a plugin crash at compile time (with this message
    // in stderr), not as a silent `.null` default that corrupts the generated schema.
    preconditionFailure("convertToValueLiteral: unmapped literal kind \(expr.syntaxNodeType); add a case here or update isLiteralExpression")
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

// MARK: - Repo-specific knob for shared helpers

extension ToolMacro {
  /// Module name to recognize in qualified `@<Module>.Parameter` attributes.
  /// Read by `isParameterAttribute` in `ToolMacroSharedHelpers.swift`.
  static let parameterAttributeModuleName = "AI"
}

// MARK: - Perform Signature Validation

extension ToolMacro {
  /// Outcome of validating a `perform()` declaration. The MemberMacro turns `.invalid`
  /// into a node-level diagnostic; the ExtensionMacro silently bails so the user only
  /// sees the MemberMacro's diagnostic, not a duplicate cascade.
  enum PerformValidation {
    case valid
    case invalid(message: String, blameNode: any SyntaxProtocol)
  }

  /// Validates the signature of a `perform()` method. Both expansion paths share this
  /// so the rules can never drift between them.
  static func validatePerformSignature(_ decl: FunctionDeclSyntax) -> PerformValidation {
    let params = decl.signature.parameterClause.parameters
    if !params.isEmpty {
      return .invalid(
        message: "@Tool requires 'perform()' to take no arguments. Use '@Parameter' properties on the struct to declare inputs.",
        blameNode: decl.signature.parameterClause,
      )
    }
    if decl.signature.effectSpecifiers?.asyncSpecifier == nil {
      return .invalid(
        message: "@Tool requires 'perform()' to be marked 'async'",
        blameNode: decl.name,
      )
    }
    if decl.signature.effectSpecifiers?.throwsClause == nil {
      return .invalid(
        message: "@Tool requires 'perform()' to be marked 'throws'",
        blameNode: decl.name,
      )
    }
    // Void returns are allowed — the macro normalizes them to AI.VoidOutput.
    return .valid
  }

  /// Returns true when the `perform()` declaration's return clause is one of
  /// the canonical Void spellings, or absent. Same syntactic check swift-mcp
  /// uses; `typealias Nothing = Void` used as `-> Nothing` won't be detected.
  static func performReturnsVoid(_ decl: FunctionDeclSyntax) -> Bool {
    guard let returnClause = decl.signature.returnClause else {
      return true
    }
    switch returnClause.type.trimmedDescription {
      case "Void", "()", "Swift.Void":
        return true
      default:
        return false
    }
  }

  /// Detects a `static let resultTypes: Set<ToolResult.ValueType>?` declaration
  /// on the `@Tool` struct. When present, the macro emits this expression
  /// verbatim into the generated `Tool` initializer instead of the type-level
  /// default from `\(outputType).resultTypes`. Used by authors to narrow the
  /// declared types on, e.g., a `Media`-returning tool that only emits images.
  static func findResultTypesOverride(in structDecl: StructDeclSyntax) -> ExprSyntax? {
    for member in structDecl.memberBlock.members {
      guard let varDecl = member.decl.as(VariableDeclSyntax.self),
            varDecl.modifiers.contains(where: { $0.name.text == "static" })
      else { continue }
      for binding in varDecl.bindings {
        guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self),
              identifier.identifier.text == "resultTypes",
              let initializer = binding.initializer
        else { continue }
        return initializer.value
      }
    }
    return nil
  }
}
