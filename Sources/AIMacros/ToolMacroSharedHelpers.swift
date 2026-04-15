// Copyright © Anthony DePasquale

// SHARED HELPERS — kept in lockstep with:
//   ../../../forked/swift-mcp/Sources/MCPMacros/ToolMacroSharedHelpers.swift
//
// If you change anything here, mirror it there. The two files should remain
// equivalent modulo indentation (this repo uses 2-space, swift-mcp uses 4-space)
// and the path comment above.
//
// What belongs in this file: mechanically generic SwiftSyntax helpers that have no
// MCP-specific or AI-specific behavior. The one repo-specific knob is
// `ToolMacro.parameterAttributeModuleName`, defined separately in each repo's
// `ToolMacro.swift`.
//
// What does NOT belong here — the authoritative list of symbols that are
// deliberately divergent between the two repos:
//   - `validatePerformSignature` / `PerformValidation` — MCP allows a
//     `context: HandlerContext` parameter, AI doesn't.
//   - `ToolMacroError` enum — MCP has `duplicateAnnotation`, AI doesn't.
//   - `convertToValueLiteral` / `generateToolDefinition` / `generateToolProperty`
//     / `generateParseMethod` — code generators emit different output types
//     (`MCP.Value` / `MCP.Tool` vs `AI.Value` / `AI.Tool`) and call different
//     runtime support APIs.
//   - `extractToolInfo` and the `ToolInfo` struct — different field sets
//     (MCP has `annotations`/`hasContextParameter`; AI has `hasTitle`).
//   - `extractAnnotationNames` / `validateAnnotations` / `hasDuplicateAnnotations`
//     — MCP only; AI has no annotations concept.
// Adding any repo-specific behavior to a shared helper breaks that invariant,
// so push the divergence back into `ToolMacro.swift` via a per-repo extension.

import Foundation
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

// MARK: - Diagnostics

/// Thrown after emitting a node-level diagnostic to silently abort macro expansion
/// without a second attribute-level error. Caught by the outer `expansion` function.
struct AbortMacroExpansion: Error {}

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

extension ToolMacro {
  /// Emits a node-level error diagnostic and throws `AbortMacroExpansion` so the
  /// outer expansion returns empty results without producing a second attribute-level error.
  static func diagnoseAndAbort(
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
}

// MARK: - Parameter Info

struct ParameterInfo {
  var propertyName: String
  var jsonKey: String
  var typeName: String
  var isOptional: Bool
  var hasDefault: Bool
  var defaultValueExpr: ExprSyntax?
  var title: String?
  var description: String?
  var minLength: String?
  var maxLength: String?
  var minimum: String?
  var maximum: String?
  var declSyntax: VariableDeclSyntax? // For pointing diagnostics at the offending @Parameter
}

extension ToolMacro {
  static func extractParameterInfo(
    from varDecl: VariableDeclSyntax,
    binding: PatternBindingSyntax,
    context: some MacroExpansionContext,
  ) throws -> ParameterInfo? {
    guard let identifier = binding.pattern.as(IdentifierPatternSyntax.self) else {
      return nil
    }

    let propertyName = identifier.identifier.text

    // An explicit type annotation is required: the schema, parse path, and
    // `_perform` bridge all need the concrete type. Type inference from a
    // default value (e.g. `@Parameter var count = 1`) would leave `typeName`
    // at its "String" default and generate `String.schema` + `String`-typed
    // parsing, which then fails to compile against the actual inferred type.
    // Reject up front with a targeted diagnostic instead of letting the
    // generated code surface a confusing downstream error.
    guard binding.typeAnnotation != nil else {
      try diagnoseAndAbort(
        message: "@Parameter property '\(propertyName)' requires an explicit type annotation (e.g. 'var \(propertyName): Int').",
        node: binding,
        in: context,
      )
    }

    // The macro-generated `parse(from:)` writes the value back via
    // `_instance.<prop> = ...`, which only compiles for a mutable stored
    // property. `let` declarations and computed properties (with accessor
    // blocks) used to slip through and surface as a confusing
    // "cannot assign to property" error in synthesized code.
    if varDecl.bindingSpecifier.text != "var" {
      try diagnoseAndAbort(
        message: "@Parameter property '\(propertyName)' must be declared with 'var', not 'let' (the macro-generated parser assigns to it).",
        node: varDecl.bindingSpecifier,
        in: context,
      )
    }
    if hasNonStoredAccessorBlock(binding) {
      try diagnoseAndAbort(
        message: "@Parameter property '\(propertyName)' must be a stored property, not computed (the macro-generated parser assigns to it).",
        node: binding,
        in: context,
      )
    }

    var jsonKey = propertyName
    var typeName = "String"
    var isOptional = false
    var hasDefault = false
    var defaultValueExpr: ExprSyntax?
    var paramTitle: String?
    var paramDescription: String?
    var minLength: String?
    var maxLength: String?
    var minimum: String?
    var maximum: String?

    // Inspect the TypeSyntax directly so we don't misinterpret types like
    // `() -> String?` (function returning optional, not an optional function)
    // or `Swift.Optional<Int>`.
    if let typeAnnotation = binding.typeAnnotation {
      let type = typeAnnotation.type
      if let optionalType = type.as(OptionalTypeSyntax.self) {
        isOptional = true
        typeName = optionalType.wrappedType.trimmedDescription
      } else if let wrapped = unwrapExplicitOptional(type) {
        isOptional = true
        typeName = wrapped.trimmedDescription
      } else {
        typeName = type.trimmedDescription
      }
    }

    // Check for default value
    if let initializer = binding.initializer {
      hasDefault = true
      defaultValueExpr = initializer.value

      // Validate that default value is a literal
      if !isLiteralExpression(initializer.value) {
        throw ToolMacroError.nonLiteralDefaultValue(propertyName)
      }
    }

    // Extract @Parameter arguments. `key`, `title`, and `description` must be
    // single-segment string literals: the schema and parse paths bake those values
    // into generated Swift source, so any non-literal form (variable references,
    // member access, or interpolated strings) would otherwise be silently dropped
    // and cause the declared tool interface to diverge from the generated schema.
    for attr in varDecl.attributes {
      if case let .attribute(attrSyntax) = attr,
         isParameterAttribute(attr),
         let arguments = attrSyntax.arguments?.as(LabeledExprListSyntax.self)
      {
        for arg in arguments {
          let label = arg.label?.text

          switch label {
            case "key":
              jsonKey = try requireStringLiteralArgument(
                arg,
                label: "key",
                propertyName: propertyName,
                context: context,
              )
            case "title":
              paramTitle = try requireStringLiteralArgument(
                arg,
                label: "title",
                propertyName: propertyName,
                context: context,
              )
            case "description":
              paramDescription = try requireStringLiteralArgument(
                arg,
                label: "description",
                propertyName: propertyName,
                context: context,
              )
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
      defaultValueExpr: defaultValueExpr,
      title: paramTitle,
      description: paramDescription,
      minLength: minLength,
      maxLength: maxLength,
      minimum: minimum,
      maximum: maximum,
      declSyntax: varDecl,
    )
  }

  static func duplicateParameterKeys(in parameters: [ParameterInfo]) -> [String] {
    var counts: [String: Int] = [:]
    for parameter in parameters {
      counts[parameter.jsonKey, default: 0] += 1
    }
    return counts
      .filter { $0.value > 1 }
      .map(\.key)
      .sorted()
  }

  /// Returns true when the binding uses getter/setter-style accessors and is
  /// therefore non-stored for the macro's purposes. Property observers
  /// (`willSet` / `didSet`) are still stored properties and remain assignable
  /// by the generated parser, so they are allowed.
  static func hasNonStoredAccessorBlock(_ binding: PatternBindingSyntax) -> Bool {
    guard let accessorBlock = binding.accessorBlock else {
      return false
    }
    switch accessorBlock.accessors {
      case .getter:
        return true
      case let .accessors(accessors):
        return accessors.contains { accessor in
          switch accessor.accessorSpecifier.text {
            case "willSet", "didSet":
              false
            default:
              true
          }
        }
    }
  }

  /// Returns true if `varDecl` has a `@Parameter` attribute whose `key`, `title`, or
  /// `description` argument is anything other than a plain (non-interpolated) string
  /// literal. The ExtensionMacro uses this as a pre-check to bail out before calling
  /// `extractParameterInfo`, which would otherwise emit the same node-level diagnostic
  /// from both macro paths.
  static func hasNonLiteralParameterMetadata(_ varDecl: VariableDeclSyntax) -> Bool {
    for attr in varDecl.attributes {
      guard case let .attribute(attrSyntax) = attr,
            isParameterAttribute(attr),
            let arguments = attrSyntax.arguments?.as(LabeledExprListSyntax.self)
      else { continue }
      for arg in arguments {
        guard let label = arg.label?.text,
              label == "key" || label == "title" || label == "description"
        else { continue }
        guard let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self),
              plainLiteralStringContent(stringLiteral) != nil
        else { return true }
      }
    }
    return false
  }

  /// Concatenates all text segments of a string literal. Returns nil if any segment
  /// is an interpolation (`ExpressionSegmentSyntax`). Multi-segment non-interpolated
  /// literals (e.g. triple-quoted strings with embedded newlines) are preserved in full.
  static func plainLiteralStringContent(_ stringLiteral: StringLiteralExprSyntax) -> String? {
    var parts: [String] = []
    for segment in stringLiteral.segments {
      guard let stringSegment = segment.as(StringSegmentSyntax.self) else {
        return nil
      }
      parts.append(stringSegment.content.text)
    }
    return parts.joined()
  }

  /// Extracts the content of a non-interpolated string-literal `@Parameter` argument,
  /// or emits a node-level diagnostic and aborts macro expansion otherwise.
  /// Interpolated strings are rejected because only the literal segments would be
  /// captured, silently dropping the user's intent.
  static func requireStringLiteralArgument(
    _ arg: LabeledExprSyntax,
    label: String,
    propertyName: String,
    context: some MacroExpansionContext,
  ) throws -> String {
    guard let stringLiteral = arg.expression.as(StringLiteralExprSyntax.self) else {
      try diagnoseAndAbort(
        message: "@Parameter '\(label)' on '\(propertyName)' must be a string literal.",
        node: arg.expression,
        in: context,
      )
    }
    guard let content = plainLiteralStringContent(stringLiteral) else {
      try diagnoseAndAbort(
        message: "@Parameter '\(label)' on '\(propertyName)' must be a plain string literal without interpolation.",
        node: arg.expression,
        in: context,
      )
    }
    return content
  }

  /// Extracts the content of a `@Tool` static property initializer, or emits a
  /// node-level diagnostic and aborts macro expansion otherwise. Matches the
  /// `@Parameter` treatment so users get a targeted "must be a string literal"
  /// diagnostic pointing at the offending expression, rather than a misleading
  /// top-level "property is missing" error when the property actually exists
  /// but has a non-literal initializer.
  static func requireStaticStringLiteralProperty(
    initializerValue: ExprSyntax,
    propertyName: String,
    context: some MacroExpansionContext,
  ) throws -> String {
    guard let stringLiteral = initializerValue.as(StringLiteralExprSyntax.self) else {
      try diagnoseAndAbort(
        message: "@Tool '\(propertyName)' must be a string literal.",
        node: initializerValue,
        in: context,
      )
    }
    guard let content = plainLiteralStringContent(stringLiteral) else {
      try diagnoseAndAbort(
        message: "@Tool '\(propertyName)' must be a plain string literal without interpolation.",
        node: initializerValue,
        in: context,
      )
    }
    return content
  }

  /// Extracts a `Bool` value from a `@Tool` static property initializer, or emits a
  /// node-level diagnostic and aborts macro expansion otherwise. The macro reads
  /// these flags at expansion time to decide what code to emit, so any non-literal
  /// `Bool` expression (constant reference, arithmetic, etc.) used to silently fall
  /// through to `false` and disable the flag without warning.
  static func requireStaticBooleanLiteralProperty(
    initializerValue: ExprSyntax,
    propertyName: String,
    context: some MacroExpansionContext,
  ) throws -> Bool {
    guard let boolLiteral = initializerValue.as(BooleanLiteralExprSyntax.self) else {
      try diagnoseAndAbort(
        message: "@Tool '\(propertyName)' must be a boolean literal (true or false).",
        node: initializerValue,
        in: context,
      )
    }
    return boolLiteral.literal.text == "true"
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

// MARK: - Tool Name Validation

extension ToolMacro {
  /// Valid characters for tool names: A-Z, a-z, 0-9, _, -, .
  private static let validToolNameCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.",
  )

  /// Validates a tool name and returns an error message if invalid, or nil if valid.
  static func validateToolName(_ name: String) -> String? {
    if name.isEmpty {
      return "Tool name cannot be empty"
    }
    // Downstream LLM hosts impose tool-name limits (OpenAI caps at 64). 128 leaves
    // headroom while still rejecting obviously bogus names early at compile time.
    if name.count > 128 {
      return "Tool name exceeds maximum length of 128 characters (got \(name.count))"
    }

    let nameCharSet = CharacterSet(charactersIn: name)
    if !nameCharSet.isSubset(of: validToolNameCharacters) {
      // Render each invalid scalar in a form that survives terminals and IDEs:
      // printable scalars quoted (so whitespace stays visible), control / DEL
      // / non-ASCII scalars as `U+XXXX`. Joined with `, ` so two invalid chars
      // don't render as an ambiguous blob.
      let invalidChars = name.unicodeScalars.filter { !validToolNameCharacters.contains($0) }
      let renderedChars = invalidChars.map { scalar -> String in
        if scalar.value >= 0x20, scalar.value < 0x7F {
          return "'\(String(scalar))'"
        }
        return String(format: "U+%04X", scalar.value)
      }
      return "Tool name contains invalid characters: \(renderedChars.joined(separator: ", ")). Only A-Z, a-z, 0-9, _, -, . are allowed"
    }

    return nil
  }

  /// Returns a warning message if the tool name has style issues, or nil if ok.
  static func toolNameStyleWarning(_ name: String) -> String? {
    if let first = name.first, first == "-" || first == "." {
      return "Tool name '\(name)' starts with '\(first)' which may cause compatibility issues"
    }
    if let last = name.last, last == "-" || last == "." {
      return "Tool name '\(name)' ends with '\(last)' which may cause compatibility issues"
    }
    return nil
  }
}

// MARK: - Attribute Matching

extension ToolMacro {
  /// Checks if an attribute is the `@Parameter` attribute. Recognizes both the bare
  /// `@Parameter` form and the qualified `@<Module>.Parameter` form (compatibility for
  /// when both MCP and AI are imported in the same file). The qualifying module name
  /// comes from `parameterAttributeModuleName`, defined per-repo.
  static func isParameterAttribute(_ attr: AttributeListSyntax.Element) -> Bool {
    guard case let .attribute(attrSyntax) = attr else { return false }

    // Check for simple `@Parameter`
    if let identifier = attrSyntax.attributeName.as(IdentifierTypeSyntax.self) {
      return identifier.name.text == "Parameter"
    }

    // Check for qualified `@<Module>.Parameter`
    if let memberType = attrSyntax.attributeName.as(MemberTypeSyntax.self),
       let baseIdentifier = memberType.baseType.as(IdentifierTypeSyntax.self)
    {
      return baseIdentifier.name.text == parameterAttributeModuleName
        && memberType.name.text == "Parameter"
    }

    return false
  }

  /// Checks if a variable declaration has the `@Parameter` attribute.
  static func hasParameterAttribute(_ varDecl: VariableDeclSyntax) -> Bool {
    varDecl.attributes.contains { isParameterAttribute($0) }
  }
}

// MARK: - Type Helpers

extension ToolMacro {
  /// Unwraps `Optional<T>` / `Swift.Optional<T>` (the explicit, non-sugared form).
  /// Returns nil for non-Optional types. The `T?` sugar is handled separately via
  /// `OptionalTypeSyntax` at the call site.
  static func unwrapExplicitOptional(_ type: TypeSyntax) -> TypeSyntax? {
    let genericClause: GenericArgumentClauseSyntax?
    if let ident = type.as(IdentifierTypeSyntax.self), ident.name.text == "Optional" {
      genericClause = ident.genericArgumentClause
    } else if let member = type.as(MemberTypeSyntax.self), member.name.text == "Optional" {
      // e.g. Swift.Optional<Int>
      genericClause = member.genericArgumentClause
    } else {
      return nil
    }
    guard let firstArg = genericClause?.arguments.first?.argument,
          case let .type(wrapped) = firstArg
    else {
      return nil
    }
    return wrapped
  }
}

// MARK: - Default Value Validation

extension ToolMacro {
  /// Checks if an expression is a supported literal value.
  /// Returns true for: integer, float, string, boolean, nil literals, plus negative
  /// numeric literals. Returns false for function calls, member access, etc.
  ///
  /// The prefix-minus branch is restricted to numeric literals so that nonsense
  /// inputs like `-true` or `-"x"` are rejected here rather than reaching
  /// `convertToValueLiteral` (which traps on unmapped kinds) and crashing the plugin.
  static func isLiteralExpression(_ expr: ExprSyntax) -> Bool {
    if expr.is(IntegerLiteralExprSyntax.self) { return true }
    if expr.is(FloatLiteralExprSyntax.self) { return true }
    if let stringLiteral = expr.as(StringLiteralExprSyntax.self) {
      return plainLiteralStringContent(stringLiteral) != nil
    }
    if expr.is(BooleanLiteralExprSyntax.self) { return true }
    if expr.is(NilLiteralExprSyntax.self) { return true }
    if let prefixExpr = expr.as(PrefixOperatorExprSyntax.self),
       prefixExpr.operator.text == "-"
    {
      return prefixExpr.expression.is(IntegerLiteralExprSyntax.self)
        || prefixExpr.expression.is(FloatLiteralExprSyntax.self)
    }
    return false
  }
}

// MARK: - Swift String Literal Escaping

extension ToolMacro {
  /// Escapes a Swift `String` value for safe interpolation into generated Swift source.
  /// Returns `"nil"` for nil input, or a properly-quoted literal otherwise.
  static func swiftStringLiteral(_ value: String?) -> String {
    guard let value else { return "nil" }
    let escaped = value
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
      .replacingOccurrences(of: "\n", with: "\\n")
      .replacingOccurrences(of: "\r", with: "\\r")
      .replacingOccurrences(of: "\t", with: "\\t")
    return "\"\(escaped)\""
  }
}
