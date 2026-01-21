// Copyright © Anthony DePasquale

import Foundation
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxMacros

/// The `@Tool` macro generates `ToolSpec` protocol conformance.
///
/// It inspects the struct to find:
/// - `static let name: String` - The tool name
/// - `static let description: String` - The tool description
/// - `static let title: String` (optional) - User-facing title
/// - Properties with `@Parameter` attribute - Tool parameters
/// - `func perform()` - The execution method
///
/// It generates:
/// - `static var tool: Tool` - The tool definition with JSON Schema
/// - `static func parse(from:)` - Argument parsing
/// - `init()` - Empty initializer
/// - `static var title: String` (if not provided) - Defaults to name
public struct ToolMacro: MemberMacro, ExtensionMacro {
  // MARK: - MemberMacro

  public static func expansion(
    of _: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo _: [TypeSyntax],
    in context: some MacroExpansionContext
  ) throws -> [DeclSyntax] {
    // Ensure we're applied to a struct
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      throw ToolMacroError.notAStruct
    }

    let structName = structDecl.name.text

    // Extract tool metadata
    let toolInfo = try extractToolInfo(from: structDecl, context: context)

    // Generate members
    var members: [DeclSyntax] = []

    // Generate init()
    members.append("""
    public init() {}
    """)

    // Generate title if not provided by user
    if !toolInfo.hasTitle {
      members.append("""
      public static var title: String { name }
      """)
    }

    // Generate tool property
    let toolDecl = generateToolProperty(
      structName: structName,
      toolInfo: toolInfo
    )
    members.append(toolDecl)

    // Generate parse(from:)
    let parseDecl = generateParseMethod(toolInfo: toolInfo)
    members.append(parseDecl)

    return members
  }

  // MARK: - ExtensionMacro

  public static func expansion(
    of _: AttributeSyntax,
    attachedTo declaration: some DeclGroupSyntax,
    providingExtensionsOf type: some TypeSyntaxProtocol,
    conformingTo _: [TypeSyntax],
    in _: some MacroExpansionContext
  ) throws -> [ExtensionDeclSyntax] {
    // Validate before adding conformance
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      // Let member macro report the error
      return []
    }

    // Check for required static properties
    var hasName = false
    var hasDescription = false
    var toolName: String?

    for member in structDecl.memberBlock.members {
      if let varDecl = member.decl.as(VariableDeclSyntax.self),
         varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
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
          }
        }
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
  }

  private static func extractToolInfo(
    from structDecl: StructDeclSyntax,
    context: some MacroExpansionContext
  ) throws -> ToolInfo {
    var name: String?
    var nameSyntax: SyntaxProtocol?
    var description: String?
    var parameters: [ParameterInfo] = []
    var outputType = "String"
    var hasTitle = false
    var strictSchema = false
    var hasPerformMethod = false

    for member in structDecl.memberBlock.members {
      let decl = member.decl

      // Look for static let name/description/title
      if let varDecl = decl.as(VariableDeclSyntax.self),
         varDecl.modifiers.contains(where: { $0.name.text == "static" })
      {
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
        hasPerformMethod = true
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

    guard hasPerformMethod else {
      throw ToolMacroError.missingPerformMethod
    }

    // Validate tool name
    if let validationError = validateToolName(toolName) {
      throw ToolMacroError.invalidToolName(validationError)
    }

    // Warn about tool name style issues
    if let styleWarning = toolNameStyleWarning(toolName),
       let syntax = nameSyntax
    {
      context.diagnose(Diagnostic(
        node: Syntax(syntax),
        message: ToolMacroDiagnostic.warning(styleWarning)
      ))
    }

    return ToolInfo(
      name: toolName,
      description: toolDescription,
      parameters: parameters,
      outputType: outputType,
      hasTitle: hasTitle,
      strictSchema: strictSchema
    )
  }

  private static func extractParameterInfo(
    from varDecl: VariableDeclSyntax,
    binding: PatternBindingSyntax,
    context _: some MacroExpansionContext
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
      maximum: maximum
    )
  }

  // MARK: - Code Generation

  private static func generateToolProperty(
    structName _: String,
    toolInfo: ToolInfo
  ) -> DeclSyntax {
    // Generate properties for each parameter
    var propertiesEntries: [String] = []

    for param in toolInfo.parameters {
      var propEntries: [String] = []

      // Get base type for schema
      let baseType = param.typeName
      let schemaType = getJSONSchemaType(for: baseType)

      propEntries.append("\"type\": AI.Value.string(\"\(schemaType)\")")

      if let title = param.title {
        propEntries.append("\"title\": AI.Value.string(\"\(title)\")")
      }

      if let desc = param.description {
        propEntries.append("\"description\": AI.Value.string(\"\(desc)\")")
      }

      // Add type-specific properties
      let additionalProps = getJSONSchemaProperties(for: baseType)
      for (key, value) in additionalProps {
        propEntries.append("\"\(key)\": \(value)")
      }

      // Add constraints
      if let minLen = param.minLength {
        propEntries.append("\"minLength\": AI.Value.int(\(minLen))")
      }
      if let maxLen = param.maxLength {
        propEntries.append("\"maxLength\": AI.Value.int(\(maxLen))")
      }
      if let min = param.minimum {
        propEntries.append("\"minimum\": AI.Value.double(\(min))")
      }
      if let max = param.maximum {
        propEntries.append("\"maximum\": AI.Value.double(\(max))")
      }

      // Add default value if present
      if let defaultVal = param.defaultValue {
        let defaultExpr = convertToValueLiteral(defaultVal, type: baseType)
        propEntries.append("\"default\": \(defaultExpr)")
      }

      // Merge in runtime jsonSchemaProperties for enums and other custom types
      let propObject = "AI.Value.object([\(propEntries.joined(separator: ", "))].merging(\(baseType).jsonSchemaProperties) { _, new in new })"
      propertiesEntries.append("\"\(param.jsonKey)\": \(propObject)")
    }

    let propertiesStr = propertiesEntries.joined(separator: ", ")

    // Required fields: non-optional without defaults
    let requiredFields = toolInfo.parameters
      .filter { !$0.isOptional && !$0.hasDefault }
      .map { "AI.Value.string(\"\($0.jsonKey)\")" }
    let requiredStr = requiredFields.joined(separator: ", ")

    // Handle empty properties - use [:] for empty dictionary literal
    let propertiesLiteral = propertiesStr.isEmpty ? "[:]" : "[\n            \(propertiesStr)\n        ]"

    // Build schema entries (fully qualified Value types for MCP compatibility)
    var schemaEntries = [
      "\"type\": AI.Value.string(\"object\")",
      "\"properties\": AI.Value.object(\(propertiesLiteral))",
      "\"required\": AI.Value.array([\(requiredStr)])",
    ]

    // Add additionalProperties: false if strictSchema is enabled
    if toolInfo.strictSchema {
      schemaEntries.append("\"additionalProperties\": AI.Value.bool(false)")
    }

    let schemaLiteral = schemaEntries.joined(separator: ",\n                    ")

    return """
    public static var tool: AI.Tool {
        AI.Tool(
            name: name,
            description: description,
            title: title,
            inputSchema: [
                \(raw: schemaLiteral)
            ],
            resultTypes: \(raw: toolInfo.outputType).resultTypes,
            execute: { parameters in
                let instance = try Self.parse(from: parameters)
                let output = try await instance.perform()
                return output.toToolResult()
            }
        )
    }
    """
  }

  private static func generateParseMethod(toolInfo: ToolInfo) -> DeclSyntax {
    // For tools with no parameters, generate a simple parse method
    if toolInfo.parameters.isEmpty {
      return """
      public static func parse(from arguments: [String: AI.Value]) throws -> Self {
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
        // Optional: use flatMap with explicit type
        parseStatements.append(
          "_instance.\(prop) = _args[\"\(key)\"].flatMap(\(type).init(parameterValue:))"
        )
      } else if param.hasDefault {
        // Has default: only use default if key is absent; throw if wrong type
        parseStatements.append(
          "if let _value = _args[\"\(key)\"] { guard let _parsed = \(type)(parameterValue: _value) else { throw ToolError.invalidParameterType(parameter: \"\(key)\", expected: \"\(type)\", got: String(describing: _value)) }; _instance.\(prop) = _parsed }"
        )
      } else {
        // Required: guard and throw
        parseStatements.append(
          "guard let _\(prop)Value = _args[\"\(key)\"], let _\(prop) = \(type)(parameterValue: _\(prop)Value) else { throw ToolError.invalidParameterType(parameter: \"\(key)\", expected: \"\(type)\", got: _args[\"\(key)\"].map { String(describing: $0) } ?? \"nil\") }"
        )
        parseStatements.append("_instance.\(prop) = _\(prop)")
      }
    }

    let statements = parseStatements.joined(separator: "\n    ")

    return """
    public static func parse(from arguments: [String: AI.Value]) throws -> Self {
        var _instance = Self()
        let _args = arguments
        \(raw: statements)
        return _instance
    }
    """
  }

  // MARK: - Type Mapping Helpers

  /// Checks if a type string represents a dictionary type `[Key: Value]` at the top level.
  private static func isDictionaryType(_ inner: String) -> Bool {
    var bracketDepth = 0
    for char in inner {
      switch char {
        case "[": bracketDepth += 1
        case "]": bracketDepth -= 1
        case ":":
          if bracketDepth == 0 {
            return true
          }
        default: break
      }
    }
    return false
  }

  /// Extracts the value type from a dictionary type string `[Key: Value]`.
  private static func extractDictionaryValueType(_ inner: String) -> String? {
    var bracketDepth = 0
    for (index, char) in inner.enumerated() {
      switch char {
        case "[": bracketDepth += 1
        case "]": bracketDepth -= 1
        case ":":
          if bracketDepth == 0 {
            let afterColon = inner.index(inner.startIndex, offsetBy: index + 1)
            return String(inner[afterColon...]).trimmingCharacters(in: .whitespaces)
          }
        default: break
      }
    }
    return nil
  }

  private static func getJSONSchemaType(for swiftType: String) -> String {
    switch swiftType {
      case "String": return "string"
      case "Int": return "integer"
      case "Double": return "number"
      case "Bool": return "boolean"
      case "Date": return "string"
      case "Data": return "string"
      default:
        // Check for collection types [T] or [K: V]
        if swiftType.hasPrefix("["), swiftType.hasSuffix("]") {
          let inner = String(swiftType.dropFirst().dropLast())
          if isDictionaryType(inner) {
            return "object"
          }
          return "array"
        }
        // Could be an enum or other type - default to string
        return "string"
    }
  }

  private static func getJSONSchemaProperties(for swiftType: String) -> [(String, String)] {
    switch swiftType {
      case "Date":
        return [("format", "AI.Value.string(\"date-time\")")]
      case "Data":
        return [("contentEncoding", "AI.Value.string(\"base64\")")]
      default:
        if swiftType.hasPrefix("["), swiftType.hasSuffix("]") {
          let inner = String(swiftType.dropFirst().dropLast())
          if let valueType = extractDictionaryValueType(inner) {
            let valueSchema = generateSchemaObject(for: valueType)
            return [("additionalProperties", valueSchema)]
          }
          let elementSchema = generateSchemaObject(for: inner)
          return [("items", elementSchema)]
        }
        return []
    }
  }

  private static func generateSchemaObject(for swiftType: String) -> String {
    let schemaType = getJSONSchemaType(for: swiftType)
    var parts = ["\"type\": AI.Value.string(\"\(schemaType)\")"]

    switch swiftType {
      case "Date":
        parts.append("\"format\": AI.Value.string(\"date-time\")")
      case "Data":
        parts.append("\"contentEncoding\": AI.Value.string(\"base64\")")
      default:
        if swiftType.hasPrefix("["), swiftType.hasSuffix("]") {
          let inner = String(swiftType.dropFirst().dropLast())
          if let valueType = extractDictionaryValueType(inner) {
            let valueSchema = generateSchemaObject(for: valueType)
            parts.append("\"additionalProperties\": \(valueSchema)")
          } else {
            let elementSchema = generateSchemaObject(for: inner)
            parts.append("\"items\": \(elementSchema)")
          }
        }
    }

    return "AI.Value.object([\(parts.joined(separator: ", "))])"
  }

  private static func convertToValueLiteral(_ value: String, type: String) -> String {
    switch type {
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

// MARK: - Tool Name Validation

extension ToolMacro {
  /// Valid characters for tool names: A-Z, a-z, 0-9, _, -, .
  private static let validToolNameCharacters = CharacterSet(
    charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-."
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
        "@Tool requires a 'func perform() async throws' method"
      case let .invalidToolName(reason):
        "Invalid tool name: \(reason)"
      case let .nonLiteralDefaultValue(param):
        "Parameter '\(param)' has a non-literal default value. Only literal values (numbers, strings, booleans) are supported. For complex defaults, make the parameter optional and handle the default in perform()."
    }
  }
}

// MARK: - Diagnostics

struct ToolMacroDiagnostic: DiagnosticMessage {
  let message: String
  let diagnosticID: MessageID
  let severity: DiagnosticSeverity

  static func warning(_ message: String) -> ToolMacroDiagnostic {
    ToolMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "ToolMacro", id: "warning"),
      severity: .warning
    )
  }

  static func error(_ message: String) -> ToolMacroDiagnostic {
    ToolMacroDiagnostic(
      message: message,
      diagnosticID: MessageID(domain: "ToolMacro", id: "error"),
      severity: .error
    )
  }
}
