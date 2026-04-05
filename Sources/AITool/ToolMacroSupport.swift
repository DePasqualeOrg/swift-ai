// Copyright © Anthony DePasquale

@_spi(ToolMacroSupport) import AI

/// Public runtime support used by `@Tool` macro expansions.
///
/// This intentionally lives in `AITool`, the module users already import to use
/// the macro, instead of leaking generated-code plumbing through `AI`.
public enum ToolMacroSupport {
  /// Runtime schema metadata for a single generated tool parameter.
  public struct SchemaParameterDescriptor: Sendable {
    public let name: String
    public let title: String?
    public let description: String?
    public let jsonSchemaType: String
    public let jsonSchemaProperties: [String: Value]
    public let isOptional: Bool
    public let hasDefault: Bool
    public let defaultValue: Value?
    public let minLength: Int?
    public let maxLength: Int?
    public let minimum: Double?
    public let maximum: Double?

    public init(
      name: String,
      title: String? = nil,
      description: String? = nil,
      jsonSchemaType: String,
      jsonSchemaProperties: [String: Value] = [:],
      isOptional: Bool,
      hasDefault: Bool = false,
      defaultValue: Value? = nil,
      minLength: Int? = nil,
      maxLength: Int? = nil,
      minimum: Double? = nil,
      maximum: Double? = nil,
    ) {
      self.name = name
      self.title = title
      self.description = description
      self.jsonSchemaType = jsonSchemaType
      self.jsonSchemaProperties = jsonSchemaProperties
      self.isOptional = isOptional
      self.hasDefault = hasDefault
      self.defaultValue = defaultValue
      self.minLength = minLength
      self.maxLength = maxLength
      self.minimum = minimum
      self.maximum = maximum
    }
  }

  /// Builds a runtime schema descriptor for a macro-generated parameter.
  public static func makeSchemaParameterDescriptor<T: ParameterValue>(
    name: String,
    title: String? = nil,
    description: String? = nil,
    type _: T.Type,
    isOptional: Bool,
    hasDefault: Bool = false,
    defaultValue: Value? = nil,
    minLength: Int? = nil,
    maxLength: Int? = nil,
    minimum: Double? = nil,
    maximum: Double? = nil,
  ) -> SchemaParameterDescriptor {
    SchemaParameterDescriptor(
      name: name,
      title: title,
      description: description,
      jsonSchemaType: T.jsonSchemaType,
      jsonSchemaProperties: T.jsonSchemaProperties,
      isOptional: isOptional,
      hasDefault: hasDefault,
      defaultValue: defaultValue,
      minLength: minLength,
      maxLength: maxLength,
      minimum: minimum,
      maximum: maximum,
    )
  }

  /// Nonthrowing schema build result used by macro-generated declarations.
  public struct BuiltObjectSchema: Sendable {
    public let schema: [String: Value]
    public let errorMessage: String?

    public init(schema: [String: Value], errorMessage: String?) {
      self.schema = schema
      self.errorMessage = errorMessage
    }
  }

  /// Builds an object JSON Schema using the shared `AI` strict-mode engine.
  public static func buildObjectSchema(
    parameters: [SchemaParameterDescriptor],
    strict: Bool,
  ) throws -> [String: Value] {
    try AI.ToolSchemaRuntime.buildObjectSchema(
      parameters: parameters,
      strict: strict,
      name: \.name,
      title: \.title,
      description: \.description,
      jsonSchemaType: \.jsonSchemaType,
      jsonSchemaProperties: \.jsonSchemaProperties,
      isOptional: \.isOptional,
      hasDefault: \.hasDefault,
      defaultValue: \.defaultValue,
      minLength: \.minLength,
      maxLength: \.maxLength,
      minimum: \.minimum,
      maximum: \.maximum,
    )
  }

  /// Builds a schema without trapping, preserving any strict-mode failure as metadata.
  public static func buildObjectSchemaResult(
    parameters: [SchemaParameterDescriptor],
    strict: Bool,
  ) -> BuiltObjectSchema {
    do {
      return try BuiltObjectSchema(
        schema: buildObjectSchema(parameters: parameters, strict: strict),
        errorMessage: nil,
      )
    } catch {
      let fallbackSchema = (try? buildObjectSchema(parameters: parameters, strict: false)) ?? [
        "type": .string("object"),
        "properties": .object([:]),
        "required": .array([]),
      ]
      return BuiltObjectSchema(
        schema: fallbackSchema,
        errorMessage: error.localizedDescription,
      )
    }
  }
}
