// Copyright © Anthony DePasquale

@_spi(ToolMacroSupport) import AI
import JSONSchemaBuilder

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

  /// Builds a runtime schema descriptor from a `JSONSchemaComponent` produced
  /// by `@Schemable` (or the built-in conformances for primitives).
  ///
  /// This is the path used by the `@Tool` macro. The component's schema is
  /// converted to the internal `[String: Value]` representation and split into
  /// the `jsonSchemaType` / `jsonSchemaProperties` fields so the shared
  /// strict-mode normalizer continues to work unchanged.
  ///
  /// This call is non-throwing — if Schemable conversion fails (which indicates
  /// a programmer error), the descriptor is returned with an empty schema and
  /// the failure surfaces as a broken tool schema at the strict-mode check.
  public static func makeSchemaParameterDescriptor(
    name: String,
    title: String? = nil,
    description: String? = nil,
    schema component: some JSONSchemaComponent,
    isOptional: Bool,
    hasDefault: Bool = false,
    defaultValue: Value? = nil,
    minLength: Int? = nil,
    maxLength: Int? = nil,
    minimum: Double? = nil,
    maximum: Double? = nil,
  ) -> SchemaParameterDescriptor {
    // A `Schemable` conformance whose schema can't be converted to `Value` is
    // a programmer error in user code (custom `schema` returning
    // non-Value-encodable structures). Trap with a parameter-named message at
    // registration time rather than silently degrading to an empty schema and
    // shipping a malformed parameter to clients.
    var properties: [String: Value]
    do {
      properties = try SchemableAdapter.valueDictionary(from: component)
    } catch {
      preconditionFailure(
        "@Tool parameter '\(name)': failed to convert Schemable schema to AI.Value (\(error)). The parameter type's `Schemable` conformance must produce a JSON Schema that round-trips through `AI.Value`.",
      )
    }
    // Split `type` back out so the existing strict-mode nullable-wrapping logic
    // in ToolSchema can operate on a simple scalar type string. For composite
    // schemas (`oneOf`, `anyOf`) that have no top-level `type`, an empty
    // string is passed — nullable wrapping of composites is not supported and
    // will surface as a strict-mode error if needed.
    let jsonSchemaType: String = if case let .string(typeName) = properties.removeValue(forKey: "type") {
      typeName
    } else {
      ""
    }
    let jsonSchemaProperties = properties
    return SchemaParameterDescriptor(
      name: name,
      title: title,
      description: description,
      jsonSchemaType: jsonSchemaType,
      jsonSchemaProperties: jsonSchemaProperties,
      isOptional: isOptional,
      hasDefault: hasDefault,
      defaultValue: defaultValue,
      minLength: minLength,
      maxLength: maxLength,
      minimum: minimum,
      maximum: maximum,
    )
  }

  /// Parses a `Value` into a Swift value using the given schema component.
  /// Thin wrapper over `SchemableAdapter.parse` so macro-emitted parse sites
  /// only reference symbols in `AITool`.
  public static func parseParameter<Component: JSONSchemaComponent>(
    _ component: Component,
    from value: Value,
    parameterName: String,
  ) throws -> Component.Output {
    try SchemableAdapter.parse(component, from: value, parameterName: parameterName)
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

  /// Builds a raw, provider-agnostic object JSON Schema from parameter descriptors.
  ///
  /// Provider-specific normalization (e.g. OpenAI strict mode) happens at send
  /// time inside the provider's client, not here.
  public static func buildObjectSchema(
    parameters: [SchemaParameterDescriptor],
  ) throws -> [String: Value] {
    try AI.ToolSchemaRuntime.buildObjectSchema(
      parameters: parameters,
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

  /// Builds a raw object JSON Schema without throwing. Raw-build failures
  /// (e.g. duplicate parameter names) are reported as `errorMessage` with an
  /// empty fallback schema so macro-generated code can surface the failure
  /// at request-encoding time.
  public static func buildObjectSchemaResult(
    parameters: [SchemaParameterDescriptor],
  ) -> BuiltObjectSchema {
    do {
      let schema = try AI.ToolSchemaRuntime.buildObjectSchema(
        parameters: parameters,
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
      return BuiltObjectSchema(schema: schema, errorMessage: nil)
    } catch {
      return BuiltObjectSchema(
        schema: [
          "type": .string("object"),
          "properties": .object([:]),
          "required": .array([]),
        ],
        errorMessage: error.localizedDescription,
      )
    }
  }

  /// Asserts that `schema` is strict JSON Schema-compatible.
  ///
  /// Intended for use from macro-generated `@Tool` code when the tool author
  /// writes `static let strictSchema = true`. Failures are invariant violations
  /// (a broken programmer assertion); the macro wraps this call in `try!` so
  /// the result is a clear runtime trap with the tool name in the message.
  public static func validateStrictCompatibility(
    _ schema: [String: Value],
    toolName: String,
  ) throws {
    do {
      try AI.ToolSchemaRuntime.validateStrictCompatibility(schema)
    } catch {
      throw StrictSchemaAssertionFailure(toolName: toolName, underlying: error)
    }
  }

  /// Error thrown when a tool declares `strictSchema: true` but its schema is
  /// not strict JSON Schema-compatible. Carries the tool name so the trap
  /// message produced by `try!` points at the offending tool.
  public struct StrictSchemaAssertionFailure: Error, CustomStringConvertible {
    public let toolName: String
    public let underlying: Error

    public var description: String {
      "@Tool '\(toolName)' declares strictSchema: true but its schema is not strict JSON Schema-compatible: \(underlying.localizedDescription)"
    }
  }
}
