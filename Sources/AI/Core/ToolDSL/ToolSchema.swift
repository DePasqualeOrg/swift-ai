// Copyright © Anthony DePasquale

private struct ToolSchemaParameter {
  let name: String
  let title: String?
  let description: String?
  let jsonSchemaType: String
  let jsonSchemaProperties: [String: Value]
  let isOptional: Bool
  let hasDefault: Bool
  let defaultValue: Value?
  let minLength: Int?
  let maxLength: Int?
  let minimum: Double?
  let maximum: Double?
}

enum ToolSchema {
  static func buildObjectSchema<Parameters: Collection>(
    parameters: Parameters,
    strict: Bool,
    name: (Parameters.Element) -> String,
    title: (Parameters.Element) -> String?,
    description: (Parameters.Element) -> String?,
    jsonSchemaType: (Parameters.Element) -> String,
    jsonSchemaProperties: (Parameters.Element) -> [String: Value],
    isOptional: (Parameters.Element) -> Bool,
    hasDefault: (Parameters.Element) -> Bool,
    defaultValue: (Parameters.Element) -> Value?,
    minLength: (Parameters.Element) -> Int?,
    maxLength: (Parameters.Element) -> Int?,
    minimum: (Parameters.Element) -> Double?,
    maximum: (Parameters.Element) -> Double?,
  ) throws -> [String: Value] {
    let descriptors = parameters.map {
      ToolSchemaParameter(
        name: name($0),
        title: title($0),
        description: description($0),
        jsonSchemaType: jsonSchemaType($0),
        jsonSchemaProperties: jsonSchemaProperties($0),
        isOptional: isOptional($0),
        hasDefault: hasDefault($0),
        defaultValue: defaultValue($0),
        minLength: minLength($0),
        maxLength: maxLength($0),
        minimum: minimum($0),
        maximum: maximum($0),
      )
    }
    let duplicateNames = duplicateParameterNames(in: descriptors)
    if !duplicateNames.isEmpty {
      throw AIError.invalidRequest(message: duplicateParameterNameErrorMessage(for: duplicateNames))
    }

    var properties: [String: Value] = [:]
    for descriptor in descriptors {
      properties[descriptor.name] = .object(propertySchema(for: descriptor, strict: strict))
    }

    var schema: [String: Value] = [
      "type": .string("object"),
      "properties": .object(properties),
      "required": .array(descriptors
        .filter { !$0.isOptional && !$0.hasDefault }
        .map(\.name)
        .map(Value.string)),
    ]

    guard strict else { return schema }

    schema = try normalizeForStrictMode(schema)
    if descriptors.isEmpty {
      schema["required"] = .array([])
    }
    return schema
  }

  static func strictModeJSONObject(_ schema: [String: Value]) throws -> [String: any Sendable] {
    try Value.toSendable(normalizeForStrictMode(schema))
  }

  static func normalizeForStrictMode(_ schema: [String: Value]) throws -> [String: Value] {
    try StrictModeNormalizer.normalize(schema)
  }

  private static func propertySchema(
    for descriptor: ToolSchemaParameter,
    strict: Bool,
  ) -> [String: Value] {
    var property = descriptor.jsonSchemaProperties

    property["type"] = nullableSchemaType(for: descriptor, strict: strict)

    if let title = descriptor.title, title != descriptor.name {
      property["title"] = .string(title)
    }

    if let description = descriptor.description {
      property["description"] = .string(description)
    }

    if let minLength = descriptor.minLength {
      property["minLength"] = .int(minLength)
    }

    if let maxLength = descriptor.maxLength {
      property["maxLength"] = .int(maxLength)
    }

    if let minimum = descriptor.minimum {
      property["minimum"] = .double(minimum)
    }

    if let maximum = descriptor.maximum {
      property["maximum"] = .double(maximum)
    }

    if let defaultValue = descriptor.defaultValue {
      property["default"] = defaultValue
    }

    if isNullable(descriptor, strict: strict),
       case let .array(enumValues)? = property["enum"],
       !enumValues.contains(.null)
    {
      property["enum"] = .array(enumValues + [.null])
    }

    return property
  }

  private static func nullableSchemaType(
    for descriptor: ToolSchemaParameter,
    strict: Bool,
  ) -> Value {
    if isNullable(descriptor, strict: strict) {
      return .array([.string(descriptor.jsonSchemaType), .string("null")])
    }
    return .string(descriptor.jsonSchemaType)
  }

  private static func isNullable(
    _ descriptor: ToolSchemaParameter,
    strict: Bool,
  ) -> Bool {
    descriptor.isOptional || (strict && descriptor.hasDefault)
  }

  private static func duplicateParameterNames(in descriptors: [ToolSchemaParameter]) -> [String] {
    var counts: [String: Int] = [:]
    for descriptor in descriptors {
      counts[descriptor.name, default: 0] += 1
    }
    return counts
      .filter { $0.value > 1 }
      .map(\.key)
      .sorted()
  }

  private static func duplicateParameterNameErrorMessage(for names: [String]) -> String {
    let quotedNames = names.map { "'\($0)'" }.joined(separator: ", ")
    let suffix = names.count == 1 ? "" : "s"
    return "Tool parameters must have unique names. Duplicate parameter name\(suffix): \(quotedNames)."
  }
}

@_spi(ToolMacroSupport) public enum ToolSchemaRuntime {
  public static func buildObjectSchema<Parameters: Collection>(
    parameters: Parameters,
    strict: Bool,
    name: (Parameters.Element) -> String,
    title: (Parameters.Element) -> String?,
    description: (Parameters.Element) -> String?,
    jsonSchemaType: (Parameters.Element) -> String,
    jsonSchemaProperties: (Parameters.Element) -> [String: Value],
    isOptional: (Parameters.Element) -> Bool,
    hasDefault: (Parameters.Element) -> Bool,
    defaultValue: (Parameters.Element) -> Value?,
    minLength: (Parameters.Element) -> Int?,
    maxLength: (Parameters.Element) -> Int?,
    minimum: (Parameters.Element) -> Double?,
    maximum: (Parameters.Element) -> Double?,
  ) throws -> [String: Value] {
    try ToolSchema.buildObjectSchema(
      parameters: parameters,
      strict: strict,
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
}

private enum StrictModeNormalizer {
  static func normalize(
    _ schema: [String: Value],
    path: [String] = [],
    root: [String: Value]? = nil,
  ) throws -> [String: Value] {
    if path.isEmpty, schema["type"]?.stringValue != "object" {
      throw AIError.invalidRequest(
        message: "Strict mode requires the root schema to have type: 'object' but got type: '\(schema["type"]?.stringValue ?? "undefined")'",
      )
    }

    let resolvedRoot = root ?? schema

    if let ref = schema["$ref"]?.stringValue, schema.count > 1 {
      let resolved = try resolveRef(ref, in: resolvedRoot)
      var merged = resolved
      for (key, value) in schema where key != "$ref" {
        merged[key] = value
      }
      return try normalize(merged, path: path, root: resolvedRoot)
    }

    var result: [String: Value] = [:]
    var propertyNames: [String] = []
    let originalRequired = requiredNames(in: schema)

    for (key, value) in schema {
      switch key {
        case "properties":
          if case let .object(properties) = value {
            var converted: [String: Value] = [:]
            for (propertyName, propertySchema) in properties {
              propertyNames.append(propertyName)
              // A property is safe in strict mode if any of: already required,
              // nullable (null is an accepted value), or carries a default
              // (OpenAI fills in the default when the model omits the field).
              // Matches the OpenAI TS SDK behavior in zod-to-json-schema/parsers/object.ts.
              if !originalRequired.contains(propertyName),
                 !isNullableSchema(propertySchema),
                 !hasDefaultValue(propertySchema)
              {
                let fieldPath = (path + ["properties", propertyName]).joined(separator: "/")
                throw AIError.invalidRequest(
                  message: "Strict mode requires all properties to be required. Property '\(fieldPath)' is optional but not nullable. "
                    + "Either add it to the 'required' array, make it nullable (e.g. type: [\"string\", \"null\"]), or disable enableStrictModeForTools.",
                )
              }
              converted[propertyName] = if case let .object(propertySchemaDict) = propertySchema {
                try .object(normalize(
                  propertySchemaDict,
                  path: path + ["properties", propertyName],
                  root: resolvedRoot,
                ))
              } else {
                propertySchema
              }
            }
            result[key] = .object(converted)
          } else {
            result[key] = value
          }
        case "items":
          if case let .object(itemSchema) = value {
            result[key] = try .object(normalize(
              itemSchema,
              path: path + ["items"],
              root: resolvedRoot,
            ))
          } else {
            result[key] = value
          }
        case "anyOf", "allOf", "oneOf":
          if case let .array(variants) = value {
            result[key] = try .array(variants.enumerated().map { index, variant in
              if case let .object(variantSchema) = variant {
                return try .object(normalize(
                  variantSchema,
                  path: path + [key, String(index)],
                  root: resolvedRoot,
                ))
              }
              return variant
            })
          } else {
            result[key] = value
          }
        case "$defs", "definitions":
          if case let .object(definitions) = value {
            var converted: [String: Value] = [:]
            for (definitionName, definitionSchema) in definitions {
              converted[definitionName] = if case let .object(definitionSchemaDict) = definitionSchema {
                try .object(normalize(
                  definitionSchemaDict,
                  path: path + [key, definitionName],
                  root: resolvedRoot,
                ))
              } else {
                definitionSchema
              }
            }
            result[key] = .object(converted)
          } else {
            result[key] = value
          }
        case "additionalProperties":
          if case let .object(additionalPropertiesSchema) = value {
            result[key] = try .object(normalize(
              additionalPropertiesSchema,
              path: path + ["additionalProperties"],
              root: resolvedRoot,
            ))
          } else {
            result[key] = value
          }
        case "required":
          continue
        default:
          result[key] = value
      }
    }

    if isObjectType(schema["type"]), result["additionalProperties"] == nil {
      result["additionalProperties"] = .bool(false)
    }

    if !propertyNames.isEmpty {
      result["required"] = .array(propertyNames.sorted().map(Value.string))
    }

    return result
  }

  private static func requiredNames(in schema: [String: Value]) -> Set<String> {
    if case let .array(requiredArray) = schema["required"] {
      return Set(requiredArray.compactMap(\.stringValue))
    }
    return []
  }

  private static func isObjectType(_ type: Value?) -> Bool {
    switch type {
      case let .string(typeName):
        typeName == "object"
      case let .array(typeNames):
        typeNames.contains(.string("object"))
      default:
        false
    }
  }

  private static func resolveRef(_ ref: String, in root: [String: Value]) throws -> [String: Value] {
    guard ref.hasPrefix("#/") else {
      throw AIError.invalidRequest(message: "Unexpected $ref format \"\(ref)\": does not start with #/")
    }

    var current: Value = .object(root)
    for part in ref.dropFirst(2).split(separator: "/").map(String.init) {
      guard case let .object(dictionary) = current, let next = dictionary[part] else {
        throw AIError.invalidRequest(message: "Key \"\(part)\" not found while resolving $ref \"\(ref)\"")
      }
      current = next
    }

    guard case let .object(resolved) = current else {
      throw AIError.invalidRequest(message: "Expected $ref \"\(ref)\" to resolve to an object schema")
    }
    return resolved
  }

  private static func isNullableSchema(_ schema: Value) -> Bool {
    guard case let .object(dictionary) = schema else { return false }

    if let type = dictionary["type"] {
      if case .string("null") = type { return true }
      if case let .array(types) = type, types.contains(.string("null")) { return true }
    }

    for key in ["oneOf", "anyOf"] {
      if case let .array(variants)? = dictionary[key], variants.contains(where: isNullableSchema) {
        return true
      }
    }

    return false
  }

  private static func hasDefaultValue(_ schema: Value) -> Bool {
    guard case let .object(dictionary) = schema else { return false }
    return dictionary["default"] != nil
  }
}
