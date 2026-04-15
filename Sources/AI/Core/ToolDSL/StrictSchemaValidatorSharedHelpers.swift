// Copyright © Anthony DePasquale

// SHARED HELPERS — kept in lockstep with:
//   ../../../../forked/swift-mcp/Sources/MCP/ToolDSL/StrictSchemaValidatorSharedHelpers.swift
//
// If you change anything here, mirror it there. The two files should remain
// equivalent modulo indentation (this repo uses 2-space, swift-mcp uses 4-space)
// and the path comment above.
//
// What belongs in this file: provider-agnostic universal-strict JSON Schema
// validation. The validator throws when a schema can't be made strict-mode
// compatible without changing its semantics — for example, an optional property
// that isn't nullable and has no default, or a non-object root.
//
// What does NOT belong here:
//   - OpenAI-specific transforms (e.g. `oneOf → anyOf` rewrite). Those live in
//     each repo's `ToolSchema.swift` and run only at OpenAI request-send time,
//     never as part of validation.
//   - The error type that gets thrown. Each repo defines its own
//     `makeStrictSchemaError` extension on `StrictSchemaValidator` that returns
//     an `Error` of the repo-native type (`AIError` here, `MCPError` in swift-mcp).
//     The shared file calls that adapter, so the rest of the body stays identical.

enum StrictSchemaValidator {
  /// Validates that `schema` can be made universally-strict-mode compatible.
  ///
  /// Throws on conditions that can't be auto-fixed by a normalizer pass —
  /// non-object root and optional-but-not-nullable properties without defaults.
  /// Conditions that *can* be auto-fixed (missing `additionalProperties: false`,
  /// missing entries in `required`) are not asserted here, since a normalizer
  /// would add them on the way out without changing schema semantics.
  static func validate(
    _ schema: [String: Value],
    path: [String] = [],
    root: [String: Value]? = nil,
  ) throws {
    if path.isEmpty, schema["type"]?.stringValue != "object" {
      throw makeStrictSchemaError(
        "Strict mode requires the root schema to have type: 'object' but got type: '\(schema["type"]?.stringValue ?? "undefined")'",
      )
    }

    let resolvedRoot = root ?? schema

    // Resolve `$ref` so the rules below apply to the referenced shape, not
    // to the placeholder. Matches the normalizer's behavior so callers get
    // consistent error messages whether they validate the raw or the
    // normalized form.
    if let ref = schema["$ref"]?.stringValue, schema.count > 1 {
      let resolved = try resolveRef(ref, in: resolvedRoot)
      var merged = resolved
      for (key, value) in schema where key != "$ref" {
        merged[key] = value
      }
      try validate(merged, path: path, root: resolvedRoot)
      return
    }

    let originalRequired = requiredNames(in: schema)

    for (key, value) in schema {
      switch key {
        case "properties":
          guard case let .object(properties) = value else { continue }
          for (propertyName, propertySchema) in properties {
            // A property is safe in strict mode if any of: already required,
            // nullable (null is an accepted value), or carries a default
            // (downstream fills in the default when the model omits the field).
            // Matches the OpenAI TS SDK behavior in zod-to-json-schema.
            if !originalRequired.contains(propertyName),
               !isNullableSchema(propertySchema),
               !hasDefaultValue(propertySchema)
            {
              let fieldPath = (path + ["properties", propertyName]).joined(separator: "/")
              throw makeStrictSchemaError(
                "Strict mode requires all properties to be required. Property '\(fieldPath)' is optional but not nullable. "
                  + "Either add it to the 'required' array, make it nullable (e.g. type: [\"string\", \"null\"]), or disable strict mode.",
              )
            }
            if case let .object(propertySchemaDict) = propertySchema {
              try validate(
                propertySchemaDict,
                path: path + ["properties", propertyName],
                root: resolvedRoot,
              )
            }
          }
        case "items":
          if case let .object(itemSchema) = value {
            try validate(itemSchema, path: path + ["items"], root: resolvedRoot)
          }
        case "anyOf", "oneOf", "allOf":
          guard case let .array(variants) = value else { continue }
          for (index, variant) in variants.enumerated() {
            if case let .object(variantSchema) = variant {
              try validate(
                variantSchema,
                path: path + [key, String(index)],
                root: resolvedRoot,
              )
            }
          }
        case "$defs", "definitions":
          guard case let .object(definitions) = value else { continue }
          for (definitionName, definitionSchema) in definitions {
            if case let .object(definitionSchemaDict) = definitionSchema {
              try validate(
                definitionSchemaDict,
                path: path + [key, definitionName],
                root: resolvedRoot,
              )
            }
          }
        case "additionalProperties":
          if case let .object(additionalPropertiesSchema) = value {
            try validate(
              additionalPropertiesSchema,
              path: path + ["additionalProperties"],
              root: resolvedRoot,
            )
          }
        default:
          continue
      }
    }
  }

  // MARK: - Internal helpers

  private static func requiredNames(in schema: [String: Value]) -> Set<String> {
    if case let .array(requiredArray) = schema["required"] {
      return Set(requiredArray.compactMap(\.stringValue))
    }
    return []
  }

  private static func resolveRef(_ ref: String, in root: [String: Value]) throws -> [String: Value] {
    guard ref.hasPrefix("#/") else {
      throw makeStrictSchemaError("Unexpected $ref format \"\(ref)\": does not start with #/")
    }

    var current: Value = .object(root)
    for part in ref.dropFirst(2).split(separator: "/").map(String.init) {
      guard case let .object(dictionary) = current, let next = dictionary[part] else {
        throw makeStrictSchemaError("Key \"\(part)\" not found while resolving $ref \"\(ref)\"")
      }
      current = next
    }

    guard case let .object(resolved) = current else {
      throw makeStrictSchemaError("Expected $ref \"\(ref)\" to resolve to an object schema")
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
