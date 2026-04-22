// Copyright © Anthony DePasquale

import Foundation

/// Normalizes a tool's `outputSchema` to Gemini's `responseJsonSchema` subset.
///
/// `@Schemable` produces `anyOf: [<schema>, {type: "null"}]` for optional
/// fields. Gemini's `responseJsonSchema` accepts a JSON Schema subset and
/// commonly rejects `oneOf` / `anyOf` composites. The normalizer rewrites the
/// canonical Optional shape to `<schema>` with `nullable: true` (Gemini's
/// idiom). For other unsupported features (non-nullable composites, `$ref`,
/// 2020-12-only constructs), the normalizer returns `nil` and the encoder
/// omits `responseJsonSchema` from the tool declaration with a one-time
/// per-tool warning log. The structured channel still flows; only the
/// declared schema is dropped.
///
/// `$defs` / `definitions` are not recursed into. In practice they only
/// appear alongside `$ref`, which the normalizer rejects wholesale — so a
/// schema with `$defs` already short-circuits to `nil` via its `$ref` users.
enum GeminiSchemaNormalizer {
  /// Returns a Gemini-compatible normalized schema, or `nil` if the schema
  /// uses features Gemini's `responseJsonSchema` doesn't support and the
  /// normalizer can't safely rewrite. The canonical schema in
  /// `AI.Tool.outputSchema` is unchanged — only the Gemini-bound copy.
  static func normalize(_ value: Value) -> Value? {
    switch value {
      case let .object(fields):
        normalizeObject(fields)
      case .array, .string, .int, .double, .bool, .null:
        value
    }
  }

  private static func normalizeObject(_ fields: [String: Value]) -> Value? {
    // Recognize the canonical anyOf-nullable Optional shape and rewrite to
    // `nullable: true`. Both orderings of the variants are accepted.
    if fields.count == 1, case let .array(variants)? = fields["anyOf"], variants.count == 2 {
      if let rewritten = rewriteAnyOfNullable(variants) {
        return rewritten
      }
      // Two-variant anyOf without a `null` arm — no safe rewrite.
      return nil
    }

    // Reject non-nullable composites and unsupported references entirely.
    if fields["oneOf"] != nil || fields["allOf"] != nil
      || fields["$ref"] != nil
      || fields["dependentSchemas"] != nil
      || fields["if"] != nil
      || fields["then"] != nil
      || fields["else"] != nil
    {
      return nil
    }
    if fields["anyOf"] != nil {
      return nil
    }

    var rewritten = fields

    // Recurse into properties.
    if case let .object(properties)? = fields["properties"] {
      var rewrittenProps: [String: Value] = [:]
      for (key, prop) in properties {
        guard let normalized = normalize(prop) else { return nil }
        rewrittenProps[key] = normalized
      }
      rewritten["properties"] = .object(rewrittenProps)
    }

    // Recurse into items / additionalProperties.
    if let items = fields["items"] {
      guard let normalized = normalize(items) else { return nil }
      rewritten["items"] = normalized
    }
    if let addl = fields["additionalProperties"] {
      // additionalProperties may be `bool` or a schema; bools pass through.
      if case .bool = addl {
        // ok
      } else {
        guard let normalized = normalize(addl) else { return nil }
        rewritten["additionalProperties"] = normalized
      }
    }

    return .object(rewritten)
  }

  /// If `variants` is `[<schema>, {type: "null"}]` (or symmetric), returns a
  /// recursed copy of `<schema>` with `nullable: true`. Otherwise returns nil.
  private static func rewriteAnyOfNullable(_ variants: [Value]) -> Value? {
    let nullSchema: Value = .object(["type": .string("null")])
    let nonNullVariant: Value
    if variants[0] == nullSchema {
      nonNullVariant = variants[1]
    } else if variants[1] == nullSchema {
      nonNullVariant = variants[0]
    } else {
      return nil
    }
    guard case let .object(fields) = nonNullVariant else {
      return nil
    }
    guard let normalized = normalizeObject(fields) else { return nil }
    guard case var .object(rewrittenFields) = normalized else { return nil }
    rewrittenFields["nullable"] = .bool(true)
    return .object(rewrittenFields)
  }
}
