// Copyright © Anthony DePasquale

import Foundation

/// Namespace for output-schema resolution helpers.
public enum AISchema {
  /// Returns the JSON Schema for a tool output type if one can be derived.
  ///
  /// Resolution order:
  /// 1. `StructuredOutput` → `outputJSONSchema` (unwrapped — the struct
  ///    carries its own named object shape).
  /// 2. `UnwrappedObjectOutput` (today only `Dictionary<String, V>`) →
  ///    `valueSchema` directly (already a top-level object).
  /// 3. `PrimitiveToolOutput` → the value schema wrapped in
  ///    `{"type": "object", "properties": {"result": <valueSchema>}, "required": ["result"], "additionalProperties": false}`.
  /// 4. `StructuredMetadataCarrier` (`ImageWithMetadata<T>` /
  ///    `AudioWithMetadata<T>` / `MediaWithMetadata<T>` /
  ///    `AssetWithMetadata<T>`) → `metadataSchema` via the existential.
  ///
  /// Returns `nil` for custom `ToolOutput` conformers — the escape hatch
  /// doesn't publish schemas.
  public static func outputSchema(for outputType: Any.Type) -> Value? {
    if let structured = outputType as? any StructuredOutput.Type {
      assert(
        !(outputType is any PrimitiveToolOutput.Type),
        "Type \(outputType) conforms to both StructuredOutput and PrimitiveToolOutput. Pick one: structs use StructuredOutput (unwrapped), primitives/arrays/optionals use PrimitiveToolOutput (wrapped under 'result').",
      )
      return structured.outputJSONSchema
    }
    if let unwrapped = outputType as? any UnwrappedObjectOutput.Type {
      assert(
        !(outputType is any PrimitiveToolOutput.Type),
        "Type \(outputType) conforms to both UnwrappedObjectOutput and PrimitiveToolOutput. UnwrappedObjectOutput emits as a top-level object; PrimitiveToolOutput wraps under 'result'. Pick one.",
      )
      return unwrapped.valueSchema
    }
    if let primitive = outputType as? any PrimitiveToolOutput.Type {
      return .object([
        "type": .string("object"),
        "properties": .object(["result": primitive.valueSchema]),
        "required": .array([.string("result")]),
        "additionalProperties": .bool(false),
      ])
    }
    if let carrier = outputType as? any StructuredMetadataCarrier.Type {
      return carrier.metadataSchema
    }
    return nil
  }
}

/// Marker protocol that lets `AISchema.outputSchema(for:)` recover the
/// metadata schema from a `MediaWithMetadata<T>` / `AssetWithMetadata<T>`
/// existential without knowing the concrete `Metadata` type at the call site.
///
/// Internal — only `MediaWithMetadata` / `AssetWithMetadata` (added in §4 / §5)
/// conform.
protocol StructuredMetadataCarrier {
  static var metadataSchema: Value { get }
}
