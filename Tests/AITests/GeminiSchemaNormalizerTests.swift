// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import Testing

struct GeminiSchemaNormalizerTests {
  @Test
  func `anyOf-nullable rewrites to nullable=true`() {
    let input: Value = .object([
      "anyOf": .array([
        .object(["type": .string("string")]),
        .object(["type": .string("null")]),
      ]),
    ])
    let normalized = GeminiSchemaNormalizer.normalize(input)
    #expect(normalized == .object([
      "type": .string("string"),
      "nullable": .bool(true),
    ]))
  }

  @Test
  func `symmetric anyOf-nullable also rewrites`() {
    let input: Value = .object([
      "anyOf": .array([
        .object(["type": .string("null")]),
        .object(["type": .string("string")]),
      ]),
    ])
    let normalized = GeminiSchemaNormalizer.normalize(input)
    #expect(normalized == .object([
      "type": .string("string"),
      "nullable": .bool(true),
    ]))
  }

  @Test
  func `nested anyOf-nullable inside properties rewrites in place`() {
    let input: Value = .object([
      "type": .string("object"),
      "properties": .object([
        "id": .object(["type": .string("string")]),
        "displayName": .object([
          "anyOf": .array([
            .object(["type": .string("string")]),
            .object(["type": .string("null")]),
          ]),
        ]),
      ]),
      "required": .array([.string("id"), .string("displayName")]),
    ])
    let normalized = GeminiSchemaNormalizer.normalize(input)
    if case let .object(fields) = normalized,
       case let .object(properties) = fields["properties"]
    {
      #expect(properties["displayName"] == .object([
        "type": .string("string"),
        "nullable": .bool(true),
      ]))
    } else {
      Issue.record("Expected normalized object with properties")
    }
  }

  @Test
  func `array items containing anyOf-nullable rewrites`() {
    let input: Value = .object([
      "type": .string("array"),
      "items": .object([
        "anyOf": .array([
          .object(["type": .string("integer")]),
          .object(["type": .string("null")]),
        ]),
      ]),
    ])
    let normalized = GeminiSchemaNormalizer.normalize(input)
    if case let .object(fields) = normalized,
       let items = fields["items"]
    {
      #expect(items == .object([
        "type": .string("integer"),
        "nullable": .bool(true),
      ]))
    } else {
      Issue.record("Expected normalized object with items")
    }
  }

  @Test
  func `non-nullable oneOf returns nil`() {
    let input: Value = .object([
      "oneOf": .array([
        .object(["type": .string("string")]),
        .object(["type": .string("integer")]),
      ]),
    ])
    #expect(GeminiSchemaNormalizer.normalize(input) == nil)
  }

  @Test
  func `non-nullable anyOf returns nil`() {
    // Two non-null variants → no safe rewrite.
    let input: Value = .object([
      "anyOf": .array([
        .object(["type": .string("string")]),
        .object(["type": .string("integer")]),
      ]),
    ])
    #expect(GeminiSchemaNormalizer.normalize(input) == nil)
  }

  @Test
  func `dollar-ref returns nil`() {
    let input: Value = .object([
      "$ref": .string("#/$defs/Foo"),
    ])
    #expect(GeminiSchemaNormalizer.normalize(input) == nil)
  }

  @Test
  func `simple object schema passes through unchanged`() {
    let input: Value = .object([
      "type": .string("object"),
      "properties": .object([
        "x": .object(["type": .string("integer")]),
      ]),
      "required": .array([.string("x")]),
    ])
    #expect(GeminiSchemaNormalizer.normalize(input) == input)
  }
}
