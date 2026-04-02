// Copyright © Anthony DePasquale

import Foundation

/// A generic container for provider-specific content blocks that must be round-tripped.
///
/// Used to preserve provider metadata (e.g., Anthropic thinking blocks with cryptographic signatures)
/// through the response → Core Data → message rebuild cycle.
public struct OpaqueBlock: Sendable, Hashable, Codable {
  /// The provider that produced this block (e.g., "anthropic", "gemini").
  public let provider: String

  /// The block type (e.g., "thinking", "redacted_thinking").
  public let type: String

  /// The thinking text content, if any. Nil for redacted blocks.
  public let content: String?

  /// The verification token / cryptographic signature.
  public let signature: String?

  /// Encrypted content for redacted blocks.
  public let data: String?

  /// Whether `content` should be surfaced through `GenerationResponse.responseText`.
  /// Used for server-side tool output (e.g., code execution results, fetched web content)
  /// that is part of the response but must be round-tripped as structured data.
  public var isResponseContent: Bool = false

  enum CodingKeys: String, CodingKey {
    case provider, type, content, signature, data, isResponseContent
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    provider = try container.decode(String.self, forKey: .provider)
    type = try container.decode(String.self, forKey: .type)
    content = try container.decodeIfPresent(String.self, forKey: .content)
    signature = try container.decodeIfPresent(String.self, forKey: .signature)
    data = try container.decodeIfPresent(String.self, forKey: .data)
    // Default to false for data persisted before this field was added
    isResponseContent = try container.decodeIfPresent(Bool.self, forKey: .isResponseContent) ?? false
  }

  public init(provider: String, type: String, content: String? = nil,
              signature: String? = nil, data: String? = nil,
              isResponseContent: Bool = false)
  {
    self.provider = provider
    self.type = type
    self.content = content
    self.signature = signature
    self.data = data
    self.isResponseContent = isResponseContent
  }
}
