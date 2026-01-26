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

  public init(provider: String, type: String, content: String? = nil,
              signature: String? = nil, data: String? = nil)
  {
    self.provider = provider
    self.type = type
    self.content = content
    self.signature = signature
    self.data = data
  }
}
