// Copyright © Anthony DePasquale

import Foundation

/// A file attachment for a message.
///
/// Uses a `Kind` enum for category-based handling (different processing per media type)
/// with `mimeType` string for format flexibility within each category.
public struct Attachment: Sendable, Hashable {
  /// The kind of attachment, including its data and MIME type.
  public enum Kind: Sendable, Hashable {
    case image(data: Data, mimeType: String)
    case document(data: Data, mimeType: String)
    case video(data: Data, mimeType: String)
    case audio(data: Data, mimeType: String)

    /// The raw data for this attachment.
    public var data: Data {
      switch self {
        case let .image(data, _): data
        case let .document(data, _): data
        case let .video(data, _): data
        case let .audio(data, _): data
      }
    }

    /// The MIME type for this attachment.
    public var mimeType: String {
      switch self {
        case let .image(_, mimeType): mimeType
        case let .document(_, mimeType): mimeType
        case let .video(_, mimeType): mimeType
        case let .audio(_, mimeType): mimeType
      }
    }
  }

  /// The type and content of this attachment.
  public let kind: Kind

  /// An optional filename for this attachment.
  public let filename: String?

  /// Creates a new attachment.
  ///
  /// - Parameters:
  ///   - kind: The type and content of the attachment.
  ///   - filename: An optional filename for the attachment.
  public init(kind: Kind, filename: String? = nil) {
    self.kind = kind
    self.filename = filename
  }
}
