// Copyright © Anthony DePasquale

import Foundation
import UniformTypeIdentifiers

public extension Attachment {
  /// Creates an attachment by loading data from a local file URL.
  ///
  /// The MIME type and attachment category (image, video, audio, document) are
  /// determined from the file extension using `UTType`.
  ///
  /// - Parameter fileURL: A file URL pointing to a local file.
  /// - Throws: If the file cannot be read or its type cannot be determined.
  init(fileURL: URL) throws {
    guard fileURL.isFileURL else {
      throw AttachmentError.notAFileURL(fileURL)
    }

    let data = try Data(contentsOf: fileURL)
    let filename = fileURL.lastPathComponent
    let ext = fileURL.pathExtension

    guard !ext.isEmpty,
          let utType = UTType(filenameExtension: ext),
          let mimeType = utType.preferredMIMEType
    else {
      throw AttachmentError.unknownFileType(filename)
    }

    let kind: Kind = if utType.conforms(to: .image) {
      .image(data: data, mimeType: mimeType)
    } else if utType.conforms(to: .movie) {
      .video(data: data, mimeType: mimeType)
    } else if utType.conforms(to: .audio) {
      .audio(data: data, mimeType: mimeType)
    } else {
      .document(data: data, mimeType: mimeType)
    }

    self.init(kind: kind, filename: filename)
  }
}

/// Errors that can occur when creating an attachment from a file.
public enum AttachmentError: LocalizedError {
  case notAFileURL(URL)
  case unknownFileType(String)

  public var errorDescription: String? {
    switch self {
      case let .notAFileURL(url):
        "Not a file URL: \(url)"
      case let .unknownFileType(filename):
        "Could not determine file type for \(filename)"
    }
  }
}
