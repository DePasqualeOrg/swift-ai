// Copyright © Anthony DePasquale

import Foundation

extension Attachment {
  var replayFallbackText: String {
    switch kind {
      case let .image(data, mimeType):
        return ToolResult.Content.image(data, mimeType: mimeType).fallbackDescription
      case let .audio(data, mimeType):
        return ToolResult.Content.audio(data, mimeType: mimeType).fallbackDescription
      case let .document(data, mimeType):
        return ToolResult.Content.file(data, mimeType: mimeType, filename: filename).fallbackDescription
      case let .video(data, mimeType):
        let size = ByteCountFormatter.string(fromByteCount: Int64(data.count), countStyle: .file)
        let filenameText = filename.map { "\($0) " } ?? ""
        return "[Unsupported attachment: \(filenameText)\(mimeType), \(size)]"
    }
  }
}

extension Message {
  func replayableTextSegmentsWithAttachmentFallback(includeEndnotes: Bool = true) -> [String] {
    replayableTextSegments(includeEndnotes: includeEndnotes, attachmentFallback: \.replayFallbackText)
  }
}
