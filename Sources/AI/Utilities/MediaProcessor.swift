// Copyright © Anthony DePasquale

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - MediaProcessor

/// Internal utility for processing media attachments for LLM APIs.
enum MediaProcessor {
  /// Default constraints for image resizing (matches Anthropic's recommendations)
  static let defaultMaxDimension: Double = 1568
  static let defaultMaxMegapixels: Double = 1.15

  /// Processes image data with orientation correction and resizing constraints.
  /// - Parameters:
  ///   - imageData: The raw image data
  ///   - mimeType: The MIME type of the image (e.g., "image/jpeg", "image/png")
  ///   - maxDimension: Maximum width or height in pixels
  ///   - maxMegapixels: Maximum total megapixels
  /// - Returns: Processed image data, or original if no processing needed
  static func resizeImageIfNeeded(
    _ imageData: Data,
    mimeType: String,
    maxDimension: Double = defaultMaxDimension,
    maxMegapixels: Double = defaultMaxMegapixels
  ) async throws -> Data {
    guard let source = CGImageSourceCreateWithData(imageData as CFData, nil) else {
      throw AIError.parsing(message: "Unable to create image source")
    }

    guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Double,
          let height = properties[kCGImagePropertyPixelHeight] as? Double
    else {
      throw AIError.parsing(message: "Unable to read image properties")
    }

    let megapixels = (width * height) / 1_000_000.0

    // Check if any processing is needed
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    let needsOrientationCorrection = orientation != 1
    let needsResizing = width > maxDimension || height > maxDimension || megapixels > maxMegapixels

    if !needsResizing, !needsOrientationCorrection {
      return imageData
    }

    // Calculate the maximum pixel size that satisfies all constraints
    let megapixelConstraint = sqrt(maxMegapixels * 1_000_000)
    let maxPixelSize = min(maxDimension, megapixelConstraint)

    // Get correctly oriented and sized thumbnail
    let thumbnail = try await Task(priority: .userInitiated) {
      let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
      ]
      guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
        throw AIError.parsing(message: "Failed to create thumbnail")
      }
      return thumbnail
    }.value

    // Determine output format from MIME type
    let outputUTType = utType(for: mimeType) ?? .jpeg
    let needsAlphaRemoval = !supportsTransparency(mimeType: mimeType)

    // For formats that don't support transparency, remove alpha channel
    let finalImage: CGImage
    if needsAlphaRemoval {
      let colorSpace = CGColorSpaceCreateDeviceRGB()
      guard let context = CGContext(
        data: nil,
        width: thumbnail.width,
        height: thumbnail.height,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
      ) else {
        throw AIError.parsing(message: "Failed to create graphics context")
      }
      // Fill with white background
      context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: thumbnail.width, height: thumbnail.height))
      context.draw(thumbnail, in: CGRect(x: 0, y: 0, width: thumbnail.width, height: thumbnail.height))
      guard let opaqueImage = context.makeImage() else {
        throw AIError.parsing(message: "Failed to create final image")
      }
      finalImage = opaqueImage
    } else {
      finalImage = thumbnail
    }

    // Convert to desired format
    guard let mutableData = CFDataCreateMutable(nil, 0),
          let destination = CGImageDestinationCreateWithData(mutableData, outputUTType.identifier as CFString, 1, nil)
    else {
      throw AIError.parsing(message: "Failed to create image destination")
    }

    let compressionQuality: CFNumber = 0.85 as CFNumber
    let destinationProperties: [CFString: Any] = needsAlphaRemoval
      ? [kCGImageDestinationLossyCompressionQuality: compressionQuality]
      : [:]

    CGImageDestinationAddImage(destination, finalImage, destinationProperties as CFDictionary)

    guard CGImageDestinationFinalize(destination) else {
      throw AIError.parsing(message: "Failed to finalize image destination")
    }

    return mutableData as Data
  }

  /// Converts data to a base64-encoded data URL string.
  /// - Parameters:
  ///   - data: The raw data to encode
  ///   - mimeType: The MIME type for the data URL
  /// - Returns: A data URL string like "data:image/jpeg;base64,..."
  static func toBase64DataURL(_ data: Data, mimeType: String) -> String {
    "data:\(mimeType);base64,\(data.base64EncodedString())"
  }

  // MARK: - Private Helpers

  /// Returns the UTType for a given MIME type.
  private static func utType(for mimeType: String) -> UTType? {
    switch mimeType.lowercased() {
      case "image/jpeg", "image/jpg": .jpeg
      case "image/png": .png
      case "image/gif": .gif
      case "image/webp": .webP
      case "image/heic": .heic
      case "image/heif": UTType("public.heif")
      default: UTType(mimeType: mimeType)
    }
  }

  /// Returns whether the given MIME type supports transparency.
  private static func supportsTransparency(mimeType: String) -> Bool {
    switch mimeType.lowercased() {
      case "image/png", "image/gif", "image/webp": true
      default: false
    }
  }
}
