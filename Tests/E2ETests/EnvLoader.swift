// Copyright © Anthony DePasquale

import Foundation

/// Loads environment variables from a .env file.
enum EnvLoader {
  /// Loads variables from a .env file and returns them as a dictionary.
  static func load(from path: String) throws -> [String: String] {
    let contents = try String(contentsOfFile: path, encoding: .utf8)
    var env: [String: String] = [:]

    for line in contents.components(separatedBy: .newlines) {
      let trimmed = line.trimmingCharacters(in: .whitespaces)

      // Skip empty lines and comments
      if trimmed.isEmpty || trimmed.hasPrefix("#") {
        continue
      }

      // Parse KEY=VALUE
      if let equalsIndex = trimmed.firstIndex(of: "=") {
        let key = String(trimmed[..<equalsIndex]).trimmingCharacters(in: .whitespaces)
        var value = String(trimmed[trimmed.index(after: equalsIndex)...]).trimmingCharacters(in: .whitespaces)

        // Remove surrounding quotes if present
        if (value.hasPrefix("\"") && value.hasSuffix("\"")) ||
          (value.hasPrefix("'") && value.hasSuffix("'"))
        {
          value = String(value.dropFirst().dropLast())
        }

        env[key] = value
      }
    }

    return env
  }

  /// Finds the package root by looking for Package.swift.
  static func findPackageRoot(from startPath: String = #filePath) -> String? {
    var current = URL(fileURLWithPath: startPath).deletingLastPathComponent()

    for _ in 0 ..< 10 {
      let packageSwift = current.appendingPathComponent("Package.swift")
      if FileManager.default.fileExists(atPath: packageSwift.path) {
        return current.path
      }
      current = current.deletingLastPathComponent()
    }

    return nil
  }

  /// Loads the .env file from the package root.
  static func loadFromPackageRoot() throws -> [String: String] {
    guard let root = findPackageRoot() else {
      throw EnvError.packageRootNotFound
    }
    let envPath = URL(fileURLWithPath: root).appendingPathComponent(".env").path
    return try load(from: envPath)
  }

  enum EnvError: Error {
    case packageRootNotFound
  }
}
