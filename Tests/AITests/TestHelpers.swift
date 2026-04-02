// Copyright © Anthony DePasquale

@testable import AI
import Foundation
import os
import Testing

/// Thread-safe collector for streaming updates in tests.
/// Uses a lock instead of an actor since update closures are synchronous.
final class UpdateCollector: Sendable {
  private let storage = OSAllocatedUnfairLock(initialState: [GenerationResponse]())

  func append(_ response: GenerationResponse) {
    storage.withLock { $0.append(response) }
  }

  var updates: [GenerationResponse] {
    storage.withLock { Array($0) }
  }
}

/// Loads a fixture file from the Fixtures directory relative to the caller's file.
func loadFixture(_ name: String, relativeTo filePath: String = #filePath) throws -> String {
  let fixturesURL = URL(fileURLWithPath: filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures")
    .appendingPathComponent(name)

  return try String(contentsOf: fixturesURL, encoding: .utf8)
}

/// Creates a test tool with a single string parameter.
func makeTestTool(name: String, description: String, paramName: String) -> Tool {
  Tool(
    name: name,
    description: description,
    title: name,
    parameters: [
      Tool.Parameter(
        name: paramName,
        title: paramName,
        type: .string,
        description: "Test parameter",
        required: true,
      ),
    ],
    execute: { _ in [.text("test result")] },
  )
}

/// Consumes an async stream and returns the last element.
func consumeStream(
  _ stream: AsyncThrowingStream<GenerationResponse, Error>,
  collecting: UpdateCollector? = nil,
) async throws -> GenerationResponse {
  var last: GenerationResponse?
  for try await response in stream {
    collecting?.append(response)
    last = response
  }
  guard let result = last else {
    fatalError("Stream ended without producing any values")
  }
  return result
}

/// Reads request body from either httpBody or httpBodyStream.
func readRequestBody(from request: URLRequest) -> Data? {
  if let body = request.httpBody {
    return body
  }
  if let stream = request.httpBodyStream {
    stream.open()
    defer { stream.close() }
    var data = Data()
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
    defer { buffer.deallocate() }
    while stream.hasBytesAvailable {
      let count = stream.read(buffer, maxLength: 4096)
      if count > 0 {
        data.append(buffer, count: count)
      } else {
        break
      }
    }
    return data.isEmpty ? nil : data
  }
  return nil
}
