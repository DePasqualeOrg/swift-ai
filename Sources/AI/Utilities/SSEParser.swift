// Copyright © Anthony DePasquale

import Foundation

/// Parses Server-Sent Events (SSE) streams and yields data payloads.
///
/// SSE streams consist of lines prefixed with "data: " containing JSON payloads.
/// This parser handles:
/// - Extracting data from "data: " prefixed lines
/// - Skipping empty lines and non-data lines (like "event:" or comments)
/// - Optional "[DONE]" termination (used by OpenAI APIs)
/// - Task cancellation checking
enum SSEParser {
  /// Parses an SSE byte stream and yields data payload strings.
  ///
  /// - Parameters:
  ///   - bytes: The async byte stream from URLSession
  ///   - terminateOnDone: If true, stops parsing when "[DONE]" is received (default: true)
  /// - Returns: An async throwing stream of data payload strings (without the "data: " prefix)
  static func dataPayloads(
    from bytes: URLSession.AsyncBytes,
    terminateOnDone: Bool = true
  ) -> AsyncThrowingStream<String, Error> {
    AsyncThrowingStream { continuation in
      let task = Task {
        do {
          for try await line in bytes.lines {
            try Task.checkCancellation()

            guard line.hasPrefix("data: ") else {
              continue
            }

            let payload = String(line.dropFirst(6))

            if terminateOnDone, payload == "[DONE]" {
              break
            }

            continuation.yield(payload)
          }
          continuation.finish()
        } catch is CancellationError {
          continuation.finish(throwing: CancellationError())
        } catch {
          continuation.finish(throwing: error)
        }
      }

      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }
}
