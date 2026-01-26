// Copyright © Anthony DePasquale

import AI
import MCP

/// A progress update from a tool call, bridging MCP progress notifications
/// to the AI tool execution layer.
public struct ToolCallProgressUpdate: Sendable {
  /// The ID of the tool call this progress belongs to, if known.
  public let toolCallId: String?

  /// The display name of the tool being executed.
  public let toolName: String

  /// The current progress value (monotonically increasing).
  public let value: Double

  /// The total progress value, if known.
  public let total: Double?

  /// A human-readable message describing current progress.
  public let message: String?

  public init(
    toolCallId: String?,
    toolName: String,
    value: Double,
    total: Double? = nil,
    message: String? = nil
  ) {
    self.toolCallId = toolCallId
    self.toolName = toolName
    self.value = value
    self.total = total
    self.message = message
  }
}

/// Provides a task-local callback for receiving tool call progress updates.
///
/// Set the `onProgress` task-local before calling `Tools.call()` to receive
/// progress notifications from MCP tools that support them.
///
/// ```swift
/// let results = await ToolCallProgress.$onProgress.withValue({ update in
///     print("\(update.toolName): \(update.message ?? "")")
/// }) {
///     await tools.call(toolCalls)
/// }
/// ```
public enum ToolCallProgress {
  @TaskLocal public static var onProgress: (@Sendable (ToolCallProgressUpdate) async -> Void)?
}
