// Copyright © Anthony DePasquale

// AIMCP - Bridge between swift-ai and swift-mcp
//
// This module provides conversions between:
// - AI.Value ↔ MCP.Value
// - AI.Tool ↔ MCP.Tool
// - AI.ToolResult ↔ MCP.CallTool.Result
// - AI.GenerationResponse.ToolCall ↔ MCP.CallTool.Parameters
//
// And a high-level MCPToolProvider for integrating MCP tools with AI providers.

@_exported import AI
@_exported import MCP
