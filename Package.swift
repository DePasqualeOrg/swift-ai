// swift-tools-version: 6.1

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "swift-ai",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
  ],
  products: [
    .library(
      name: "AI",
      targets: ["AI"],
    ),
    .library(
      name: "AITool",
      targets: ["AITool"],
    ),
    .library(
      name: "AIMCP",
      targets: ["AIMCP"],
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-sse", .upToNextMinor(from: "0.1.0")),
    .package(url: "https://github.com/DePasqualeOrg/swift-mcp", from: "1.1.0"),
    .package(url: "https://github.com/ajevans99/swift-json-schema", .upToNextMinor(from: "0.11.2")),
    .package(url: "https://github.com/swiftlang/swift-syntax", "603.0.0" ..< "604.0.0"),
  ],
  targets: [
    .macro(
      name: "AIMacros",
      dependencies: [
        .product(name: "SwiftSyntax", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
      ],
    ),
    .target(
      name: "AI",
      dependencies: [
        .product(name: "SSE", package: "swift-sse"),
        .product(name: "JSONSchema", package: "swift-json-schema"),
        .product(name: "JSONSchemaBuilder", package: "swift-json-schema"),
      ],
    ),
    .target(
      name: "AITool",
      dependencies: [
        "AI",
        "AIMacros",
        .product(name: "JSONSchema", package: "swift-json-schema"),
        .product(name: "JSONSchemaBuilder", package: "swift-json-schema"),
      ],
    ),
    .target(
      name: "AIMCP",
      dependencies: [
        "AI",
        .product(name: "MCP", package: "swift-mcp"),
      ],
    ),
    .testTarget(
      name: "AITests",
      dependencies: ["AI", "AITool"],
      resources: [.copy("Fixtures")],
    ),
    .testTarget(
      name: "AIMCPTests",
      dependencies: ["AIMCP"],
    ),
    .testTarget(
      name: "AIMacroTests",
      dependencies: [
        "AIMacros",
        .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax"),
      ],
    ),
    .testTarget(
      name: "E2ETests",
      dependencies: ["AIMCP", "AITool"],
    ),

    // MARK: - Examples

    .target(
      name: "ExamplesShared",
      path: "Examples/Shared",
    ),
    .executableTarget(
      name: "AgenticLoop",
      dependencies: ["AI", "AITool", "ExamplesShared"],
      path: "Examples/AgenticLoop",
    ),
  ],
)
