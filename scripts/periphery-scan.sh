#!/bin/bash
# Wrapper script for Periphery that handles the package manifest issue
# The swift package describe command outputs a warning before JSON, which breaks Periphery

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Generate clean package manifest (skip any warning lines before the JSON)
swift package describe --type json 2>/dev/null | sed -n '/^{/,$p' > /tmp/swift-ai-package-manifest.json

# Run Periphery with the pre-generated manifest
exec periphery scan --json-package-manifest-path /tmp/swift-ai-package-manifest.json "$@"
