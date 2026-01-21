// Copyright © Anthony DePasquale

import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct AIMacrosPlugin: CompilerPlugin {
  let providingMacros: [Macro.Type] = [
    ToolMacro.self,
  ]
}
