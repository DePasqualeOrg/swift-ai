// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isOpenAIChatCompletionsRefusal: Bool {
    provider == Self.ProviderID.openAIChatCompletions && type == Self.OpenAIChatCompletionsType.refusal
  }

  var isOpenAIResponsesAnnotatedOutputText: Bool {
    provider == Self.ProviderID.openAIResponses && type == Self.OpenAIResponsesType.annotatedOutputText
  }

  var isOpenAIResponsesRefusal: Bool {
    provider == Self.ProviderID.openAIResponses && type == Self.OpenAIResponsesType.refusal
  }

  var isOpenAIResponsesMessageMetadata: Bool {
    provider == Self.ProviderID.openAIResponses && type == Self.OpenAIResponsesType.messageMetadata
  }

  var isOpenAIResponsesReasoning: Bool {
    provider == Self.ProviderID.openAIResponses && type == Self.OpenAIResponsesType.reasoning
  }
}
