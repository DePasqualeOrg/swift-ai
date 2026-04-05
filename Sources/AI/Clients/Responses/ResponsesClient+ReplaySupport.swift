// Copyright © Anthony DePasquale

extension OpaqueBlock {
  var isOpenAIChatCompletionsRefusal: Bool {
    provider == "openai-chat-completions" && type == "refusal"
  }

  var isOpenAIResponsesAnnotatedOutputText: Bool {
    provider == "openai-responses" && type == "annotated_output_text"
  }

  var isOpenAIResponsesRefusal: Bool {
    provider == "openai-responses" && type == "refusal"
  }

  var isOpenAIResponsesMessageMetadata: Bool {
    provider == "openai-responses" && type == "message_metadata"
  }

  var isOpenAIResponsesReasoning: Bool {
    provider == "openai-responses" && type == "reasoning"
  }
}
