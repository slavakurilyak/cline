import { Groq } from "groq-sdk"
import { ApiHandler } from "../index"
import { ApiHandlerOptions, GroqModelId, groqDefaultModelId, groqModels, ModelInfo } from "../../shared/api"
import { ApiStream } from "../transform/stream"

export type ChatCompletionContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }

export type ToolUseParam = {
	type: "tool_use"
	id: string
	name: string
	input: Record<string, any>
}

export type ToolResultParam = {
	type: "tool_result"
	tool_use_id: string
	content: string | ChatCompletionContentPart[]
}

export type ChatMessageParam = {
	role: "user" | "assistant"
	content: string | ChatCompletionContentPart[] | ToolUseParam[] | ToolResultParam[]
}

export class GroqHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: Groq

	constructor(options: ApiHandlerOptions) {
		this.options = options
		this.client = new Groq({
			apiKey: this.options.groqApiKey,
		})
	}

	async *createMessage(systemPrompt: string, messages: unknown[]): ApiStream {
		// Ensure messages are in the expected format for Groq
		const chatMessages = messages as ChatMessageParam[]

		// Groq advises to merge system prompt with first user message
		const systemPromptMessage = {
			role: "system",
			content: systemPrompt.trim(),
		}

		const finalMessages = [systemPromptMessage, ...chatMessages] as {
			role: "user" | "assistant" | "system"
			content: string
		}[]

		const { id: modelId } = this.getModel()

		const response = await this.client.chat.completions.create({
			model: modelId,
			messages: finalMessages,
			stream: true,
		})

		for await (const chunk of response) {
			const delta = chunk.choices?.[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}
		}
	}

	getModel(): { id: GroqModelId; info: ModelInfo } {
		const modelId = this.options.groqModelId
		if (modelId && modelId in groqModels) {
			const id = modelId as GroqModelId
			return { id, info: groqModels[id] }
		}
		return {
			id: groqDefaultModelId,
			info: groqModels[groqDefaultModelId],
		}
	}
}
