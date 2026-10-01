import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import type { ZodTypeAny } from 'zod';

const client = new OpenAI();

export async function complete(messages: OpenAI.ChatCompletionMessageParam[], schema: ZodTypeAny) {
  return client.chat.completions.parse({
    model: 'gpt-4o-mini',
    messages,
    response_format: zodResponseFormat(schema, 'result'),
  });
}
