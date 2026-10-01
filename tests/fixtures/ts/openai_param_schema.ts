import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';

const client = new OpenAI();

export async function ask(prompt: string, schema: any) {
  return client.chat.completions.parse({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    response_format: zodResponseFormat(schema, 'answer'),
  });
}
