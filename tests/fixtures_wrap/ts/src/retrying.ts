import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z, type ZodTypeAny } from 'zod';

const client = new OpenAI();

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch {
    return await fn();
  }
}

export async function pickLabel(text: string, schema: ZodTypeAny) {
  return withRetry(() =>
    client.chat.completions.parse({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: text }],
      response_format: zodResponseFormat(schema, 'result'),
    }),
  );
}

const Label = z.object({ label: z.enum(['bug', 'feature', 'question']) });

export async function labelIssue(body: string) {
  return pickLabel(body, Label);
}

export async function labelInline(body: string) {
  return client.chat.completions.parse({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: body }],
    response_format: zodResponseFormat(Label, 'result'),
  });
}
