import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

const client = new OpenAI();

export async function triageBy(kind: string, text: string) {
  switch (kind) {
    case 'email':
      const Priority = z.object({ priority: z.enum(['low', 'high']) });
      return client.chat.completions.parse({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: text }],
        response_format: zodResponseFormat(Priority, 'result'),
      });
    default:
      const Verdict = z.object({ verdict: z.enum(['spam', 'ham']) });
      return client.chat.completions.parse({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: text }],
        response_format: zodResponseFormat(Verdict, 'result'),
      });
  }
}
