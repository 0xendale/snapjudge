import OpenAI from 'openai';
import { zodTextFormat } from 'openai/helpers/zod';
import { z } from 'zod';

const client = new OpenAI();
const Toxicity = z.object({ toxic: z.boolean(), category: z.enum(['insult', 'threat', 'none']) });

export async function check(text: string) {
  const r = await client.responses.parse({
    model: 'gpt-4o',
    input: [{ role: 'user', content: text }],
    text: { format: zodTextFormat(Toxicity, 'toxicity') },
  });
  return r.output_parsed;
}
