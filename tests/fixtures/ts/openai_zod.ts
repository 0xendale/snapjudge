import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

const client = new OpenAI();

const Sentiment = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']).describe('Overall sentiment of the review'),
  stars: z.number().int().min(1).max(5),
  reason: z.string(),
});

export async function sentiment(review: string) {
  const completion = await client.chat.completions.parse({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: review }],
    response_format: zodResponseFormat(Sentiment, 'sentiment'),
  });
  return completion.choices[0].message.parsed;
}
