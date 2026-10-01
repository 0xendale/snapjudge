import { generateObject } from 'ai';
import { openai } from '@ai-sdk/openai';

export default async function decideLabel(prompt: string, labels: string[]) {
  const { object } = await generateObject({
    model: openai('gpt-4o-mini'),
    output: 'enum',
    enum: labels,
    prompt,
  });
  return object;
}
