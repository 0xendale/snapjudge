import { generateObject } from 'ai';
import { openai } from '@ai-sdk/openai';

export async function genre(title: string) {
  const { object } = await generateObject({
    model: openai('gpt-4o-mini'),
    output: 'enum',
    enum: ['action', 'comedy', 'drama', 'horror', 'sci-fi'],
    prompt: `Classify the movie genre: ${title}`,
  });
  return object;
}
