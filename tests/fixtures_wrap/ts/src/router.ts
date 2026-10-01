import { generateObject } from 'ai';
import { openai } from '@ai-sdk/openai';
import { z } from 'zod';

interface DecideOptions {
  prompt: string;
  schema: z.ZodTypeAny;
  maxTokens?: number;
}

export async function decide({ prompt, schema, ...rest }: DecideOptions) {
  const { object } = await generateObject({ model: openai('gpt-4o-mini'), prompt, schema, ...rest });
  return object;
}

const Route = z.object({ destination: z.enum(['search', 'chat', 'handoff']) });

export async function routeMessage(message: string) {
  return decide({ prompt: `Route this message: ${message}`, schema: Route, maxTokens: 20 });
}
