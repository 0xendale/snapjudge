import type { ZodTypeAny } from 'zod';
import { complete } from './llm.js';

export async function triage(text: string, schema: ZodTypeAny) {
  return complete(
    [
      { role: 'system', content: 'Triage the support ticket.' },
      { role: 'user', content: text },
    ],
    schema,
  );
}
