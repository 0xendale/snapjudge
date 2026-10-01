import { z } from 'zod';
import { complete } from './llm.js';
import { triage } from './triage.js';
import { route } from './route.js';
import { handle } from './handler.js';

const Priority = z.object({ level: z.enum(['low', 'medium', 'high']) });

export async function serve(req: Request) {
  const urgent = await complete([{ role: 'user', content: 'How urgent is this ticket?' }], Priority);
  const triaged = await triage('The printer is on fire', Priority);
  const routed = await route('The printer is on fire', Priority);
  const handled = await handle(req, Priority);
  return [urgent, triaged, routed, handled];
}
