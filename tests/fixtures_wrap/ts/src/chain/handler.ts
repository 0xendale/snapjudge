import type { ZodTypeAny } from 'zod';
import { route } from './route.js';

export async function handle(req: Request, schema: ZodTypeAny) {
  return route(await req.text(), schema);
}
