import type { ZodTypeAny } from 'zod';
import { triage } from './triage.js';

export const route = async (text: string, schema: ZodTypeAny) => triage(text, schema);
