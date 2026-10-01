import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

const Relevance = z.object({ relevant: z.boolean().describe('Is the document relevant to the question?') });

export const grader = new ChatOpenAI({ model: 'gpt-4o-mini', temperature: 0 }).withStructuredOutput(Relevance);
