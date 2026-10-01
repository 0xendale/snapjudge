import { generateText } from 'ai';

export async function summarize(doc: string) {
  const { text } = await generateText({
    model: 'anthropic/claude-sonnet-5',
    system: 'Summarize the document in three short paragraphs.',
    prompt: doc,
  });
  return text;
}
