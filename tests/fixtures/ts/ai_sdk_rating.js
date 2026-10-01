import { generateText } from 'ai';

export async function rate(answer) {
  const { text } = await generateText({
    model: 'openai/gpt-5-mini',
    prompt: `Rate the answer on a scale of 1 to 5. Reply with the number only.\n\n${answer}`,
    maxOutputTokens: 2,
  });
  return Number(text);
}
