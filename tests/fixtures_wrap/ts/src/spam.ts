import OpenAI from 'openai';

const client = new OpenAI();

export async function ask(prompt: string, opts: Partial<OpenAI.ChatCompletionCreateParamsNonStreaming> = {}) {
  return client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    ...opts,
  });
}

export async function isSpam(message: string) {
  return ask(`Is this message spam? Answer yes or no.\n${message}`, { max_tokens: 1 });
}
