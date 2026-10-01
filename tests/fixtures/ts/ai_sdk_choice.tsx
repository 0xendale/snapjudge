import { generateText, Output } from 'ai';

export async function Badge({ text }: { text: string }) {
  const { output } = await generateText({
    model: 'openai/gpt-5-mini',
    output: Output.choice({ options: ['urgent', 'normal', 'low'] }),
    prompt: `How urgent is this message? ${text}`,
  });
  return <span className="badge">{output}</span>;
}
