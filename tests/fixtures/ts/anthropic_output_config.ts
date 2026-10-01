import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

export async function isRefundRequest(message: string) {
  return anthropic.messages.create({
    model: 'claude-haiku-4-5',
    max_tokens: 32,
    messages: [{ role: 'user', content: message }],
    output_config: {
      format: {
        type: 'json_schema',
        schema: { type: 'object', properties: { refund: { type: 'boolean' } }, required: ['refund'] },
      },
    },
  });
}
