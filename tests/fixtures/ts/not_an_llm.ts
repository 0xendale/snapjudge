import { messages } from './chat';

export function post(text: string) {
  return messages.create({ text });
}
