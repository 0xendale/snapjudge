import { ask } from 'prompt-kit';

export async function confirm() {
  return ask('Continue with the deployment?');
}
