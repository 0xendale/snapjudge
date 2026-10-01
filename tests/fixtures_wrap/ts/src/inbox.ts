import pickLabel from './labeler.js';

export async function sortEmail(body: string) {
  return pickLabel(`Which folder should this email go to?\n${body}`, ['inbox', 'promotions', 'spam']);
}
