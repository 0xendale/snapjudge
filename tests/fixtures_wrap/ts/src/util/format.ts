export function decide(score: number): string {
  return score > 0.5 ? 'yes' : 'no';
}

export function ask(question: string): string {
  return `${question.trim()}?`;
}

console.log(decide(0.7), ask('Ready'));
