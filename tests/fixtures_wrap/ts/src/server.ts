import express from 'express';
import OpenAI from 'openai';

const client = new OpenAI();
const app = express();

app.post('/chat', async (req, res) => {
  const completion = await client.chat.completions.create({ model: 'gpt-4o-mini', messages: req.body.messages });
  res.json(completion);
});
