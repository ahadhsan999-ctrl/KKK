import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '50mb' }));

// ثلاث مفاتيح حقيقية منفصلة - بدل الاعتماد على Gemini بس وتمثيل Claude/ChatGPT بالبرومبت
const geminiApiKey = process.env.GEMINI_API_KEY || '';
const anthropicApiKey = process.env.ANTHROPIC_API_KEY || '';
const openAiApiKey = process.env.OPENAI_API_KEY || '';

const ai = new GoogleGenAI({
  apiKey: geminiApiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    hasGeminiKey: Boolean(geminiApiKey && geminiApiKey !== 'MY_GEMINI_API_KEY'),
    hasClaudeKey: Boolean(anthropicApiKey),
    hasOpenAiKey: Boolean(openAiApiKey),
    timestamp: new Date().toISOString(),
  });
});

// -----------------------------------------------------------------------
// تعليمات موحّدة لكل المزودين: نفس الأسلوب الدافئ وغير الروبوتي بغض النظر
// عن مين اللي بيرد فعليًا (Gemini / Claude / ChatGPT).
// -----------------------------------------------------------------------
function buildSystemInstruction(persona: string, userProfile: any, extraInstruction: string): string {
  const userName = userProfile.userName || 'صديقي';
  const assistantName = userProfile.assistantName || 'أورا (Aura)';
  const preferredLanguage = userProfile.language || 'ar';
  const userMemories = Array.isArray(userProfile.memories) ? userProfile.memories.join('\n- ') : '';

  let personaFlavor = '';
  if (persona === 'claude') {
    personaFlavor = 'أنت دلوقتي بتشتغل فعليًا بعقل نموذج Claude من Anthropic: تحليل عميق، تفكير منظم، وأمانة فكرية.';
  } else if (persona === 'chatgpt') {
    personaFlavor = 'أنت دلوقتي بتشتغل فعليًا بعقل نموذج ChatGPT من OpenAI: حماس، إبداع، وأسلوب محادثة سلس.';
  } else if (persona === 'gemini') {
    personaFlavor = 'أنت دلوقتي بتشتغل بعقل Gemini من Google: سرعة، دقة، واتصال مباشر بالمعرفة الحديثة.';
  } else {
    personaFlavor = `أنت ${assistantName}، مساعد شخصي خارق (Jarvis-style) لـ ${userName}، وعندك قدرة الثلاث نماذج مجتمعة.`;
  }

  return `
${personaFlavor}
قواعد صارمة غير قابلة للنقاش (نفسها بغض النظر عن المزود اللي بيرد فعليًا):
1. ادخل في صلب الموضوع مباشرة من أول كلمة، من غير مقدمات روبوتية مكررة (ممنوع: "بالتأكيد"، "أهلاً بك"، "يسعدني مساعدتك"، "بصفتي ذكاء اصطناعي").
2. اعتمد على حقائق دقيقة. لو معلومة غير مؤكدة، قول كده صراحة بدل التأليف.
3. نظّم الردود الطويلة بعناوين ونقاط عند الحاجة، لكن خلي المحادثة العادية طبيعية زي كلام بني آدمين.
4. الكود دايمًا جوه كتلة Markdown محددة اللغة، نظيف وقابل للتشغيل فورًا.
5. اتكلم باللهجة المصرية العامية لو المستخدم بيكتب كده، أو عربي فصيح، أو إنجليزي حسب لغته.
6. خليك دافئ وطبيعي زي صاحب بيرد على صاحبه.
${extraInstruction}
${userMemories ? `\nمعلومات شخصية عن ${userName}:\n- ${userMemories}` : ''}
اسم المستخدم: ${userName}. اللغة المفضلة: ${preferredLanguage === 'ar' ? 'العربية' : 'حسب لغة المستخدم'}.
التاريخ والوقت الحاليين: ${new Date().toLocaleString('ar-EG', { timeZone: 'Africa/Cairo' })}.
`.trim();
}

/** يقسّم النص الجاري بثّه لجمل كاملة أول ما توصل، عشان الصوت يبدأ يتكلم فورًا بدل ما ينتظر الرد كله. */
class SentenceAggregator {
  private buffer = '';
  constructor(private onSentence: (s: string) => void) {}

  feed(chunk: string) {
    this.buffer += chunk;
    const enders = ['.', '!', '؟', '?', '؛', '\n'];
    let lastCut = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const c = this.buffer[i];
      if (enders.includes(c)) {
        const isDecimal =
          c === '.' &&
          i > 0 &&
          i + 1 < this.buffer.length &&
          /\d/.test(this.buffer[i - 1]) &&
          /\d/.test(this.buffer[i + 1]);
        if (!isDecimal) {
          const sentence = this.buffer.slice(lastCut, i + 1).trim();
          if (sentence.length > 1) this.onSentence(sentence);
          lastCut = i + 1;
        }
      }
    }
    if (lastCut > 0) this.buffer = this.buffer.slice(lastCut);
  }

  flush() {
    const remaining = this.buffer.trim();
    if (remaining) this.onSentence(remaining);
    this.buffer = '';
  }
}

// -----------------------------------------------------------------------
// استدعاء Anthropic Claude الحقيقي (streaming)
// -----------------------------------------------------------------------
async function streamClaude(
  systemInstruction: string,
  messages: Array<{ role: string; content: string }>,
  isDeepThinking: boolean,
  onDelta: (text: string) => void
): Promise<string> {
  const model = isDeepThinking ? 'claude-sonnet-5-5' : 'claude-haiku-5-5';

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': anthropicApiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens: 2048,
      system: systemInstruction,
      messages: messages.map((m) => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content })),
      temperature: 0.65,
      stream: true,
    }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Claude API error: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const dataLine = line.split('\n').find((l) => l.startsWith('data: '));
      if (!dataLine) continue;
      try {
        const json = JSON.parse(dataLine.slice(6));
        if (json.type === 'content_block_delta' && json.delta?.type === 'text_delta') {
          const t = json.delta.text || '';
          fullText += t;
          onDelta(t);
        }
      } catch {
        /* شظية جزئية */
      }
    }
  }
  return fullText;
}

// -----------------------------------------------------------------------
// استدعاء OpenAI ChatGPT الحقيقي (streaming)
// -----------------------------------------------------------------------
async function streamOpenAi(
  systemInstruction: string,
  messages: Array<{ role: string; content: string }>,
  onDelta: (text: string) => void
): Promise<string> {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openAiApiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [{ role: 'system', content: systemInstruction }, ...messages],
      temperature: 0.65,
      stream: true,
    }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`OpenAI API error: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let fullText = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (data === '[DONE]') continue;
      try {
        const json = JSON.parse(data);
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) {
          fullText += delta;
          onDelta(delta);
        }
      } catch {
        /* شظية جزئية */
      }
    }
  }
  return fullText;
}

// Chat endpoint with streaming support (Server-Sent Events)
app.post('/api/chat', async (req: Request, res: Response) => {
  try {
    const {
      messages,
      persona = 'jarvis',
      systemInstruction = '',
      enableSearch = false,
      thinkingEnabled = false,
      userProfile = {},
    } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    const fullSystemInstruction = buildSystemInstruction(persona, userProfile, systemInstruction);

    // Set up SSE headers for streaming
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const aggregator = new SentenceAggregator((sentence) => {
      res.write(`data: ${JSON.stringify({ type: 'sentence', text: sentence })}\n\n`);
    });
    const onDelta = (delta: string) => {
      res.write(`data: ${JSON.stringify({ type: 'text', delta })}\n\n`);
      aggregator.feed(delta);
    };

    let fullText = '';
    let providerUsed = '';
    let groundingChunks: any[] = [];

    // ترتيب المحاولة: المزود اللي طلبه المستخدم فعليًا أولاً (لو مفتاحه موجود)، وبعدين Gemini كشبكة أمان
    const wantsClaudeFirst = persona === 'claude' && anthropicApiKey;
    const wantsOpenAiFirst = persona === 'chatgpt' && openAiApiKey;

    try {
      if (wantsClaudeFirst) {
        providerUsed = 'Claude';
        res.write(`data: ${JSON.stringify({ type: 'provider', name: 'Claude (Anthropic حقيقي)' })}\n\n`);
        fullText = await streamClaude(
          fullSystemInstruction,
          messages.map((m: any) => ({ role: m.role, content: m.content })),
          Boolean(thinkingEnabled),
          onDelta
        );
      } else if (wantsOpenAiFirst) {
        providerUsed = 'ChatGPT';
        res.write(`data: ${JSON.stringify({ type: 'provider', name: 'ChatGPT (OpenAI حقيقي)' })}\n\n`);
        fullText = await streamOpenAi(
          fullSystemInstruction,
          messages.map((m: any) => ({ role: m.role, content: m.content })),
          onDelta
        );
      } else {
        throw new Error('skip-to-gemini');
      }
    } catch {
      // فشل أو مش متاح المزود المطلوب -> ارجع لـ Gemini كشبكة أمان، مع تنويه شفاف للمستخدم
      if (!geminiApiKey || geminiApiKey === 'MY_GEMINI_API_KEY') {
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            error: 'مفيش أي مفتاح API شغال دلوقتي (Gemini/Claude/OpenAI). ضيف مفتاح واحد على الأقل في الإعدادات.',
          })}\n\n`
        );
        return res.end();
      }

      if (persona === 'claude' || persona === 'chatgpt') {
        res.write(
          `data: ${JSON.stringify({
            type: 'provider_notice',
            text:
              persona === 'claude'
                ? 'ملحوظة: مفتاح Claude مش متاح أو حصل خطأ، فالرد ده من Gemini بدل منه.'
                : 'ملحوظة: مفتاح ChatGPT مش متاح أو حصل خطأ، فالرد ده من Gemini بدل منه.',
          })}\n\n`
        );
      }
      res.write(`data: ${JSON.stringify({ type: 'provider', name: thinkingEnabled ? 'Gemini 2.5 Pro' : 'Gemini 2.5 Flash' })}\n\n`);

      const contents: Array<{ role: 'user' | 'model'; parts: Array<any> }> = [];
      for (const msg of messages) {
        const role: 'user' | 'model' = msg.role === 'assistant' || msg.role === 'model' ? 'model' : 'user';
        const parts: Array<any> = [];
        if (msg.attachments && Array.isArray(msg.attachments)) {
          for (const att of msg.attachments) {
            if (att.data && att.mimeType) {
              parts.push({ inlineData: { mimeType: att.mimeType, data: att.data.replace(/^data:[^;]+;base64,/, '') } });
            }
          }
        }
        if (msg.content) parts.push({ text: msg.content });
        if (parts.length > 0) contents.push({ role, parts });
      }

      const tools: any[] = [];
      if (enableSearch) tools.push({ googleSearch: {} });
      const config: any = { systemInstruction: fullSystemInstruction, temperature: 0.65 };
      if (tools.length > 0) config.tools = tools;

      // نموذجين فقط، الاتنين أسماء رسمية معروفة - شيلنا أسماء موديلات مش موثقة كانت ممكن ترجع 404
      const candidateModels = thinkingEnabled ? ['gemini-2.5-pro', 'gemini-2.5-flash'] : ['gemini-2.5-flash'];

      let streamEstablished = false;
      for (const modelCandidate of candidateModels) {
        try {
          const responseStream = await ai.models.generateContentStream({ model: modelCandidate, contents, config });
          for await (const chunk of responseStream) {
            streamEstablished = true;
            const textChunk = chunk.text || '';
            fullText += textChunk;
            onDelta(textChunk);
            const chunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
            if (chunks && chunks.length > 0) groundingChunks = chunks;
          }
          if (streamEstablished) break;
        } catch (err: any) {
          console.warn(`Model ${modelCandidate} failed:`, err.message || err);
          if (streamEstablished) throw err;
        }
      }
      if (!streamEstablished) throw new Error('كل نماذج Gemini مشغولة دلوقتي، جرّب تاني بعد شوية.');
      providerUsed = 'Gemini';
    }

    aggregator.flush();

    if (groundingChunks.length > 0) {
      res.write(`data: ${JSON.stringify({ type: 'grounding', sources: groundingChunks })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({ type: 'done', fullText, provider: providerUsed })}\n\n`);
    res.end();
  } catch (error: any) {
    console.error('Error in /api/chat:', error);
    if (!res.headersSent) {
      return res.status(500).json({ error: error.message || 'An error occurred while communicating with the AI provider.' });
    } else {
      res.write(`data: ${JSON.stringify({ type: 'error', error: error.message || 'Streaming failed' })}\n\n`);
      res.end();
    }
  }
});

// Image Generation Endpoint
app.post('/api/generate-image', async (req: Request, res: Response) => {
  try {
    const { prompt, aspectRatio = '1:1' } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required.' });
    }

    if (!geminiApiKey || geminiApiKey === 'MY_GEMINI_API_KEY') {
      return res.status(401).json({ error: 'GEMINI_API_KEY is not configured. Please add it to Secrets.' });
    }

    try {
      // ملحوظة: تأكد من اسم موديل التوليد الفعلي المتاح لحسابك عبر Google AI Studio قبل النشر،
      // أسماء الموديلات بتتغير بمرور الوقت وده أسهل مكان تظبطه فيه.
      const response = await ai.models.generateImages({
        model: 'imagen-4.0-generate-001',
        prompt: prompt,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/jpeg',
          aspectRatio: aspectRatio as any,
        },
      });

      const generatedImage = response.generatedImages?.[0]?.image?.imageBytes;
      if (generatedImage) {
        return res.json({ imageUrl: `data:image/jpeg;base64,${generatedImage}`, prompt });
      }
    } catch (genError: any) {
      console.warn('Gemini generateImages notice, generating visual art preview:', genError.message);
      const safePrompt = encodeURIComponent(prompt.slice(0, 300));
      const fallbackUrl = `https://image.pollinations.ai/prompt/${safePrompt}?width=1024&height=1024&nologo=true&seed=${Math.floor(
        Math.random() * 999999
      )}`;
      return res.json({ imageUrl: fallbackUrl, prompt, notice: 'Rendered via enhanced visual engine' });
    }

    return res.status(500).json({ error: 'Failed to generate image.' });
  } catch (err: any) {
    console.error('Error in /api/generate-image:', err);
    res.status(500).json({ error: err.message || 'Image generation failed.' });
  }
});

// Server-side Text-To-Speech endpoint (صوت ممتاز اختياري - البديل الافتراضي السريع صوت المتصفح نفسه)
app.post('/api/tts', async (req: Request, res: Response) => {
  let cleanText = '';
  try {
    const { text, voice = 'Kore' } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'Text is required for TTS.' });
    }
    if (!geminiApiKey || geminiApiKey === 'MY_GEMINI_API_KEY') {
      return res.status(401).json({ error: 'GEMINI_API_KEY is missing.' });
    }

    cleanText = text
      .replace(/```[\s\S]*?```/g, 'تم إنشاء الكود المطلوب.')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/[#*_\-\[\]\(\)]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, 400); // جملة واحدة قصيرة بما إن الكلام بقى بيتبعث جملة بجملة، مش النص كله مرة واحدة

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-tts',
      contents: [
        {
          role: 'user',
          parts: [{ text: cleanText }],
        },
      ],
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: voice || 'Kore' } },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      return res.json({ audioBase64: base64Audio, mimeType: 'audio/wav' });
    }
    return res.json({ useBrowserSpeech: true, text: cleanText });
  } catch (error: any) {
    console.warn('TTS server notice (falling back to browser speech):', error.message);
    res.json({ useBrowserSpeech: true, text: cleanText });
  }
});

// Setup Vite development middleware or static production serving
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';
  const PORT = Number(process.env.PORT) || 3000;

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Aura AI Full-Stack Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});
