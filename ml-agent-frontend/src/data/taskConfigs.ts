/**
 * Hardcoded task → dataset → model → simulation configs.
 * Each entry drives a fully customised simulation run.
 */

export interface IterationTemplate {
  experiment: string;
  diff: string;
  lossAfter: number;
  metricAfter: number;   // normalised 0-1 (displayed in logs as real value)
  status: 'KEPT' | 'REVERTED';
}

export interface DataSampleRow {
  [field: string]: string | number;
}

export interface TaskConfig {
  // ── Matching ──────────────────────────────────────────────────────────────
  keywords: string[];

  // ── Display ───────────────────────────────────────────────────────────────
  taskLabel: string;
  datasets: Array<{ name: string; size: string }>;
  model: string;
  evalMetric: string;       // e.g. "accuracy", "F1", "ROUGE-L", "BLEU"
  metricLabel: string;      // e.g. "Val Accuracy", "F1 Score", "ROUGE-L"
  strategy: 'fine-tune' | 'pre-train';
  trainingType: 'SFT' | 'RL';
  loraRank: number;
  targetModules: string;

  // ── Sample rows shown in the Data Discovery panel ─────────────────────────
  samples: DataSampleRow[];

  // ── Metric ranges (normalised 0-1; chart shows ×100) ─────────────────────
  baseline: { loss: number; metric: number };
  final: { loss: number; metric: number };

  // ── Per-experiment iterations ─────────────────────────────────────────────
  iterations: IterationTemplate[];
}

// ─────────────────────────────────────────────────────────────────────────────
// TEXT CLASSIFICATION
// ─────────────────────────────────────────────────────────────────────────────

const classificationIterations: IterationTemplate[] = [
  {
    experiment: 'Decrease learning_rate 3e-4→1.5e-4 to reduce loss spikes.',
    diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015',
    lossAfter: 0.289, metricAfter: 0.882, status: 'KEPT',
  },
  {
    experiment: 'Increase lora_rank 16→32 to expand model capacity.',
    diff: '- lora_rank: 16\n+ lora_rank: 32',
    lossAfter: 0.271, metricAfter: 0.901, status: 'KEPT',
  },
  {
    experiment: 'Increase learning_rate 1.5e-4→6.1e-4 (±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00061',
    lossAfter: 0.334, metricAfter: 0.862, status: 'REVERTED',
  },
  {
    experiment: 'Increase warmup_steps 100→500 to stabilise early training.',
    diff: '- warmup_steps: 100\n+ warmup_steps: 500',
    lossAfter: 0.248, metricAfter: 0.910, status: 'KEPT',
  },
  {
    experiment: 'Add weight_decay 0→0.01 to reduce overfitting.',
    diff: '- weight_decay: 0.0\n+ weight_decay: 0.01',
    lossAfter: 0.253, metricAfter: 0.906, status: 'REVERTED',
  },
  {
    experiment: 'Decrease dropout 0.1→0.06 (±20% perturbation).',
    diff: '- dropout: 0.1\n+ dropout: 0.06',
    lossAfter: 0.243, metricAfter: 0.914, status: 'KEPT',
  },
  {
    experiment: 'Decrease learning_rate 1.5e-4→1.2e-4 (±20% perturbation).',
    diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00012',
    lossAfter: 0.231, metricAfter: 0.923, status: 'KEPT',
  },
  {
    experiment: 'Increase num_epochs 3→4 to allow longer convergence.',
    diff: '- num_epochs: 3\n+ num_epochs: 4',
    lossAfter: 0.228, metricAfter: 0.919, status: 'REVERTED',
  },
];

export const MOVIE_SENTIMENT: TaskConfig = {
  keywords: ['movie', 'film', 'review', 'sentiment', 'positive', 'negative', 'imdb'],
  taskLabel: 'Sentiment Classification',
  datasets: [
    { name: 'SetFit/imdb', size: '25k' },
    { name: 'stanfordnlp/sst2', size: '67k' },
  ],
  model: 'distilbert-base-uncased',
  evalMetric: 'accuracy', metricLabel: 'Val Accuracy',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { text: 'One of the best films I have seen in years. The acting was superb and the story kept me hooked throughout.', label: 'positive' },
    { text: 'Terrible waste of two hours. The plot made no sense and the characters were completely unbelievable.', label: 'negative' },
    { text: 'An absolute masterpiece. Spielberg at his finest — emotional, thrilling, and visually stunning.', label: 'positive' },
    { text: 'I fell asleep halfway through. Nothing happens for the first hour and the ending is disappointing.', label: 'negative' },
    { text: 'Surprisingly good! I had low expectations but this film genuinely moved me. Highly recommended.', label: 'positive' },
  ],
  baseline: { loss: 0.421, metric: 0.741 },
  final:    { loss: 0.214, metric: 0.931 },
  iterations: classificationIterations,
};

export const HATE_SPEECH: TaskConfig = {
  keywords: ['hate', 'speech', 'toxic', 'offensive', 'tweet', 'twitter', 'harmful'],
  taskLabel: 'Hate Speech Detection',
  datasets: [{ name: 'SetFit/hate_speech_offensive', size: '24k' }],
  model: 'distilbert-base-uncased',
  evalMetric: 'F1', metricLabel: 'Macro F1',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { tweet: 'Just had the most amazing day at the park with my family!', label: 'neither' },
    { tweet: '@user you are such a [slur], go back to where you came from', label: 'hate_speech' },
    { tweet: 'These politicians are all corrupt idiots who should be fired', label: 'offensive' },
    { tweet: 'Congratulations on your promotion! You deserve it so much 🎉', label: 'neither' },
    { tweet: '@user I hope you get what\'s coming to you, you disgusting excuse for a human', label: 'hate_speech' },
  ],
  baseline: { loss: 0.481, metric: 0.712 },
  final:    { loss: 0.261, metric: 0.891 },
  iterations: [
    { experiment: 'Add class_weights to address label imbalance.', diff: '- class_weights: null\n+ class_weights: [0.3, 0.7]', lossAfter: 0.398, metricAfter: 0.761, status: 'KEPT' },
    { experiment: 'Increase lora_rank 16→32 for richer representations.', diff: '- lora_rank: 16\n+ lora_rank: 32', lossAfter: 0.361, metricAfter: 0.793, status: 'KEPT' },
    { experiment: 'Decrease learning_rate 3e-4→1.5e-4.', diff: '- learning_rate: 0.0003\n+ learning_rate: 0.00015', lossAfter: 0.318, metricAfter: 0.831, status: 'KEPT' },
    { experiment: 'Increase dropout 0.05→0.15 for regularisation.', diff: '- dropout: 0.05\n+ dropout: 0.15', lossAfter: 0.341, metricAfter: 0.811, status: 'REVERTED' },
    { experiment: 'Tune warmup_steps 100→400.', diff: '- warmup_steps: 100\n+ warmup_steps: 400', lossAfter: 0.298, metricAfter: 0.851, status: 'KEPT' },
    { experiment: 'Add label_smoothing 0→0.1.', diff: '- label_smoothing: 0.0\n+ label_smoothing: 0.1', lossAfter: 0.312, metricAfter: 0.841, status: 'REVERTED' },
    { experiment: 'Decrease learning_rate 1.5e-4→1.1e-4.', diff: '- learning_rate: 0.00015\n+ learning_rate: 0.00011', lossAfter: 0.271, metricAfter: 0.878, status: 'KEPT' },
    { experiment: 'Increase batch_size 16→32.', diff: '- batch_size: 16\n+ batch_size: 32', lossAfter: 0.261, metricAfter: 0.891, status: 'KEPT' },
  ],
};

export const NEWS_CLASSIFICATION: TaskConfig = {
  keywords: ['news', 'article', 'sports', 'politics', 'tech', 'technology', 'category', 'topic'],
  taskLabel: 'News Topic Classification',
  datasets: [{ name: 'SetFit/ag_news', size: '120k' }],
  model: 'distilbert-base-uncased',
  evalMetric: 'accuracy', metricLabel: 'Val Accuracy',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { text: 'Apple unveils the new M4 chip promising 40% faster performance for developers.', label: 'sci/tech' },
    { text: 'The Lakers secured a playoff spot after defeating the Celtics 112-98 on Tuesday.', label: 'sports' },
    { text: 'Congress passes bipartisan infrastructure bill with $1.2 trillion in new spending.', label: 'world/politics' },
    { text: 'Wall Street surges as Fed signals pause in interest rate hikes amid cooling inflation.', label: 'business' },
    { text: 'Researchers at MIT develop new battery technology that charges in under 3 minutes.', label: 'sci/tech' },
  ],
  baseline: { loss: 0.389, metric: 0.821 },
  final:    { loss: 0.198, metric: 0.948 },
  iterations: classificationIterations,
};

export const EMOTION: TaskConfig = {
  keywords: ['emotion', 'joy', 'anger', 'sadness', 'fear', 'happy', 'sad', 'feeling'],
  taskLabel: 'Emotion Classification',
  datasets: [{ name: 'SetFit/emotion', size: '20k' }],
  model: 'distilbert-base-uncased',
  evalMetric: 'accuracy', metricLabel: 'Val Accuracy',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { text: 'I just got the job offer I\'ve been waiting for — I can\'t stop smiling!', label: 'joy' },
    { text: 'My dog passed away this morning. I don\'t know how to cope.', label: 'sadness' },
    { text: 'How dare they cancel the event without telling anyone! This is unacceptable.', label: 'anger' },
    { text: 'I have to give a speech in front of 500 people tomorrow and I\'m terrified.', label: 'fear' },
    { text: 'The meeting was fine I guess. Nothing really happened.', label: 'neutral' },
  ],
  baseline: { loss: 0.441, metric: 0.731 },
  final:    { loss: 0.231, metric: 0.921 },
  iterations: classificationIterations,
};

export const CUSTOMER_SUPPORT: TaskConfig = {
  keywords: ['customer', 'support', 'ticket', 'department', 'banking', 'intent', 'service'],
  taskLabel: 'Customer Intent Classification',
  datasets: [{ name: 'mteb/banking77', size: '13k' }],
  model: 'distilbert-base-uncased',
  evalMetric: 'accuracy', metricLabel: 'Val Accuracy',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { text: 'I need to reset my online banking password right away.', intent: 'reset_password' },
    { text: 'Can I see all the transactions from last month on my account?', intent: 'transaction_history' },
    { text: 'My card was declined at the supermarket even though I have funds.', intent: 'card_declined' },
    { text: 'How do I increase my daily ATM withdrawal limit?', intent: 'atm_limit' },
    { text: 'I think someone made an unauthorised transfer from my account.', intent: 'fraud_report' },
  ],
  baseline: { loss: 0.511, metric: 0.701 },
  final:    { loss: 0.271, metric: 0.911 },
  iterations: classificationIterations,
};

export const SPAM: TaskConfig = {
  keywords: ['spam', 'email', 'phishing', 'scam', 'junk', 'ham'],
  taskLabel: 'Spam Detection',
  datasets: [{ name: 'SetFit/enron_spam', size: '33k' }],
  model: 'distilbert-base-uncased',
  evalMetric: 'F1', metricLabel: 'F1 Score',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { subject: 'Project update for Q3', body: 'Hi team, please find the attached slides for our Q3 review.', label: 'ham' },
    { subject: 'YOU WON $1,000,000!!!', body: 'Click here NOW to claim your prize. Limited time offer!', label: 'spam' },
    { subject: 'Lunch tomorrow?', body: 'Hey, are you free for lunch at 12:30? Let me know!', label: 'ham' },
    { subject: 'Verify your account IMMEDIATELY', body: 'Your account will be suspended. Enter your password at http://bit.ly/fake', label: 'spam' },
    { subject: 'Invoice #4821 attached', body: 'Please find the invoice for last month\'s services. Payment due in 30 days.', label: 'ham' },
  ],
  baseline: { loss: 0.361, metric: 0.851 },
  final:    { loss: 0.181, metric: 0.971 },
  iterations: classificationIterations,
};

export const PARAPHRASE: TaskConfig = {
  keywords: ['paraphrase', 'duplicate', 'similar', 'sentence', 'semantic', 'equivalent', 'mrpc'],
  taskLabel: 'Paraphrase Detection',
  datasets: [{ name: 'SetFit/mrpc', size: '5.8k' }],
  model: 'bert-base-uncased',
  evalMetric: 'F1', metricLabel: 'F1 Score',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { sentence1: 'The stock market fell sharply on Friday.', sentence2: 'Shares dropped significantly at the end of the week.', label: 'paraphrase' },
    { sentence1: 'He said he would call back in an hour.', sentence2: 'She promised to return the message by morning.', label: 'not_paraphrase' },
    { sentence1: 'The company announced record profits this quarter.', sentence2: 'The firm reported its highest-ever earnings for the period.', label: 'paraphrase' },
    { sentence1: 'The train arrives at 9 AM.', sentence2: 'The flight departs at noon.', label: 'not_paraphrase' },
    { sentence1: 'Scientists discovered a new species of bird in the Amazon.', sentence2: 'Researchers found a previously unknown bird in the Amazon rainforest.', label: 'paraphrase' },
  ],
  baseline: { loss: 0.421, metric: 0.811 },
  final:    { loss: 0.231, metric: 0.901 },
  iterations: classificationIterations,
};

// ─────────────────────────────────────────────────────────────────────────────
// QUESTION ANSWERING
// ─────────────────────────────────────────────────────────────────────────────

const qaIterations: IterationTemplate[] = [
  { experiment: 'Increase max_seq_len 384→512 to capture longer contexts.', diff: '- max_seq_len: 384\n+ max_seq_len: 512', lossAfter: 0.781, metricAfter: 0.731, status: 'KEPT' },
  { experiment: 'Decrease learning_rate 3e-5→1.5e-5.', diff: '- learning_rate: 3e-5\n+ learning_rate: 1.5e-5', lossAfter: 0.701, metricAfter: 0.771, status: 'KEPT' },
  { experiment: 'Increase doc_stride 128→192 for better passage overlap.', diff: '- doc_stride: 128\n+ doc_stride: 192', lossAfter: 0.661, metricAfter: 0.801, status: 'KEPT' },
  { experiment: 'Increase n_best_size 20→30 for span selection.', diff: '- n_best_size: 20\n+ n_best_size: 30', lossAfter: 0.641, metricAfter: 0.821, status: 'KEPT' },
  { experiment: 'Decrease learning_rate 1.5e-5→3.0e-5 (revert test).', diff: '- learning_rate: 1.5e-5\n+ learning_rate: 3.0e-5', lossAfter: 0.698, metricAfter: 0.779, status: 'REVERTED' },
  { experiment: 'Add warmup_steps 0→200 to stabilise QA head training.', diff: '- warmup_steps: 0\n+ warmup_steps: 200', lossAfter: 0.601, metricAfter: 0.841, status: 'KEPT' },
  { experiment: 'Increase lora_rank 16→24 for richer QA representations.', diff: '- lora_rank: 16\n+ lora_rank: 24', lossAfter: 0.571, metricAfter: 0.861, status: 'KEPT' },
  { experiment: 'Decrease max_answer_length 30→20 to reduce false spans.', diff: '- max_answer_length: 30\n+ max_answer_length: 20', lossAfter: 0.591, metricAfter: 0.849, status: 'REVERTED' },
];

export const QA: TaskConfig = {
  keywords: ['question', 'answer', 'qa', 'squad', 'passage', 'paragraph', 'reading', 'comprehension'],
  taskLabel: 'Extractive Question Answering',
  datasets: [
    { name: 'rajpurkar/squad', size: '87.6k' },
    { name: 'rajpurkar/squad_v2', size: '130k' },
  ],
  model: 'bert-base-uncased',
  evalMetric: 'F1', metricLabel: 'Exact Match / F1',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value, key]',
  samples: [
    { context: 'The Amazon River is the largest river by discharge volume in the world, located in South America.', question: 'Where is the Amazon River located?', answer: 'South America' },
    { context: 'Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was born in Warsaw in 1867.', question: 'Where was Marie Curie born?', answer: 'Warsaw' },
    { context: 'The Python programming language was created by Guido van Rossum and first released in 1991.', question: 'Who created Python?', answer: 'Guido van Rossum' },
    { context: 'The Great Wall of China stretches over 13,000 miles and was built to protect against invasions from the north.', question: 'How long is the Great Wall of China?', answer: 'over 13,000 miles' },
    { context: 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy.', question: 'What do plants produce during photosynthesis?', answer: 'oxygen and energy' },
  ],
  baseline: { loss: 0.891, metric: 0.681 },
  final:    { loss: 0.541, metric: 0.871 },
  iterations: qaIterations,
};

// ─────────────────────────────────────────────────────────────────────────────
// NATURAL LANGUAGE INFERENCE
// ─────────────────────────────────────────────────────────────────────────────

export const NLI: TaskConfig = {
  keywords: ['entail', 'contradict', 'neutral', 'inference', 'hypothesis', 'premise', 'nli', 'snli', 'mnli'],
  taskLabel: 'Natural Language Inference',
  datasets: [
    { name: 'stanfordnlp/snli', size: '570k' },
    { name: 'SetFit/mnli', size: '393k' },
  ],
  model: 'bert-base-uncased',
  evalMetric: 'accuracy', metricLabel: 'Val Accuracy',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { premise: 'A soccer player is kicking a ball on a grass field.', hypothesis: 'A person is playing sports outside.', label: 'entailment' },
    { premise: 'The woman is reading a book in the library.', hypothesis: 'The woman is watching TV.', label: 'contradiction' },
    { premise: 'Two men are sitting at a table eating lunch.', hypothesis: 'The men are hungry.', label: 'neutral' },
    { premise: 'A child is running in the park.', hypothesis: 'A young person is moving quickly outdoors.', label: 'entailment' },
    { premise: 'The cat is sleeping on the couch.', hypothesis: 'The cat is chasing a mouse.', label: 'contradiction' },
  ],
  baseline: { loss: 0.521, metric: 0.761 },
  final:    { loss: 0.271, metric: 0.901 },
  iterations: [
    { experiment: 'Increase lora_rank 16→32 for 3-class NLI head.', diff: '- lora_rank: 16\n+ lora_rank: 32', lossAfter: 0.471, metricAfter: 0.801, status: 'KEPT' },
    { experiment: 'Decrease learning_rate 2e-5→1e-5.', diff: '- learning_rate: 2e-5\n+ learning_rate: 1e-5', lossAfter: 0.421, metricAfter: 0.831, status: 'KEPT' },
    { experiment: 'Increase max_seq_len 128→256 for long premise-hypothesis pairs.', diff: '- max_seq_len: 128\n+ max_seq_len: 256', lossAfter: 0.391, metricAfter: 0.851, status: 'KEPT' },
    { experiment: 'Add label_smoothing 0→0.05.', diff: '- label_smoothing: 0.0\n+ label_smoothing: 0.05', lossAfter: 0.411, metricAfter: 0.839, status: 'REVERTED' },
    { experiment: 'Increase warmup_steps 200→600.', diff: '- warmup_steps: 200\n+ warmup_steps: 600', lossAfter: 0.361, metricAfter: 0.869, status: 'KEPT' },
    { experiment: 'Decrease dropout 0.1→0.05.', diff: '- dropout: 0.1\n+ dropout: 0.05', lossAfter: 0.341, metricAfter: 0.881, status: 'KEPT' },
    { experiment: 'Increase batch_size 32→64.', diff: '- batch_size: 32\n+ batch_size: 64', lossAfter: 0.361, metricAfter: 0.869, status: 'REVERTED' },
    { experiment: 'Decrease learning_rate 1e-5→8e-6.', diff: '- learning_rate: 1e-5\n+ learning_rate: 8e-6', lossAfter: 0.271, metricAfter: 0.901, status: 'KEPT' },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// SUMMARISATION
// ─────────────────────────────────────────────────────────────────────────────

const summarisationIterations: IterationTemplate[] = [
  { experiment: 'Increase num_beams 4→8 for better beam search coverage.', diff: '- num_beams: 4\n+ num_beams: 8', lossAfter: 1.941, metricAfter: 0.341, status: 'KEPT' },
  { experiment: 'Tune length_penalty 1.0→0.8 to prefer shorter summaries.', diff: '- length_penalty: 1.0\n+ length_penalty: 0.8', lossAfter: 1.881, metricAfter: 0.361, status: 'KEPT' },
  { experiment: 'Decrease learning_rate 5e-5→2.5e-5.', diff: '- learning_rate: 5e-5\n+ learning_rate: 2.5e-5', lossAfter: 1.821, metricAfter: 0.381, status: 'KEPT' },
  { experiment: 'Increase max_target_length 64→128.', diff: '- max_target_length: 64\n+ max_target_length: 128', lossAfter: 1.861, metricAfter: 0.369, status: 'REVERTED' },
  { experiment: 'Add repetition_penalty 1.0→1.2 to reduce repeated phrases.', diff: '- repetition_penalty: 1.0\n+ repetition_penalty: 1.2', lossAfter: 1.761, metricAfter: 0.401, status: 'KEPT' },
  { experiment: 'Increase warmup_steps 500→1000.', diff: '- warmup_steps: 500\n+ warmup_steps: 1000', lossAfter: 1.721, metricAfter: 0.411, status: 'KEPT' },
  { experiment: 'Increase num_beams 8→12.', diff: '- num_beams: 8\n+ num_beams: 12', lossAfter: 1.741, metricAfter: 0.408, status: 'REVERTED' },
  { experiment: 'Decrease learning_rate 2.5e-5→2.0e-5.', diff: '- learning_rate: 2.5e-5\n+ learning_rate: 2.0e-5', lossAfter: 1.681, metricAfter: 0.421, status: 'KEPT' },
];

export const NEWS_SUMMARY: TaskConfig = {
  keywords: ['summarize', 'summarise', 'summary', 'news', 'article', 'abstract', 'condense', 'tldr', 'cnn'],
  taskLabel: 'News Summarisation',
  datasets: [{ name: 'abisee/cnn_dailymail', size: '287k' }],
  model: 't5-small',
  evalMetric: 'ROUGE-L', metricLabel: 'ROUGE-L',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 8, targetModules: '[q, v]',
  samples: [
    { article: 'The Federal Reserve raised interest rates by 0.25% on Wednesday, citing persistent inflation pressures despite recent cooling in consumer prices. Fed Chair Jerome Powell said further hikes remain on the table.', summary: 'Fed raises rates 0.25%, warns more hikes possible.' },
    { article: 'NASA\'s Artemis II crew was announced today, featuring four astronauts who will orbit the Moon in 2025 — the first crewed lunar mission since Apollo 17 in 1972.', summary: 'NASA names Artemis II crew for 2025 Moon orbit mission.' },
    { article: 'Tesla reported Q2 earnings that beat Wall Street estimates, driven by strong vehicle deliveries and cost reductions. Shares rose 6% in after-hours trading.', summary: 'Tesla beats Q2 estimates; shares jump 6% after hours.' },
    { article: 'A new study published in Nature found that regular exercise of at least 30 minutes per day reduces the risk of heart disease by up to 35% in adults over 50.', summary: 'Study: 30 min daily exercise cuts heart disease risk 35% for over-50s.' },
    { article: 'OpenAI released GPT-5 with significantly improved reasoning capabilities, supporting 256k-token context windows and multimodal inputs including images and audio.', summary: 'OpenAI launches GPT-5 with enhanced reasoning and 256k context.' },
  ],
  baseline: { loss: 2.11, metric: 0.301 },
  final:    { loss: 1.64, metric: 0.421 },
  iterations: summarisationIterations,
};

export const ARXIV_SUMMARY: TaskConfig = {
  keywords: ['scientific', 'science', 'paper', 'arxiv', 'abstract', 'research', 'academic'],
  taskLabel: 'Scientific Paper Summarisation',
  datasets: [{ name: 'ccdv/arxiv-summarization', size: '215k' }],
  model: 't5-small',
  evalMetric: 'ROUGE-L', metricLabel: 'ROUGE-L',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 8, targetModules: '[q, v]',
  samples: [
    { abstract: 'We propose a novel attention mechanism that scales linearly with sequence length by approximating the full attention matrix using low-rank decompositions. Experiments on GLUE and SuperGLUE show competitive performance with 3× speedup.', summary: 'Linear attention via low-rank approximation; 3× speedup on GLUE.' },
    { abstract: 'This paper investigates the effects of data augmentation strategies on low-resource neural machine translation. We show that back-translation combined with token-level noise yields significant BLEU gains.', summary: 'Back-translation + token noise improves low-resource NMT BLEU.' },
    { abstract: 'We introduce a reinforcement learning framework for protein folding that outperforms AlphaFold2 on orphan proteins by leveraging evolutionary co-variation signals as reward shaping.', summary: 'RL-based protein folding surpasses AlphaFold2 on orphan proteins.' },
    { abstract: 'Large language models exhibit emergent capabilities at scale, but the mechanisms remain poorly understood. We provide evidence that in-context learning arises from implicit Bayesian inference over latent task variables.', summary: 'In-context learning emerges from implicit Bayesian inference over tasks.' },
    { abstract: 'We present a unified benchmark for evaluating multimodal models across vision, language, and audio modalities. Results show current models excel on unimodal subtasks but underperform on cross-modal reasoning.', summary: 'Multimodal benchmark reveals gap in cross-modal reasoning for current models.' },
  ],
  baseline: { loss: 2.41, metric: 0.271 },
  final:    { loss: 1.82, metric: 0.391 },
  iterations: summarisationIterations,
};

// ─────────────────────────────────────────────────────────────────────────────
// CONVERSATIONAL / INSTRUCTION FOLLOWING
// ─────────────────────────────────────────────────────────────────────────────

export const INSTRUCTION_FOLLOWING: TaskConfig = {
  keywords: ['instruction', 'helpful', 'assistant', 'follow', 'chatbot', 'chat', 'rlhf', 'hh-rlhf', 'human', 'feedback'],
  taskLabel: 'Instruction Following (SFT)',
  datasets: [{ name: 'Anthropic/hh-rlhf', size: '160k' }],
  model: 'meta-llama/Llama-3.2-1B',
  evalMetric: 'reward_score', metricLabel: 'Reward Score',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 32, targetModules: '[q_proj, v_proj, k_proj]',
  samples: [
    { prompt: 'Explain quantum entanglement in simple terms.', chosen: 'Quantum entanglement is when two particles become linked so that measuring one instantly determines the state of the other, no matter how far apart they are.', rejected: 'It\'s a physics thing about particles.' },
    { prompt: 'Write a professional email declining a job offer.', chosen: 'Dear [Name], Thank you for the generous offer. After careful consideration, I have decided to pursue a different opportunity. I appreciate your time and hope we can connect in the future.', rejected: 'No thanks, found something better.' },
    { prompt: 'What are three benefits of regular exercise?', chosen: '1. Improved cardiovascular health. 2. Better mental health and reduced anxiety. 3. Increased muscle strength and bone density.', rejected: 'It makes you healthier.' },
    { prompt: 'Summarise the French Revolution in two sentences.', chosen: 'The French Revolution (1789–1799) was a period of radical political and social transformation that overthrew the monarchy and established a republic. It was driven by Enlightenment ideals and popular discontent over inequality and financial crisis.', rejected: 'France had a revolution and the king was killed.' },
    { prompt: 'How do I sort a list in Python?', chosen: 'You can sort a list in Python using the built-in `sorted()` function, which returns a new list, or the `.sort()` method, which sorts the list in place. For example: `sorted([3, 1, 2])` returns `[1, 2, 3]`.', rejected: 'Use sort().' },
  ],
  baseline: { loss: 1.89, metric: 0.621 },
  final:    { loss: 1.21, metric: 0.841 },
  iterations: [
    { experiment: 'Increase lora_rank 16→32 for broader instruction coverage.', diff: '- lora_rank: 16\n+ lora_rank: 32', lossAfter: 1.721, metricAfter: 0.671, status: 'KEPT' },
    { experiment: 'Tune learning_rate 2e-4→1e-4 to prevent catastrophic forgetting.', diff: '- learning_rate: 2e-4\n+ learning_rate: 1e-4', lossAfter: 1.631, metricAfter: 0.711, status: 'KEPT' },
    { experiment: 'Add KL divergence penalty weight 0→0.01.', diff: '- kl_penalty: 0.0\n+ kl_penalty: 0.01', lossAfter: 1.591, metricAfter: 0.731, status: 'KEPT' },
    { experiment: 'Increase max_seq_len 512→1024 for multi-turn conversations.', diff: '- max_seq_len: 512\n+ max_seq_len: 1024', lossAfter: 1.561, metricAfter: 0.751, status: 'KEPT' },
    { experiment: 'Increase lora_alpha 32→64.', diff: '- lora_alpha: 32\n+ lora_alpha: 64', lossAfter: 1.598, metricAfter: 0.739, status: 'REVERTED' },
    { experiment: 'Add gradient_checkpointing to stabilise long-context training.', diff: '- gradient_checkpointing: false\n+ gradient_checkpointing: true', lossAfter: 1.511, metricAfter: 0.781, status: 'KEPT' },
    { experiment: 'Decrease learning_rate 1e-4→8e-5.', diff: '- learning_rate: 1e-4\n+ learning_rate: 8e-5', lossAfter: 1.481, metricAfter: 0.801, status: 'KEPT' },
    { experiment: 'Increase kl_penalty 0.01→0.05 (too strong).', diff: '- kl_penalty: 0.01\n+ kl_penalty: 0.05', lossAfter: 1.541, metricAfter: 0.769, status: 'REVERTED' },
  ],
};

export const CHATBOT: TaskConfig = {
  keywords: ['chatbot', 'oasst', 'conversation', 'dialogue', 'open', 'assistant', 'conversation'],
  taskLabel: 'Conversational Chatbot (SFT)',
  datasets: [{ name: 'OpenAssistant/oasst1', size: '88k' }],
  model: 'meta-llama/Llama-3.2-1B',
  evalMetric: 'perplexity', metricLabel: 'Val Perplexity',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 32, targetModules: '[q_proj, v_proj]',
  samples: [
    { role_user: 'What\'s the capital of Australia?', role_assistant: 'The capital of Australia is Canberra, not Sydney as many people assume.' },
    { role_user: 'Can you help me write a haiku about autumn?', role_assistant: 'Leaves turn gold and red / Crisp wind carries them away / Autumn fades to frost.' },
    { role_user: 'What is the difference between RAM and ROM?', role_assistant: 'RAM (Random Access Memory) is temporary storage used while your computer runs programs; it resets when powered off. ROM (Read-Only Memory) is permanent storage that holds firmware and cannot be easily modified.' },
    { role_user: 'Give me a healthy breakfast idea.', role_assistant: 'Try overnight oats made with rolled oats, Greek yogurt, chia seeds, and fresh berries. It\'s high in protein and fibre, takes 5 minutes to prep, and is ready to eat in the morning.' },
    { role_user: 'How do I reverse a string in JavaScript?', role_assistant: 'You can reverse a string in JavaScript with: `str.split(\'\').reverse().join(\'\')`. This splits the string into characters, reverses the array, then joins them back.' },
  ],
  baseline: { loss: 2.01, metric: 0.611 },
  final:    { loss: 1.38, metric: 0.811 },
  iterations: INSTRUCTION_FOLLOWING.iterations,
};

// ─────────────────────────────────────────────────────────────────────────────
// TOKEN CLASSIFICATION / NER
// ─────────────────────────────────────────────────────────────────────────────

export const NER: TaskConfig = {
  keywords: ['named', 'entity', 'ner', 'person', 'place', 'organization', 'token', 'tag', 'pos', 'part', 'speech', 'conll', 'extract'],
  taskLabel: 'Named Entity Recognition',
  datasets: [{ name: 'conll2003', size: '20k' }],
  model: 'bert-base-uncased',
  evalMetric: 'F1', metricLabel: 'Entity F1',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 16, targetModules: '[query, value]',
  samples: [
    { tokens: 'Apple Inc. acquired Beats Electronics in 2014 for $3 billion.', entities: 'Apple Inc. [ORG], Beats Electronics [ORG], 2014 [DATE]' },
    { tokens: 'Elon Musk visited Berlin last Tuesday to meet with German Chancellor Olaf Scholz.', entities: 'Elon Musk [PER], Berlin [LOC], Olaf Scholz [PER]' },
    { tokens: 'The World Health Organization declared COVID-19 a pandemic on March 11, 2020.', entities: 'World Health Organization [ORG], COVID-19 [MISC], March 11, 2020 [DATE]' },
    { tokens: 'Amazon\'s headquarters are in Seattle, Washington, United States.', entities: 'Amazon [ORG], Seattle [LOC], Washington [LOC], United States [LOC]' },
    { tokens: 'Serena Williams won her 23rd Grand Slam title at the 2017 Australian Open.', entities: 'Serena Williams [PER], 2017 Australian Open [MISC]' },
  ],
  baseline: { loss: 0.291, metric: 0.831 },
  final:    { loss: 0.148, metric: 0.921 },
  iterations: [
    { experiment: 'Decrease learning_rate 3e-5→1e-5 for stable token-level CRF.', diff: '- learning_rate: 3e-5\n+ learning_rate: 1e-5', lossAfter: 0.251, metricAfter: 0.861, status: 'KEPT' },
    { experiment: 'Add class_weights to handle O-label dominance.', diff: '- class_weights: null\n+ class_weights: balanced', lossAfter: 0.228, metricAfter: 0.881, status: 'KEPT' },
    { experiment: 'Increase max_seq_len 128→256 for long documents.', diff: '- max_seq_len: 128\n+ max_seq_len: 256', lossAfter: 0.218, metricAfter: 0.891, status: 'KEPT' },
    { experiment: 'Increase dropout 0.1→0.2 (too aggressive).', diff: '- dropout: 0.1\n+ dropout: 0.2', lossAfter: 0.261, metricAfter: 0.851, status: 'REVERTED' },
    { experiment: 'Tune warmup_steps 0→300.', diff: '- warmup_steps: 0\n+ warmup_steps: 300', lossAfter: 0.198, metricAfter: 0.901, status: 'KEPT' },
    { experiment: 'Increase lora_rank 16→24.', diff: '- lora_rank: 16\n+ lora_rank: 24', lossAfter: 0.181, metricAfter: 0.911, status: 'KEPT' },
    { experiment: 'Add label_smoothing 0→0.05.', diff: '- label_smoothing: 0.0\n+ label_smoothing: 0.05', lossAfter: 0.188, metricAfter: 0.906, status: 'REVERTED' },
    { experiment: 'Decrease learning_rate 1e-5→8e-6.', diff: '- learning_rate: 1e-5\n+ learning_rate: 8e-6', lossAfter: 0.148, metricAfter: 0.921, status: 'KEPT' },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// TRANSLATION
// ─────────────────────────────────────────────────────────────────────────────

const translationIterations: IterationTemplate[] = [
  { experiment: 'Increase num_beams 4→6 for better translation coverage.', diff: '- num_beams: 4\n+ num_beams: 6', lossAfter: 2.201, metricAfter: 0.261, status: 'KEPT' },
  { experiment: 'Tune length_penalty 1.0→0.9.', diff: '- length_penalty: 1.0\n+ length_penalty: 0.9', lossAfter: 2.101, metricAfter: 0.281, status: 'KEPT' },
  { experiment: 'Decrease learning_rate 5e-5→2e-5.', diff: '- learning_rate: 5e-5\n+ learning_rate: 2e-5', lossAfter: 1.981, metricAfter: 0.301, status: 'KEPT' },
  { experiment: 'Increase num_beams 6→10 (diminishing returns).', diff: '- num_beams: 6\n+ num_beams: 10', lossAfter: 2.021, metricAfter: 0.295, status: 'REVERTED' },
  { experiment: 'Add no_repeat_ngram_size 0→3.', diff: '- no_repeat_ngram_size: 0\n+ no_repeat_ngram_size: 3', lossAfter: 1.921, metricAfter: 0.311, status: 'KEPT' },
  { experiment: 'Increase warmup_steps 500→1000.', diff: '- warmup_steps: 500\n+ warmup_steps: 1000', lossAfter: 1.861, metricAfter: 0.321, status: 'KEPT' },
  { experiment: 'Adjust beam_alpha 0.6→0.8.', diff: '- beam_alpha: 0.6\n+ beam_alpha: 0.8', lossAfter: 1.901, metricAfter: 0.315, status: 'REVERTED' },
  { experiment: 'Decrease learning_rate 2e-5→1.5e-5.', diff: '- learning_rate: 2e-5\n+ learning_rate: 1.5e-5', lossAfter: 1.811, metricAfter: 0.331, status: 'KEPT' },
];

export const TRANSLATION_FR: TaskConfig = {
  keywords: ['translate', 'translation', 'french', 'français', 'english', 'opus', 'fr'],
  taskLabel: 'English → French Translation',
  datasets: [{ name: 'Helsinki-NLP/opus-100', size: '1M' }],
  model: 'Helsinki-NLP/opus-mt-en-fr',
  evalMetric: 'BLEU', metricLabel: 'BLEU Score',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 8, targetModules: '[q, v]',
  samples: [
    { source: 'The weather is beautiful today.', target: 'Le temps est magnifique aujourd\'hui.' },
    { source: 'Can you please pass the salt?', target: 'Pouvez-vous passer le sel, s\'il vous plaît?' },
    { source: 'I would like to book a table for two at 7 PM.', target: 'Je voudrais réserver une table pour deux personnes à 19 heures.' },
    { source: 'The train to Paris departs in ten minutes.', target: 'Le train pour Paris part dans dix minutes.' },
    { source: 'She has been studying French for three years.', target: 'Elle étudie le français depuis trois ans.' },
  ],
  baseline: { loss: 2.41, metric: 0.221 },
  final:    { loss: 1.78, metric: 0.331 },
  iterations: translationIterations,
};

export const TRANSLATION_DE: TaskConfig = {
  keywords: ['translate', 'translation', 'german', 'deutsch', 'wmt', 'de', 'english'],
  taskLabel: 'English → German Translation',
  datasets: [{ name: 'wmt/wmt14', size: '4.5M' }],
  model: 't5-small',
  evalMetric: 'BLEU', metricLabel: 'BLEU Score',
  strategy: 'fine-tune', trainingType: 'SFT',
  loraRank: 8, targetModules: '[q, v]',
  samples: [
    { source: 'Good morning, how are you?', target: 'Guten Morgen, wie geht es Ihnen?' },
    { source: 'The library closes at nine o\'clock tonight.', target: 'Die Bibliothek schließt heute Abend um neun Uhr.' },
    { source: 'I am looking for the train station.', target: 'Ich suche den Bahnhof.' },
    { source: 'This product is made in Germany.', target: 'Dieses Produkt ist in Deutschland hergestellt.' },
    { source: 'She won the championship for the third consecutive year.', target: 'Sie gewann die Meisterschaft zum dritten Mal in Folge.' },
  ],
  baseline: { loss: 2.61, metric: 0.201 },
  final:    { loss: 1.91, metric: 0.311 },
  iterations: translationIterations,
};

// ─────────────────────────────────────────────────────────────────────────────
// Default fallback
// ─────────────────────────────────────────────────────────────────────────────

export const DEFAULT_CONFIG: TaskConfig = MOVIE_SENTIMENT;

export const ALL_CONFIGS: TaskConfig[] = [
  MOVIE_SENTIMENT, HATE_SPEECH, NEWS_CLASSIFICATION, EMOTION,
  CUSTOMER_SUPPORT, SPAM, PARAPHRASE,
  QA,
  NLI,
  NEWS_SUMMARY, ARXIV_SUMMARY,
  INSTRUCTION_FOLLOWING, CHATBOT,
  NER,
  TRANSLATION_FR, TRANSLATION_DE,
];
