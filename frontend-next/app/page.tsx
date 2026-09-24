"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { TextStreamChatTransport } from "ai";
import type { UIMessage } from "ai";
import { Trash2, Download, ExternalLink, ChevronDown, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { Components } from "react-markdown";

const API_URL = "http://localhost:8000/query";
const PHOENIX_URL = "http://localhost:6006";

const APP_TITLE = "Enterprise Knowledge Assistant";
const APP_SUBTITLE = "Ask questions about internal company data, Slack threads, and Confluence runbooks.";

// Real, domain-relevant examples (matches frontend/app.py, the Streamlit
// app this page replaces), not generic placeholders.
const SUGGESTED_QUESTIONS = [
  "What is the deployment process?",
  "What are our data retention requirements?",
  "Summarize the latest incident postmortem.",
];

type Citation = {
  document_id: string;
  chunk_id: string;
  source_name: string;
  score: number;
  department: string | null;
  role: string | null;
  content: string;
};

// Must match backend/api.py's CITATIONS_STREAM_MARKER_PREFIX/SUFFIX exactly.
const CITATIONS_MARKER_PREFIX = "\n\n<!--CITATIONS:";
const CITATIONS_MARKER_SUFFIX = "-->";

type TextPart = Extract<UIMessage["parts"][number], { type: "text" }>;

function getMessageText(parts: UIMessage["parts"]): string {
  return parts
    .filter((part): part is TextPart => part.type === "text")
    .map((part) => part.text)
    .join("");
}

/**
 * Split a streamed answer from its trailing citations sentinel.
 *
 * Slices at the marker's *start* regardless of whether the closing `-->`
 * has arrived yet, so a citations payload that is still streaming in never
 * flashes as visible answer text.
 */
function splitAnswerAndCitations(fullText: string): { answer: string; citations: Citation[] } {
  const markerStart = fullText.indexOf(CITATIONS_MARKER_PREFIX);
  if (markerStart === -1) {
    return { answer: fullText, citations: [] };
  }
  const answer = fullText.slice(0, markerStart);
  const payloadStart = markerStart + CITATIONS_MARKER_PREFIX.length;
  const payloadEnd = fullText.indexOf(CITATIONS_MARKER_SUFFIX, payloadStart);
  if (payloadEnd === -1) {
    return { answer, citations: [] };
  }
  const base64Payload = fullText.slice(payloadStart, payloadEnd);
  try {
    const citations = JSON.parse(atob(base64Payload)) as Citation[];
    return { answer, citations };
  } catch {
    return { answer, citations: [] };
  }
}

/** Guardrail 400s surface here as `error.message` = the raw `{"detail": "..."}` JSON body text. */
function parseErrorMessage(rawMessage: string): string {
  try {
    const parsed = JSON.parse(rawMessage) as { detail?: string };
    return parsed.detail ?? rawMessage;
  } catch {
    return rawMessage;
  }
}

function formatConversationAsMarkdown(messages: UIMessage[]): string {
  if (messages.length === 0) {
    return "# Enterprise RAG Assistant Conversation\n\n_No messages yet._\n";
  }
  const lines = ["# Enterprise RAG Assistant Conversation", ""];
  for (const message of messages) {
    const speaker = message.role === "user" ? "You" : "Assistant";
    const { answer, citations } = splitAnswerAndCitations(getMessageText(message.parts));
    lines.push(`**${speaker}:** ${answer}`, "");
    if (citations.length > 0) {
      lines.push("Sources:");
      for (const citation of citations) {
        lines.push(`- ${citation.source_name} (score: ${citation.score.toFixed(4)})`);
      }
      lines.push("");
    }
  }
  return lines.join("\n");
}

// react-markdown v10 no longer passes an `inline` prop to the `code`
// component (removed upstream) - fenced code blocks are distinguished from
// inline code by the `language-xxx` className mdast-util-to-hast attaches
// to fenced blocks with a specified language; inline code gets none.
const markdownComponents: Components = {
  code(props) {
    const { className, children, ...rest } = props;
    const match = /language-(\w+)/.exec(className ?? "");
    if (match) {
      return (
        <SyntaxHighlighter language={match[1]} style={vscDarkPlus} PreTag="div" customStyle={{ borderRadius: 8, fontSize: "0.85em" }}>
          {String(children).replace(/\n$/, "")}
        </SyntaxHighlighter>
      );
    }
    return (
      <code className="rounded bg-gray-100 px-1 py-0.5 text-sm text-gray-800" {...rest}>
        {children}
      </code>
    );
  },
  a(props) {
    return <a {...props} target="_blank" rel="noreferrer" className="text-blue-600 underline" />;
  },
  ul(props) {
    return <ul {...props} className="list-disc space-y-1 pl-5" />;
  },
  ol(props) {
    return <ol {...props} className="list-decimal space-y-1 pl-5" />;
  },
  p(props) {
    return <p {...props} className="mb-2 last:mb-0" />;
  },
  table(props) {
    return (
      <div className="my-2 overflow-x-auto">
        <table {...props} className="border-collapse text-sm" />
      </div>
    );
  },
  th(props) {
    return <th {...props} className="border border-gray-200 bg-gray-50 px-2 py-1 text-left font-semibold" />;
  },
  td(props) {
    return <td {...props} className="border border-gray-200 px-2 py-1" />;
  },
};

function AssistantMarkdown({ text }: { text: string }) {
  return (
    <div className="prose-sm max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

function CitationsDrawer({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-1 text-sm font-medium text-gray-600 hover:text-gray-900"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        View Retrieved Context ({citations.length})
      </button>
      {open && (
        <div className="mt-2 space-y-3">
          {citations.map((citation, index) => (
            <div key={citation.chunk_id} className="rounded-lg bg-gray-50 p-3 text-sm">
              <div className="mb-1 font-medium text-gray-800">
                Document {index + 1} — {citation.source_name}
              </div>
              <div className="mb-1 text-xs text-gray-500">Relevance score: {citation.score.toFixed(4)}</div>
              <div className="whitespace-pre-wrap text-gray-600">{citation.content || "No content available."}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [input, setInput] = useState("");
  const { messages, sendMessage, status, error, setMessages } = useChat({
    transport: new TextStreamChatTransport({
      api: API_URL,
      // Reshapes the SDK's default { messages: UIMessage[], id, trigger, ... }
      // request body into the backend's existing { "question": "..." } contract.
      prepareSendMessagesRequest: ({ messages: uiMessages }) => {
        const lastMessage = uiMessages[uiMessages.length - 1];
        const question = getMessageText(lastMessage.parts);
        return { body: { question } };
      },
    }),
  });

  const isBusy = status === "submitted" || status === "streaming";

  function handleSend(text: string) {
    const trimmed = text.trim();
    if (!trimmed || isBusy) return;
    sendMessage({ text: trimmed });
    setInput("");
  }

  function handleClear() {
    setMessages([]);
  }

  function handleDownload() {
    const blob = new Blob([formatConversationAsMarkdown(messages)], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "conversation.md";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-xl">🏢</span>
            <span className="font-semibold text-gray-900">{APP_TITLE}</span>
          </div>
          <div className="flex items-center gap-1">
            <a
              href={PHOENIX_URL}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm text-gray-600 transition hover:bg-gray-100"
            >
              Telemetry <ExternalLink size={14} />
            </a>
            <button
              onClick={handleDownload}
              disabled={messages.length === 0}
              className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm text-gray-600 transition hover:bg-gray-100 disabled:opacity-40"
            >
              <Download size={14} /> Download
            </button>
            <button
              onClick={handleClear}
              disabled={messages.length === 0}
              className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm text-gray-600 transition hover:bg-gray-100 disabled:opacity-40"
            >
              <Trash2 size={14} /> Clear Chat
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="mb-3 text-4xl">🏢</div>
            <h1 className="mb-2 text-2xl font-bold text-gray-900">{APP_TITLE}</h1>
            <p className="max-w-md text-gray-500">{APP_SUBTITLE}</p>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((message) => {
              const { answer, citations } = splitAnswerAndCitations(getMessageText(message.parts));
              if (message.role === "user") {
                return (
                  <div key={message.id} className="flex justify-end">
                    <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-blue-600 px-4 py-2.5 text-white shadow-sm">
                      {answer}
                    </div>
                  </div>
                );
              }
              return (
                <div key={message.id} className="flex justify-start">
                  <div className="max-w-[85%] rounded-2xl bg-white px-4 py-3 text-gray-900 shadow-md">
                    <AssistantMarkdown text={answer} />
                    {citations.length > 0 && <CitationsDrawer citations={citations} />}
                  </div>
                </div>
              );
            })}
            {status === "error" && error && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-2xl border border-amber-200 bg-amber-50 px-4 py-2.5 text-amber-800 shadow-sm">
                  {parseErrorMessage(error.message)}
                </div>
              </div>
            )}
            {status === "submitted" && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-white px-4 py-3 text-gray-400 shadow-md">Searching enterprise knowledge base and reranking…</div>
              </div>
            )}
          </div>
        )}
      </main>

      <div className="sticky bottom-0 border-t border-gray-200 bg-white/95 backdrop-blur">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <div className="mb-3 flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map((question) => (
              <button
                key={question}
                onClick={() => handleSend(question)}
                disabled={isBusy}
                className="rounded-full border border-gray-300 px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-100 disabled:opacity-40"
              >
                {question}
              </button>
            ))}
          </div>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              handleSend(input);
            }}
            className="flex gap-2"
          >
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about internal data..."
              disabled={isBusy}
              className="flex-1 rounded-full border border-gray-300 px-4 py-2.5 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-60"
            />
            <button
              type="submit"
              disabled={isBusy || !input.trim()}
              className="rounded-full bg-blue-600 px-5 py-2.5 font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-40"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
