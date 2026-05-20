'use client';

import * as React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/Button';
import { usePatientStore } from '@/stores/patient';
import { Send, User, Bot, AlertTriangle } from 'lucide-react';
import type { ChatMessage } from '@/lib/types';

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className }: ChatInterfaceProps) {
  const { chatMessages, addChatMessage, currentGlucose, patient } = usePatientStore();
  const [input, setInput] = React.useState('');
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [streamingText, setStreamingText] = React.useState('');
  const messagesEndRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [chatMessages, streamingText]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message
    addChatMessage({
      id: Date.now().toString(),
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
    });

    setIsStreaming(true);
    setStreamingText('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          patientId: patient?.id,
          context: {
            currentGlucose,
            patientName: patient?.name,
            diabetesType: patient?.diabetes_type,
          },
        }),
      });

      if (!response.ok) throw new Error('Chat failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          fullText += chunk;
          setStreamingText(fullText);
        }
      }

      // Add assistant message
      addChatMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: fullText || 'I apologize, but I could not generate a response. Please try again.',
        timestamp: new Date(),
      });
    } catch (error) {
      console.error('Chat error:', error);
      addChatMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error. Please make sure the AI service is running and try again.',
        timestamp: new Date(),
      });
    }

    setIsStreaming(false);
    setStreamingText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatMessages.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
            <Bot className="h-12 w-12 mb-4 opacity-50" />
            <h3 className="font-medium text-foreground mb-2">AI Health Assistant</h3>
            <p className="text-sm max-w-md">
              Ask me anything about diabetes management, your glucose predictions,
              or get personalized recommendations based on your health data.
            </p>
            {!currentGlucose && (
              <div className="flex items-center gap-2 mt-4 text-warning">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm">Enter your glucose level first for personalized advice</span>
              </div>
            )}
          </div>
        )}

        <AnimatePresence mode="popLayout">
          {chatMessages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>

        {/* Streaming message */}
        {isStreaming && streamingText && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
              <Bot className="h-4 w-4" />
            </div>
            <div className="bg-card border border-border rounded-2xl rounded-tl-none px-4 py-3 max-w-[80%]">
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {streamingText}
                </ReactMarkdown>
              </div>
              <span className="inline-block w-2 h-4 bg-foreground animate-pulse ml-1" />
            </div>
          </motion.div>
        )}

        {/* Loading indicator */}
        {isStreaming && !streamingText && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
              <Bot className="h-4 w-4" />
            </div>
            <div className="bg-card border border-border rounded-2xl rounded-tl-none px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce" />
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:0.1s]" />
                <span className="w-2 h-2 rounded-full bg-muted-foreground animate-bounce [animation-delay:0.2s]" />
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border p-4">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your health..."
            disabled={isStreaming}
            className={cn(
              'flex-1 h-12 px-4 rounded-xl bg-card border border-border',
              'focus:outline-none focus:ring-2 focus:ring-ring',
              'placeholder:text-muted-foreground',
              'disabled:opacity-50'
            )}
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            size="icon"
            className="h-12 w-12"
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className={cn('flex gap-3', isUser && 'flex-row-reverse')}
    >
      <div
        className={cn(
          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          isUser ? 'bg-foreground text-background' : 'bg-accent'
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={cn(
          'rounded-2xl px-4 py-3 max-w-[80%]',
          isUser
            ? 'bg-foreground text-background rounded-tr-none'
            : 'bg-card border border-border rounded-tl-none'
        )}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                h2: ({ children }) => <h2 className="text-base font-semibold mb-2">{children}</h2>,
                h3: ({ children }) => <h3 className="text-sm font-semibold mb-1">{children}</h3>,
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                li: ({ children }) => <li className="text-sm">{children}</li>,
                code: ({ children, className }) => {
                  const isInline = !className;
                  return isInline ? (
                    <code className="bg-accent px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
                  ) : (
                    <code className="block bg-accent p-3 rounded-lg text-xs font-mono overflow-x-auto mb-2">{children}</code>
                  );
                },
                pre: ({ children }) => <pre className="bg-accent p-3 rounded-lg overflow-x-auto mb-2">{children}</pre>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-muted-foreground pl-3 italic text-muted-foreground mb-2">
                    {children}
                  </blockquote>
                ),
                a: ({ href, children }) => (
                  <a href={href} className="text-primary underline hover:no-underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
                table: ({ children }) => (
                  <div className="overflow-x-auto mb-2">
                    <table className="min-w-full text-sm border border-border rounded">{children}</table>
                  </div>
                ),
                th: ({ children }) => <th className="px-3 py-2 bg-accent text-left font-medium border-b border-border">{children}</th>,
                td: ({ children }) => <td className="px-3 py-2 border-b border-border">{children}</td>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </motion.div>
  );
}
