import { NextRequest, NextResponse } from 'next/server';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export async function POST(request: NextRequest) {
  try {
    const { message, patientId } = await request.json();

    if (!patientId) {
      return NextResponse.json(
        { error: 'Patient ID is required' },
        { status: 400 }
      );
    }

    // Call the Python backend's AI agent endpoint with full RAG context
    const response = await fetch(`${API_BASE}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patient_id: Number(patientId),
        message,
        include_context: true,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Backend unavailable' }));
      throw new Error(error.detail || `Backend error: ${response.status}`);
    }

    const data = await response.json();

    // Stream the response word by word for better UX
    const encoder = new TextEncoder();
    const words = data.response.split(' ');

    const stream = new ReadableStream({
      async start(controller) {
        for (let i = 0; i < words.length; i++) {
          const word = i === words.length - 1 ? words[i] : words[i] + ' ';
          controller.enqueue(encoder.encode(word));
          // Small delay between words for streaming effect
          await new Promise(resolve => setTimeout(resolve, 30));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain',
        'Transfer-Encoding': 'chunked',
      },
    });
  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to connect to AI backend',
        details: 'Make sure the Python backend is running on port 8080'
      },
      { status: 500 }
    );
  }
}
