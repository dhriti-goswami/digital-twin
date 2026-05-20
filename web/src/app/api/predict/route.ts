import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.API_URL || 'http://localhost:8080/api/v1';

export async function POST(request: NextRequest) {
  // Parse body once — cannot call request.json() more than once on the same request
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({ detail: `Backend HTTP ${response.status}` }));
      return NextResponse.json(
        { error: detail.detail || `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Predict API error:', error);
    return NextResponse.json(
      { error: 'ML backend is unavailable. Ensure the Python server is running on port 8080.' },
      { status: 503 }
    );
  }
}
