<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Live Transcription</title>

        <link rel="icon" href="/favicon.ico" sizes="any">
        <link rel="preconnect" href="https://fonts.bunny.net">
    <link href="https://fonts.bunny.net/css?family=jetbrains-mono:400,500,600,700&family=plus-jakarta-sans:400,500,600,700" rel="stylesheet" />
    
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        midnight: {
                            900: '#0a0a0f',
                            800: '#12121a',
                            700: '#1a1a24',
                            600: '#22222e',
                        },
                        accent: {
                            cyan: '#06b6d4',
                            purple: '#a855f7',
                            pink: '#ec4899',
                            orange: '#f97316',
                            green: '#22c55e',
                        }
                    }
                }
            }
        }
    </script>
        <style>
        .gradient-border {
            background: linear-gradient(135deg, #06b6d4, #a855f7, #ec4899);
            padding: 1px;
            border-radius: 1rem;
        }
        .gradient-border-inner {
            background: #12121a;
            border-radius: calc(1rem - 1px);
        }
        .speaker-0 { --speaker-color: #06b6d4; }
        .speaker-1 { --speaker-color: #a855f7; }
        .speaker-2 { --speaker-color: #ec4899; }
        .speaker-3 { --speaker-color: #f97316; }
        .speaker-4 { --speaker-color: #22c55e; }
        .speaker-5 { --speaker-color: #eab308; }
        .speaker-6 { --speaker-color: #ef4444; }
        .speaker-7 { --speaker-color: #3b82f6; }
        
        .pulse-ring {
            animation: pulse-ring 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse-ring {
            0%, 100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.1); opacity: 0.1; }
        }
        
        .waveform-bar {
            animation: waveform 0.5s ease-in-out infinite alternate;
        }
        .waveform-bar:nth-child(1) { animation-delay: 0s; }
        .waveform-bar:nth-child(2) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(3) { animation-delay: 0.2s; }
        .waveform-bar:nth-child(4) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(5) { animation-delay: 0s; }
        
        @keyframes waveform {
            from { height: 8px; }
            to { height: 24px; }
        }
        
        .transcript-enter {
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #22222e; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #2a2a38; }
        </style>
    </head>
<body class="bg-midnight-900 text-white min-h-screen font-sans antialiased">
    <div class="min-h-screen flex flex-col">
        <!-- Header -->
        <header class="border-b border-white/5 backdrop-blur-xl bg-midnight-900/80 sticky top-0 z-50">
            <div class="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                        </svg>
                    </div>
                    <div>
                        <h1 class="font-semibold text-lg">Live Transcription</h1>
                        <p class="text-xs text-white/40">Real-time speech-to-text with speaker identification</p>
                    </div>
                </div>
                
                <!-- Status indicator -->
                <div id="status-badge" class="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 text-sm">
                    <span id="status-dot" class="w-2 h-2 rounded-full bg-white/20"></span>
                    <span id="status-text" class="text-white/60">Disconnected</span>
                </div>
            </div>
        </header>

        <!-- Main content -->
        <main class="flex-1 flex flex-col max-w-5xl mx-auto w-full px-6 py-8">
            <!-- Transcript area -->
            <div class="flex-1 mb-6">
                <div class="gradient-border h-full">
                    <div class="gradient-border-inner h-full p-6">
                        <div id="transcript-container" class="h-[calc(100vh-340px)] min-h-[300px] overflow-y-auto space-y-4">
                            <!-- Empty state -->
                            <div id="empty-state" class="h-full flex flex-col items-center justify-center text-center">
                                <div class="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-purple-600/20 flex items-center justify-center mb-6">
                                    <svg class="w-10 h-10 text-white/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                                    </svg>
                                </div>
                                <h3 class="text-xl font-semibold text-white/80 mb-2">Ready to transcribe</h3>
                                <p class="text-white/40 max-w-md">Click the button below to start recording. Your speech will be transcribed in real-time with automatic speaker identification.</p>
                            </div>
                            
                            <!-- Transcripts will be inserted here -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- Controls -->
            <div class="flex flex-col items-center gap-6">
                <!-- Buttons row -->
                <div class="flex items-center gap-6">
                    <!-- Upload button -->
                    <button 
                        id="upload-btn" 
                        class="group relative"
                        onclick="document.getElementById('audio-file-input').click()"
                    >
                        <div class="relative w-16 h-16 rounded-full bg-gradient-to-br from-orange-500 to-pink-600 flex items-center justify-center shadow-lg shadow-orange-500/25 transition-all duration-300 group-hover:scale-105 group-hover:shadow-xl group-hover:shadow-orange-500/30">
                            <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                        </div>
                    </button>
                    <input type="file" id="audio-file-input" accept="audio/*" class="hidden" onchange="handleFileUpload(event)">
                    
                    <!-- Recording button -->
                    <button 
                        id="record-btn" 
                        class="group relative"
                        onclick="toggleRecording()"
                    >
                        <!-- Pulse rings (visible when recording) -->
                        <div id="pulse-rings" class="absolute inset-0 hidden">
                            <div class="absolute inset-0 rounded-full bg-red-500/20 pulse-ring"></div>
                            <div class="absolute inset-2 rounded-full bg-red-500/20 pulse-ring" style="animation-delay: 0.5s"></div>
                        </div>
                        
                        <!-- Button -->
                        <div id="record-btn-inner" class="relative w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/25 transition-all duration-300 group-hover:scale-105 group-hover:shadow-xl group-hover:shadow-cyan-500/30">
                            <!-- Mic icon (not recording) -->
                            <svg id="mic-icon" class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                            </svg>

                            <!-- Waveform (recording) -->
                            <div id="waveform" class="hidden flex items-center gap-1">
                                <div class="waveform-bar w-1 bg-white rounded-full"></div>
                                <div class="waveform-bar w-1 bg-white rounded-full"></div>
                                <div class="waveform-bar w-1 bg-white rounded-full"></div>
                                <div class="waveform-bar w-1 bg-white rounded-full"></div>
                                <div class="waveform-bar w-1 bg-white rounded-full"></div>
                            </div>
                        </div>
                    </button>
                </div>
                
                <p id="record-text" class="text-white/40 text-sm">Click mic to record • Click cloud to upload audio file</p>
                
                <!-- Upload progress -->
                <div id="upload-progress" class="hidden w-full max-w-md">
                    <div class="flex items-center justify-between mb-2">
                        <span id="upload-filename" class="text-sm text-white/60 truncate"></span>
                        <span id="upload-percent" class="text-sm text-white/40 font-mono">0%</span>
                    </div>
                    <div class="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div id="upload-bar" class="h-full bg-gradient-to-r from-orange-500 to-pink-500 rounded-full transition-all duration-300" style="width: 0%"></div>
                    </div>
                </div>
                
                <!-- Session info -->
                <div id="session-info" class="hidden flex items-center gap-4 text-xs text-white/30">
                    <span>Session: <span id="session-id" class="font-mono text-white/50">--</span></span>
                    <span>•</span>
                    <span>Speakers: <span id="speaker-count" class="text-white/50">0</span></span>
                    <span>•</span>
                    <span>Duration: <span id="duration" class="font-mono text-white/50">00:00</span></span>
                </div>
            </div>
            </main>

        <!-- Footer -->
        <footer class="border-t border-white/5 py-4">
            <div class="max-w-5xl mx-auto px-6 flex items-center justify-between text-xs text-white/30">
                <span>Powered by Whisper AI + Pyannote</span>
                <span>16kHz • 16-bit PCM • Mono</span>
            </div>
        </footer>
        </div>

    <script>
        // Configuration
        const WEBSOCKET_URL = 'ws://localhost:8765';
        const SAMPLE_RATE = 16000;
        
        // State
        let isRecording = false;
        let websocket = null;
        let mediaRecorder = null;
        let audioContext = null;
        let audioWorklet = null;
        let sessionId = null;
        let startTime = null;
        let durationInterval = null;
        let speakers = new Set();
        
        // Speaker colors
        const speakerColors = [
            'from-cyan-500 to-cyan-400',
            'from-purple-500 to-purple-400',
            'from-pink-500 to-pink-400',
            'from-orange-500 to-orange-400',
            'from-green-500 to-green-400',
            'from-yellow-500 to-yellow-400',
            'from-red-500 to-red-400',
            'from-blue-500 to-blue-400',
        ];
        
        function getSpeakerColor(speaker) {
            if (!speaker) return 'from-gray-500 to-gray-400';
            const match = speaker.match(/SPEAKER_(\d+)/);
            if (match) {
                const idx = parseInt(match[1]) % speakerColors.length;
                return speakerColors[idx];
            }
            return 'from-gray-500 to-gray-400';
        }
        
        function getSpeakerIndex(speaker) {
            if (!speaker) return -1;
            const match = speaker.match(/SPEAKER_(\d+)/);
            return match ? parseInt(match[1]) : -1;
        }
        
        function updateStatus(status, color) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            
            const colors = {
                'disconnected': 'bg-white/20',
                'connecting': 'bg-yellow-500 animate-pulse',
                'connected': 'bg-green-500',
                'recording': 'bg-red-500 animate-pulse',
                'error': 'bg-red-500',
            };
            
            dot.className = `w-2 h-2 rounded-full ${colors[color] || colors.disconnected}`;
            text.textContent = status;
            text.className = color === 'error' ? 'text-red-400' : 'text-white/60';
        }
        
        function updateDuration() {
            if (!startTime) return;
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('duration').textContent = `${minutes}:${seconds}`;
        }
        
        function addTranscript(data) {
            const container = document.getElementById('transcript-container');
            const emptyState = document.getElementById('empty-state');
            
            // Hide empty state
            if (emptyState) {
                emptyState.classList.add('hidden');
            }
            
            // Track speakers
            if (data.speaker) {
                speakers.add(data.speaker);
                document.getElementById('speaker-count').textContent = speakers.size;
            }
            
            const speakerIdx = getSpeakerIndex(data.speaker);
            const speakerColor = getSpeakerColor(data.speaker);
            const speakerClass = speakerIdx >= 0 ? `speaker-${speakerIdx % 8}` : '';
            
            const div = document.createElement('div');
            div.className = `transcript-enter flex gap-4 ${speakerClass}`;
            
            const timestamp = new Date().toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit',
                hour12: false 
            });
            
            div.innerHTML = `
                <div class="flex-shrink-0">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-br ${speakerColor} flex items-center justify-center text-white text-sm font-semibold shadow-lg">
                        ${data.speaker ? data.speaker.replace('SPEAKER_', 'S') : '?'}
                    </div>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="font-medium text-white/90">${data.speaker || 'Unknown'}</span>
                        <span class="text-xs text-white/30 font-mono">${timestamp}</span>
                        <span class="text-xs text-white/20">${data.start?.toFixed(1) || '0.0'}s - ${data.end?.toFixed(1) || '0.0'}s</span>
                    </div>
                    <p class="text-white/70 leading-relaxed">${escapeHtml(data.transcript)}</p>
                </div>
            `;
            
            container.appendChild(div);
            
            // Scroll to bottom
            container.scrollTop = container.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }
        
        async function startRecording() {
            try {
                updateStatus('Connecting...', 'connecting');
                
                // Connect to WebSocket
                websocket = new WebSocket(WEBSOCKET_URL);
                
                websocket.onopen = async () => {
                    console.log('WebSocket connected');
                };
                
                websocket.onmessage = async (event) => {
                    try {
                        // Handle both text and Blob messages
                        let text;
                        if (event.data instanceof Blob) {
                            text = await event.data.text();
                        } else {
                            text = event.data;
                        }
                        
                        const data = JSON.parse(text);
                        console.log('Received:', data);
                        
                        if (data.type === 'session_start') {
                            sessionId = data.session_id;
                            document.getElementById('session-id').textContent = sessionId.substring(0, 8);
                            document.getElementById('session-info').classList.remove('hidden');
                            updateStatus('Recording...', 'recording');
                        } else if (data.transcript) {
                            addTranscript(data);
                        }
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
                
                websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateStatus('Connection error', 'error');
                    stopRecording();
                };
                
                websocket.onclose = () => {
                    console.log('WebSocket closed');
                    if (isRecording) {
                        stopRecording();
                    }
                };
                
                // Wait for connection
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);
                    websocket.addEventListener('open', () => {
                        clearTimeout(timeout);
                        resolve();
                    });
                    websocket.addEventListener('error', () => {
                        clearTimeout(timeout);
                        reject(new Error('Connection failed'));
                    });
                });
                
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });
                
                // Create audio context for resampling
                audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                const source = audioContext.createMediaStreamSource(stream);
                
                // Use ScriptProcessorNode for audio processing (simpler than AudioWorklet)
                const bufferSize = 4096;
                const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording || websocket.readyState !== WebSocket.OPEN) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    
                    // Convert float32 to int16
                    const int16Data = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    
                    // Send as binary
                    websocket.send(int16Data.buffer);
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                // Update UI
                isRecording = true;
                startTime = Date.now();
                speakers.clear();
                
                document.getElementById('mic-icon').classList.add('hidden');
                document.getElementById('waveform').classList.remove('hidden');
                document.getElementById('pulse-rings').classList.remove('hidden');
                document.getElementById('record-btn-inner').classList.remove('from-cyan-500', 'to-purple-600', 'shadow-cyan-500/25');
                document.getElementById('record-btn-inner').classList.add('from-red-500', 'to-red-600', 'shadow-red-500/25');
                document.getElementById('record-text').textContent = 'Click to stop recording';
                
                // Start duration timer
                durationInterval = setInterval(updateDuration, 1000);
                
            } catch (error) {
                console.error('Error starting recording:', error);
                updateStatus('Error: ' + error.message, 'error');
                stopRecording();
            }
        }
        
        function stopRecording() {
            isRecording = false;
            
            // Close WebSocket
            if (websocket) {
                websocket.close();
                websocket = null;
            }
            
            // Close audio context
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            
            // Clear duration timer
            if (durationInterval) {
                clearInterval(durationInterval);
                durationInterval = null;
            }
            
            // Update UI
            document.getElementById('mic-icon').classList.remove('hidden');
            document.getElementById('waveform').classList.add('hidden');
            document.getElementById('pulse-rings').classList.add('hidden');
            document.getElementById('record-btn-inner').classList.add('from-cyan-500', 'to-purple-600', 'shadow-cyan-500/25');
            document.getElementById('record-btn-inner').classList.remove('from-red-500', 'to-red-600', 'shadow-red-500/25');
            document.getElementById('record-text').textContent = 'Click mic to record • Click cloud to upload audio file';
            
            updateStatus('Disconnected', 'disconnected');
        }
        
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            // Reset file input so same file can be selected again
            event.target.value = '';
            
            // Show progress
            const progressDiv = document.getElementById('upload-progress');
            const filenameSpan = document.getElementById('upload-filename');
            const percentSpan = document.getElementById('upload-percent');
            const progressBar = document.getElementById('upload-bar');
            
            progressDiv.classList.remove('hidden');
            filenameSpan.textContent = file.name;
            percentSpan.textContent = '0%';
            progressBar.style.width = '0%';
            
            try {
                updateStatus('Processing audio...', 'connecting');
                
                // Read the audio file
                const arrayBuffer = await file.arrayBuffer();
                
                // Decode the audio
                const tempContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                let audioBuffer;
                try {
                    audioBuffer = await tempContext.decodeAudioData(arrayBuffer);
                } catch (e) {
                    throw new Error('Could not decode audio file. Please try a different format (MP3, WAV, M4A, etc.)');
                }
                
                // Resample to 16kHz mono
                const offlineContext = new OfflineAudioContext(1, audioBuffer.duration * SAMPLE_RATE, SAMPLE_RATE);
                const source = offlineContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineContext.destination);
                source.start();
                
                const resampledBuffer = await offlineContext.startRendering();
                const floatData = resampledBuffer.getChannelData(0);
                
                // Convert float32 to int16
                const int16Data = new Int16Array(floatData.length);
                for (let i = 0; i < floatData.length; i++) {
                    const s = Math.max(-1, Math.min(1, floatData[i]));
                    int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                await tempContext.close();
                
                // Connect to WebSocket
                updateStatus('Connecting...', 'connecting');
                const ws = new WebSocket(WEBSOCKET_URL);
                
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);
                    ws.onopen = () => { clearTimeout(timeout); resolve(); };
                    ws.onerror = () => { clearTimeout(timeout); reject(new Error('Connection failed')); };
                });
                
                // Set up message handler
                ws.onmessage = async (event) => {
                    try {
                        let text;
                        if (event.data instanceof Blob) {
                            text = await event.data.text();
                        } else {
                            text = event.data;
                        }
                        
                        const data = JSON.parse(text);
                        console.log('Received:', data);
                        
                        if (data.type === 'session_start') {
                            sessionId = data.session_id;
                            document.getElementById('session-id').textContent = sessionId.substring(0, 8);
                            document.getElementById('session-info').classList.remove('hidden');
                        } else if (data.transcript) {
                            addTranscript(data);
                        }
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
                
                updateStatus('Uploading audio...', 'recording');
                speakers.clear();
                
                // Send audio in chunks to simulate streaming
                const CHUNK_SIZE = SAMPLE_RATE * 3; // 3 seconds of audio per chunk
                const totalChunks = Math.ceil(int16Data.length / CHUNK_SIZE);
                
                for (let i = 0; i < int16Data.length; i += CHUNK_SIZE) {
                    if (ws.readyState !== WebSocket.OPEN) break;
                    
                    const chunk = int16Data.slice(i, i + CHUNK_SIZE);
                    ws.send(chunk.buffer);
                    
                    // Update progress
                    const progress = Math.min(100, Math.round((i + CHUNK_SIZE) / int16Data.length * 100));
                    percentSpan.textContent = `${progress}%`;
                    progressBar.style.width = `${progress}%`;
                    
                    // Small delay between chunks to let the server process
                    await new Promise(r => setTimeout(r, 100));
                }
                
                // Wait a bit for final processing then close
                await new Promise(r => setTimeout(r, 2000));
                ws.close();
                
                updateStatus('Complete', 'connected');
                progressBar.style.width = '100%';
                percentSpan.textContent = '100%';
                
                // Hide progress after a delay
                setTimeout(() => {
                    progressDiv.classList.add('hidden');
                    updateStatus('Disconnected', 'disconnected');
                }, 3000);
                
            } catch (error) {
                console.error('Error processing file:', error);
                updateStatus('Error: ' + error.message, 'error');
                progressDiv.classList.add('hidden');
            }
        }
    </script>
    </body>
</html>
