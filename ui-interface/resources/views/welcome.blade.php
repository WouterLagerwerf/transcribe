<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Live Transcription</title>
    <link rel="icon" href="/favicon.ico" sizes="any">
    <link rel="preconnect" href="https://fonts.bunny.net">
    <link href="https://fonts.bunny.net/css?family=inter:400,500,600&display=swap" rel="stylesheet" />
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                    },
                    colors: {
                        whatsapp: {
                            dark: '#111b21',
                            darker: '#0b141a',
                            panel: '#202c33',
                            header: '#202c33',
                            teal: '#00a884',
                            tealDark: '#008069',
                            bubble: '#005c4b',
                            bubbleIn: '#202c33',
                            input: '#2a3942',
                            border: '#2a3942',
                            text: '#e9edef',
                            textSecondary: '#8696a0',
                        }
                    }
                }
            }
        }
    </script>
    <style>
        /* Chat background pattern */
        .chat-bg {
            background-color: #0b141a;
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23182229' fill-opacity='0.4'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }
        
        /* Message bubble tails */
        .bubble-out::before {
            content: '';
            position: absolute;
            top: 0;
            right: -8px;
            width: 0;
            height: 0;
            border-left: 8px solid #005c4b;
            border-top: 8px solid transparent;
        }
        .bubble-in::before {
            content: '';
            position: absolute;
            top: 0;
            left: -8px;
            width: 0;
            height: 0;
            border-right: 8px solid #202c33;
            border-top: 8px solid transparent;
        }
        
        /* Speaker colors for avatars */
        .speaker-0 { background: linear-gradient(135deg, #00a884, #008069); }
        .speaker-1 { background: linear-gradient(135deg, #7c3aed, #5b21b6); }
        .speaker-2 { background: linear-gradient(135deg, #ec4899, #be185d); }
        .speaker-3 { background: linear-gradient(135deg, #f97316, #c2410c); }
        .speaker-4 { background: linear-gradient(135deg, #14b8a6, #0d9488); }
        .speaker-5 { background: linear-gradient(135deg, #eab308, #a16207); }
        .speaker-6 { background: linear-gradient(135deg, #ef4444, #b91c1c); }
        .speaker-7 { background: linear-gradient(135deg, #3b82f6, #1d4ed8); }
        
        /* Bubble colors per speaker */
        .bubble-speaker-0 { background-color: #005c4b; }
        .bubble-speaker-1 { background-color: #4c1d95; }
        .bubble-speaker-2 { background-color: #831843; }
        .bubble-speaker-3 { background-color: #7c2d12; }
        .bubble-speaker-4 { background-color: #134e4a; }
        .bubble-speaker-5 { background-color: #713f12; }
        .bubble-speaker-6 { background-color: #7f1d1d; }
        .bubble-speaker-7 { background-color: #1e3a8a; }
        
        .message-enter {
            animation: messageSlide 0.2s ease-out;
        }
        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
        
        /* Waveform animation */
        .waveform-bar { animation: wave 0.5s ease-in-out infinite alternate; }
        .waveform-bar:nth-child(1) { animation-delay: 0s; }
        .waveform-bar:nth-child(2) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(3) { animation-delay: 0.2s; }
        .waveform-bar:nth-child(4) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(5) { animation-delay: 0s; }
        @keyframes wave {
            from { height: 12px; }
            to { height: 24px; }
        }
    </style>
</head>
<body class="bg-whatsapp-darker text-whatsapp-text font-sans antialiased">
    <div class="h-screen flex flex-col max-w-4xl mx-auto shadow-2xl">
        <!-- Header -->
        <header class="bg-whatsapp-header px-4 py-3 flex items-center gap-4 border-b border-whatsapp-border">
            <div class="w-10 h-10 rounded-full bg-gradient-to-br from-whatsapp-teal to-whatsapp-tealDark flex items-center justify-center">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                </svg>
            </div>
            <div class="flex-1">
                <h1 class="font-semibold text-whatsapp-text">Live Transcription</h1>
                <p id="status-text" class="text-xs text-whatsapp-textSecondary">Tap mic to start</p>
            </div>
            <div id="speaker-badges" class="flex -space-x-2">
                <!-- Speaker avatars will appear here -->
            </div>
        </header>

        <!-- Chat Area -->
        <main id="chat-container" class="flex-1 overflow-y-auto chat-bg p-4">
            <!-- Empty state -->
            <div id="empty-state" class="h-full flex flex-col items-center justify-center text-center px-8">
                <div class="w-24 h-24 rounded-full bg-whatsapp-panel flex items-center justify-center mb-6">
                    <svg class="w-12 h-12 text-whatsapp-textSecondary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                    </svg>
                </div>
                <h3 class="text-xl font-medium text-whatsapp-text mb-2">Start a conversation</h3>
                <p class="text-whatsapp-textSecondary text-sm max-w-sm">Record audio or upload a file. Speech will be transcribed with automatic speaker identification.</p>
            </div>
            
            <!-- Messages will be inserted here -->
            <div id="messages-container" class="space-y-1 hidden"></div>
        </main>

        <!-- Input Area -->
        <footer class="bg-whatsapp-header px-4 py-3 border-t border-whatsapp-border">
            <!-- Upload progress -->
            <div id="upload-progress" class="hidden mb-3">
                <div class="flex items-center gap-3 bg-whatsapp-panel rounded-lg p-3">
                    <div class="flex-1">
                        <div class="flex justify-between text-xs mb-1">
                            <span id="upload-filename" class="text-whatsapp-text truncate"></span>
                            <span id="upload-percent" class="text-whatsapp-textSecondary">0%</span>
                        </div>
                        <div class="h-1 bg-whatsapp-dark rounded-full overflow-hidden">
                            <div id="upload-bar" class="h-full bg-whatsapp-teal rounded-full transition-all duration-300" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="flex items-center gap-3">
                <!-- Upload button -->
                <button onclick="document.getElementById('audio-file-input').click()" 
                        class="w-10 h-10 rounded-full bg-whatsapp-panel hover:bg-whatsapp-input transition-colors flex items-center justify-center text-whatsapp-textSecondary hover:text-whatsapp-text">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"></path>
                    </svg>
                </button>
                <input type="file" id="audio-file-input" accept="audio/*" class="hidden" onchange="handleFileUpload(event)">
                
                <!-- Session info display -->
                <div id="session-info" class="hidden flex-1 bg-whatsapp-panel rounded-full px-4 py-2">
                    <div class="flex items-center gap-4 text-xs text-whatsapp-textSecondary">
                        <span>Session: <span id="session-id" class="text-whatsapp-text font-mono">--</span></span>
                        <span>•</span>
                        <span><span id="speaker-count" class="text-whatsapp-text">0</span> speakers</span>
                        <span>•</span>
                        <span id="duration" class="text-whatsapp-text font-mono">00:00</span>
                    </div>
                </div>
                <div id="placeholder-input" class="flex-1 bg-whatsapp-panel rounded-full px-4 py-2.5 text-sm text-whatsapp-textSecondary">
                    Tap mic to record or attach audio file
                </div>
                
                <!-- Record button -->
                <button id="record-btn" onclick="toggleRecording()" 
                        class="w-12 h-12 rounded-full bg-whatsapp-teal hover:bg-whatsapp-tealDark transition-all flex items-center justify-center shadow-lg">
                    <svg id="mic-icon" class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                    </svg>
                    <div id="waveform" class="hidden flex items-center gap-0.5">
                        <div class="waveform-bar w-1 bg-white rounded-full"></div>
                        <div class="waveform-bar w-1 bg-white rounded-full"></div>
                        <div class="waveform-bar w-1 bg-white rounded-full"></div>
                        <div class="waveform-bar w-1 bg-white rounded-full"></div>
                        <div class="waveform-bar w-1 bg-white rounded-full"></div>
                    </div>
                </button>
            </div>
        </footer>
    </div>

    <script>
        const WEBSOCKET_URL = 'ws://localhost:8765';
        const SAMPLE_RATE = 16000;
        
        let isRecording = false;
        let websocket = null;
        let audioContext = null;
        let sessionId = null;
        let startTime = null;
        let durationInterval = null;
        let speakers = new Map(); // speaker -> {color, messageCount}
        let lastSpeaker = null;
        let lastMessageMeta = null; // { speaker, textEl, timeEl, startSeconds }
        
        const speakerColors = ['speaker-0', 'speaker-1', 'speaker-2', 'speaker-3', 'speaker-4', 'speaker-5', 'speaker-6', 'speaker-7'];
        const bubbleColors = ['bubble-speaker-0', 'bubble-speaker-1', 'bubble-speaker-2', 'bubble-speaker-3', 'bubble-speaker-4', 'bubble-speaker-5', 'bubble-speaker-6', 'bubble-speaker-7'];
        
        function getSpeakerIndex(speaker) {
            if (!speaker) return -1;
            const match = speaker.match(/SPEAKER_(\d+)/);
            return match ? parseInt(match[1]) : -1;
        }
        
        function updateStatus(text) {
            document.getElementById('status-text').textContent = text;
        }
        
        function updateDuration() {
            if (!startTime) return;
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('duration').textContent = `${minutes}:${seconds}`;
        }
        
        function updateSpeakerBadges() {
            const container = document.getElementById('speaker-badges');
            container.innerHTML = '';
            
            speakers.forEach((data, speaker) => {
                const idx = getSpeakerIndex(speaker);
                const colorClass = speakerColors[idx % speakerColors.length];
                const badge = document.createElement('div');
                badge.className = `w-8 h-8 rounded-full ${colorClass} flex items-center justify-center text-white text-xs font-semibold ring-2 ring-whatsapp-header`;
                badge.textContent = `S${idx}`;
                badge.title = speaker;
                container.appendChild(badge);
            });
        }
        
        function addMessage(data) {
            const container = document.getElementById('messages-container');
            const emptyState = document.getElementById('empty-state');
            
            // Show messages, hide empty state
            emptyState.classList.add('hidden');
            container.classList.remove('hidden');
            
            const speaker = data.speaker || 'Unknown';
            const speakerIdx = getSpeakerIndex(speaker);
            const colorClass = speakerColors[Math.max(0, speakerIdx) % speakerColors.length];
            const bubbleColor = bubbleColors[Math.max(0, speakerIdx) % bubbleColors.length];
            
            // Track speakers
            if (!speakers.has(speaker)) {
                speakers.set(speaker, { messageCount: 0 });
                document.getElementById('speaker-count').textContent = speakers.size;
                updateSpeakerBadges();
            }
            speakers.get(speaker).messageCount++;
            
            const timestamp = new Date().toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: false 
            });
            
            // Determine if this is a new speaker (show avatar) or continuation
            const isNewSpeaker = !lastMessageMeta || lastSpeaker !== speaker;
            lastSpeaker = speaker;
            
            // Alternate sides based on speaker index (even = left, odd = right)
            const isRight = speakerIdx % 2 === 0;
            let textEl, timeEl;
            
            if (isNewSpeaker || !lastMessageMeta) {
                const wrapper = document.createElement('div');
                wrapper.className = `message-enter flex ${isRight ? 'justify-end' : 'justify-start'} ${isNewSpeaker ? 'mt-4' : 'mt-0.5'}`;
                
                const messageGroup = document.createElement('div');
                messageGroup.className = `flex items-end gap-2 max-w-[85%] ${isRight ? 'flex-row-reverse' : ''}`;
                
                // Avatar (only for new speaker)
                const avatar = document.createElement('div');
                avatar.className = `w-8 h-8 rounded-full ${colorClass} flex items-center justify-center text-white text-xs font-semibold flex-shrink-0`;
                avatar.textContent = `S${speakerIdx >= 0 ? speakerIdx : '?'}`;
                messageGroup.appendChild(avatar);
                
                // Bubble
                const bubble = document.createElement('div');
                bubble.className = `relative px-3 py-2 rounded-lg ${bubbleColor} ${(isRight ? 'rounded-tr-none' : 'rounded-tl-none')}`;
                
                // Add tail
                const tail = document.createElement('div');
                tail.className = `absolute top-0 ${isRight ? '-right-2' : '-left-2'} w-0 h-0`;
                tail.style.cssText = isRight 
                    ? `border-left: 8px solid; border-left-color: inherit; border-top: 8px solid transparent;`
                    : `border-right: 8px solid; border-right-color: inherit; border-top: 8px solid transparent;`;
                bubble.appendChild(tail);
                
                // Speaker name
                const nameEl = document.createElement('div');
                nameEl.className = 'text-xs font-medium text-whatsapp-teal mb-1';
                nameEl.textContent = speaker;
                bubble.appendChild(nameEl);
                
                // Message text
                textEl = document.createElement('p');
                textEl.className = 'text-sm text-whatsapp-text leading-relaxed';
                textEl.textContent = data.transcript;
                bubble.appendChild(textEl);
                
                // Timestamp
                timeEl = document.createElement('div');
                timeEl.className = 'flex items-center justify-end gap-1 mt-1';
                timeEl.innerHTML = `
                    <span class="text-[10px] text-whatsapp-textSecondary">${data.start?.toFixed(1) || '0'}s</span>
                    <span class="text-[10px] text-whatsapp-textSecondary">${timestamp}</span>
                `;
                bubble.appendChild(timeEl);
                
                messageGroup.appendChild(bubble);
                wrapper.appendChild(messageGroup);
                container.appendChild(wrapper);
                
                lastMessageMeta = {
                    speaker,
                    textEl,
                    timeEl,
                    startSeconds: data.start ?? 0,
                };
            } else {
                // Same speaker: append text to last bubble
                textEl = lastMessageMeta.textEl;
                timeEl = lastMessageMeta.timeEl;
                const existing = textEl.textContent || '';
                textEl.textContent = existing ? `${existing} ${data.transcript}` : data.transcript;
                
                // Keep earliest start, keep wall-clock timestamp unchanged
                lastMessageMeta.startSeconds = Math.min(lastMessageMeta.startSeconds, data.start ?? lastMessageMeta.startSeconds);
                timeEl.innerHTML = `
                    <span class="text-[10px] text-whatsapp-textSecondary">${lastMessageMeta.startSeconds.toFixed(1)}s</span>
                    <span class="text-[10px] text-whatsapp-textSecondary">${timestamp}</span>
                `;
            }
            
            // Scroll to bottom
            document.getElementById('chat-container').scrollTop = document.getElementById('chat-container').scrollHeight;
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
                updateStatus('Connecting...');
                
                websocket = new WebSocket(WEBSOCKET_URL);
                
                websocket.onmessage = async (event) => {
                    try {
                        let text = event.data instanceof Blob ? await event.data.text() : event.data;
                        const data = JSON.parse(text);
                        
                        if (data.type === 'session_start') {
                            sessionId = data.session_id;
                            document.getElementById('session-id').textContent = sessionId.substring(0, 8);
                            document.getElementById('session-info').classList.remove('hidden');
                            document.getElementById('placeholder-input').classList.add('hidden');
                            updateStatus(`Recording • ${speakers.size} speakers`);
                        } else if (data.transcript) {
                            addMessage(data);
                            updateStatus(`Recording • ${speakers.size} speakers`);
                        }
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };
                
                websocket.onerror = () => {
                    updateStatus('Connection error');
                    stopRecording();
                };
                
                websocket.onclose = () => {
                    if (isRecording) stopRecording();
                };
                
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Timeout')), 5000);
                    websocket.addEventListener('open', () => { clearTimeout(timeout); resolve(); });
                    websocket.addEventListener('error', () => { clearTimeout(timeout); reject(); });
                });
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                });
                
                audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!isRecording || websocket.readyState !== WebSocket.OPEN) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const int16Data = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    websocket.send(int16Data.buffer);
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                isRecording = true;
                startTime = Date.now();
                speakers.clear();
                lastSpeaker = null;
                lastMessageMeta = null;
                
                document.getElementById('mic-icon').classList.add('hidden');
                document.getElementById('waveform').classList.remove('hidden');
                document.getElementById('record-btn').classList.add('bg-red-500', 'hover:bg-red-600');
                document.getElementById('record-btn').classList.remove('bg-whatsapp-teal', 'hover:bg-whatsapp-tealDark');
                
                durationInterval = setInterval(updateDuration, 1000);
                
            } catch (error) {
                console.error('Error:', error);
                updateStatus('Error: ' + error.message);
                stopRecording();
            }
        }
        
        function stopRecording() {
            isRecording = false;
            
            if (websocket) { websocket.close(); websocket = null; }
            if (audioContext) { audioContext.close(); audioContext = null; }
            if (durationInterval) { clearInterval(durationInterval); durationInterval = null; }
            
            lastMessageMeta = null;
            document.getElementById('mic-icon').classList.remove('hidden');
            document.getElementById('waveform').classList.add('hidden');
            document.getElementById('record-btn').classList.remove('bg-red-500', 'hover:bg-red-600');
            document.getElementById('record-btn').classList.add('bg-whatsapp-teal', 'hover:bg-whatsapp-tealDark');
            
            updateStatus('Tap mic to start');
        }
        
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            event.target.value = '';
            
            const progressDiv = document.getElementById('upload-progress');
            const filenameSpan = document.getElementById('upload-filename');
            const percentSpan = document.getElementById('upload-percent');
            const progressBar = document.getElementById('upload-bar');
            
            progressDiv.classList.remove('hidden');
            filenameSpan.textContent = file.name;
            percentSpan.textContent = '0%';
            progressBar.style.width = '0%';
            
            try {
                updateStatus('Processing audio...');
                
                const arrayBuffer = await file.arrayBuffer();
                const tempContext = new AudioContext({ sampleRate: SAMPLE_RATE });
                const audioBuffer = await tempContext.decodeAudioData(arrayBuffer);
                
                const offlineContext = new OfflineAudioContext(1, audioBuffer.duration * SAMPLE_RATE, SAMPLE_RATE);
                const source = offlineContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineContext.destination);
                source.start();
                
                const resampledBuffer = await offlineContext.startRendering();
                const floatData = resampledBuffer.getChannelData(0);
                
                const int16Data = new Int16Array(floatData.length);
                for (let i = 0; i < floatData.length; i++) {
                    const s = Math.max(-1, Math.min(1, floatData[i]));
                    int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                await tempContext.close();
                
                updateStatus('Connecting...');
                const ws = new WebSocket(WEBSOCKET_URL);
                
                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => reject(new Error('Timeout')), 5000);
                    ws.onopen = () => { clearTimeout(timeout); resolve(); };
                    ws.onerror = () => { clearTimeout(timeout); reject(); };
                });
                
                let lastMessageTime = Date.now();
                speakers.clear();
                lastSpeaker = null;
                lastMessageMeta = null;
                
                ws.onmessage = async (event) => {
                    lastMessageTime = Date.now();
                    try {
                        let text = event.data instanceof Blob ? await event.data.text() : event.data;
                        const data = JSON.parse(text);
                        
                        if (data.type === 'session_start') {
                            sessionId = data.session_id;
                            document.getElementById('session-id').textContent = sessionId.substring(0, 8);
                            document.getElementById('session-info').classList.remove('hidden');
                            document.getElementById('placeholder-input').classList.add('hidden');
                        } else if (data.transcript) {
                            addMessage(data);
                            updateStatus(`Processing • ${speakers.size} speakers`);
                        }
                    } catch (e) {}
                };
                
                updateStatus('Uploading...');
                const CHUNK_SIZE = SAMPLE_RATE * 3;
                
                for (let i = 0; i < int16Data.length; i += CHUNK_SIZE) {
                    if (ws.readyState !== WebSocket.OPEN) break;
                    ws.send(int16Data.slice(i, i + CHUNK_SIZE).buffer);
                    const progress = Math.min(100, Math.round((i + CHUNK_SIZE) / int16Data.length * 100));
                    percentSpan.textContent = `${progress}%`;
                    progressBar.style.width = `${progress}%`;
                    await new Promise(r => setTimeout(r, 100));
                }
                
                updateStatus('Processing...');
                progressBar.style.width = '100%';
                percentSpan.textContent = '100%';
                
                await new Promise(resolve => {
                    const check = setInterval(() => {
                        if (Date.now() - lastMessageTime > 5000) {
                            clearInterval(check);
                            resolve();
                        }
                    }, 500);
                });
                
                ws.close();
                updateStatus(`Complete • ${speakers.size} speakers`);
                
                setTimeout(() => {
                    progressDiv.classList.add('hidden');
                }, 2000);
                
            } catch (error) {
                console.error('Error:', error);
                updateStatus('Error: ' + error.message);
                progressDiv.classList.add('hidden');
            }
        }
    </script>
</body>
</html>
