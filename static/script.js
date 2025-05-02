let controller = null;

function processVideo(type) {
    const progressDiv = document.getElementById('progress');
    const resultList = document.getElementById('result-list');
    resultList.innerHTML = '';
    progressDiv.innerHTML = 'Video işleniyor... <button onclick="cancelProcess()">İptal Et</button>';

    controller = new AbortController();
    const signal = controller.signal;

    let formData = new FormData();
    if (type === 'url') {
        const url = document.getElementById('video-url').value;
        formData.append('video_url', url);
    } else {
        const fileInput = document.getElementById('video-file');
        if (fileInput.files.length === 0) {
            progressDiv.textContent = 'Lütfen bir dosya seçin!';
            return;
        }
        formData.append('video_file', fileInput.files[0]);
    }

    fetch('/process', {
        method: 'POST',
        body: formData,
        signal: signal,
        keepalive: true,  // Bağlantıyı canlı tutmaya çalış
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Sunucu yanıt vermedi: ' + response.statusText);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    if (buffer) {
                        console.warn('Stream bitti, ancak işlenmemiş veri var:', buffer);
                        // İşlenmemiş veriyi kontrol et ve işle
                        try {
                            const data = JSON.parse(buffer);
                            handleData(data);
                        } catch (e) {
                            console.error('Tamamlanmamış JSON parse error:', e);
                        }
                    }
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                let boundary;
                while ((boundary = buffer.indexOf('\n')) !== -1) {
                    const jsonStr = buffer.substring(0, boundary);
                    buffer = buffer.substring(boundary + 1);

                    if (jsonStr.trim() === '') continue;

                    try {
                        const data = JSON.parse(jsonStr);
                        handleData(data);
                    } catch (e) {
                        console.error('JSON parse error:', e, 'Raw data:', jsonStr);
                    }
                }

                readStream();
            }).catch(error => {
                if (error.name === 'AbortError') {
                    progressDiv.textContent = 'İşlem iptal edildi.';
                } else {
                    progressDiv.textContent = `Bağlantı hatası: ${error.message}`;
                    console.error('Stream okuma hatası:', error);
                }
            });
        }

        function handleData(data) {
            if (data.error) {
                progressDiv.textContent = `Hata: ${data.error}`;
                return;
            }

            if (data.progress) {
                progressDiv.innerHTML = `${data.progress} <button onclick="cancelProcess()">İptal Et</button>`;
            }

            if (data.results) {
                resultList.innerHTML = '';
                for (const [name, time] of Object.entries(data.results)) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${name}</td>
                        <td>${time.toFixed(2)}</td>
                    `;
                    resultList.appendChild(row);
                }
            }
        }

        readStream();
    })
    .catch(error => {
        if (error.name === 'AbortError') {
            progressDiv.textContent = 'İşlem iptal edildi.';
        } else {
            progressDiv.textContent = `Bağlantı hatası: ${error.message}`;
            console.error('Fetch hatası:', error);
        }
    });
}

function cancelProcess() {
    if (controller) {
        controller.abort();
    }
}