// ...existing code...
// Senior-level frontend: state, accessibility, hotkeys, filters, export
(function(){
  const qs = (s, r=document)=>r.querySelector(s);
  const qsa = (s, r=document)=>Array.from(r.querySelectorAll(s));
  const input = qs('#fileInput');
  const drop = qs('#drop');
  const btn = qs('#sendBtn');
  const btnClear = qs('#clearBtn');
  const preview = qs('#preview');
  const labels = qs('#labels');
  const meta = qs('#meta');
  const confSlider = qs('#confSlider');
  const confValue = qs('#confValue');
  const optFill = qs('#optFill');
  const optLabels = qs('#optLabels');
  const fltDirt = qs('#fltDirt');
  const fltScratch = qs('#fltScratch');
  const fltDent = qs('#fltDent');
  const fltRust = qs('#fltRust');
  const fltDamage = qs('#fltDamage');
  const imgszInput = qs('#imgszInput');
  const themeBtn = qs('#themeBtn');
  const bgBtn = qs('#bgBtn');
  const helpBtn = qs('#helpBtn');
  const helpModal = qs('#helpModal');
  const helpClose = qs('#helpClose');
  const toasts = qs('#toasts');
  const busy = qs('#busy');

  const classify = [];
  const canvasStore = new WeakMap();

  const clamp = (v,min,max)=>Math.min(max,Math.max(min,v));
  const savedConf = parseFloat(localStorage.getItem('yoloConf') || '0.05');
  const confMin = 0.01, confMax = 0.50;
  const savedFill = localStorage.getItem('optFill');
  const savedLabels = localStorage.getItem('optLabels');
  const savedTheme = localStorage.getItem('theme') || 'dark';
  const savedFilters = JSON.parse(localStorage.getItem('filters')||'{}');
  const savedImgSz = localStorage.getItem('imgsz') || '';
  const savedBg = localStorage.getItem('bgIntensity') || 'high';

  const state = {
    conf: clamp(savedConf, confMin, confMax),
    fill: savedFill === null ? true : savedFill === 'true',
    labels: savedLabels === null ? true : savedLabels === 'true',
    filters: {
      dirt: savedFilters.dirt !== false,
      scratch: savedFilters.scratch !== false,
      dent: savedFilters.dent !== false,
      rust: savedFilters.rust !== false,
      damage: savedFilters.damage !== false,
    },
    imgsz: (savedImgSz && !isNaN(+savedImgSz)) ? +savedImgSz : null,
    files: []
  };

  document.documentElement.setAttribute('data-theme', savedTheme === 'light' ? 'light' : 'dark');
  document.documentElement.setAttribute('data-bg', savedBg === 'light' ? 'light' : 'high');
  function applyThemeLabel(){ themeBtn.textContent = `Тема: ${document.documentElement.getAttribute('data-theme')==='light'?'Светлая':'Тёмная'}`; }
  function applyBgLabel(){ bgBtn.textContent = `Фон: ${document.documentElement.getAttribute('data-bg')==='light'?'Лёгкий':'Высокий'}`; }
  applyThemeLabel();
  applyBgLabel();

  // Toasts
  function toast(msg, type){
    const t = document.createElement('div');
    t.className = 'toast' + (type==='err'?' err':'');
    t.textContent = msg;
    toasts.appendChild(t);
    setTimeout(()=>{ t.remove(); }, 4000);
  }

  // Metadata
  fetch('/api/metadata').then(r=>r.json()).then(m=>{
    const parts = [];
    parts.push(`модель: ${m.backbone}`);
    if (m.yolo_loaded) {
      parts.push(`YOLO: ${m.yolo_weights || 'загружен'}`);
      if (m.yolo_classes?.length) parts.push(`классы: ${m.yolo_classes.join(', ')}`);
    } else {
      parts.push('YOLO: не загружен — боксы недоступны');
    }
    meta.textContent = parts.join(' | ');
  }).catch(()=>{});

  // Bind controls initial state
  confSlider.value = String(state.conf);
  confValue.textContent = state.conf.toFixed(2);
  optFill.checked = state.fill;
  optLabels.checked = state.labels;
  if (imgszInput && state.imgsz) imgszInput.value = String(state.imgsz);
  if (fltDirt) fltDirt.checked = state.filters.dirt;
  if (fltScratch) fltScratch.checked = state.filters.scratch;
  if (fltDent) fltDent.checked = state.filters.dent;
  if (fltRust) fltRust.checked = state.filters.rust;
  if (fltDamage) fltDamage.checked = state.filters.damage;

  confSlider.addEventListener('input', () => {
    state.conf = clamp(parseFloat(confSlider.value || '0.05'), confMin, confMax);
    confValue.textContent = state.conf.toFixed(2);
    localStorage.setItem('yoloConf', String(state.conf));
  });
  optFill.addEventListener('change', ()=>{ state.fill = !!optFill.checked; localStorage.setItem('optFill', String(state.fill)); rerenderAllCanvases(); });
  optLabels.addEventListener('change', ()=>{ state.labels = !!optLabels.checked; localStorage.setItem('optLabels', String(state.labels)); rerenderAllCanvases(); });
  if (imgszInput) imgszInput.addEventListener('change', ()=>{ const v = parseInt(imgszInput.value||''); state.imgsz = (!isNaN(v) && v>=320 && v<=1536) ? v : null; localStorage.setItem('imgsz', state.imgsz?String(state.imgsz):''); });
  [fltDirt, fltScratch, fltDent, fltRust, fltDamage].forEach((el, idx)=>{
    if (!el) return;
    el.addEventListener('change', ()=>{
      const map = ['dirt','scratch','dent','rust','damage'];
      const k = map[idx];
      state.filters[k] = !!el.checked;
      localStorage.setItem('filters', JSON.stringify(state.filters));
      rerenderAllCanvases();
    });
  });

  // Theme & help
  themeBtn.addEventListener('click', ()=>{
    const now = document.documentElement.getAttribute('data-theme')==='light'?'dark':'light';
    document.documentElement.setAttribute('data-theme', now);
    localStorage.setItem('theme', now);
    applyThemeLabel();
    rerenderAllCanvases();
  });
  helpBtn.addEventListener('click', ()=> helpModal.classList.remove('hidden'));
  helpClose.addEventListener('click', ()=> helpModal.classList.add('hidden'));
  helpModal.addEventListener('click', (e)=>{ if (e.target === helpModal) helpModal.classList.add('hidden'); });
  bgBtn.addEventListener('click', ()=>{
    const now = document.documentElement.getAttribute('data-bg')==='light'?'high':'light';
    document.documentElement.setAttribute('data-bg', now);
    localStorage.setItem('bgIntensity', now);
    applyBgLabel();
  });

  // Hotkeys
  window.addEventListener('keydown', (e)=>{
    if (e.key === '?'){ e.preventDefault(); helpModal.classList.toggle('hidden'); return; }
    if (e.key === 't' || e.key === 'T'){ themeBtn.click(); }
  if (e.key === 'b' || e.key === 'B'){ bgBtn.click(); }
    if (e.key === 'l' || e.key === 'L'){ optLabels.click(); }
    if (e.key === 'f' || e.key === 'F'){ optFill.click(); }
    if (e.key === 'd' || e.key === 'D'){
      const btn = qs('.canvas-wrap .dl'); if (btn) btn.click();
    }
  });

  // Dropzone
  drop.addEventListener('click', ()=> input.click());
  drop.addEventListener('keydown', (e)=>{ if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); input.click(); }});
  drop.addEventListener('dragover', (e)=>{ e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', ()=> drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e)=>{
    e.preventDefault(); drop.classList.remove('dragover');
    const files = e.dataTransfer?.files;
    if (files && files.length){
      state.files = Array.from(files);
      renderPreview();
    }
  });

  input.addEventListener('change', () => {
    state.files = input.files ? Array.from(input.files) : [];
    renderPreview();
  });

  function renderPreview(){
    preview.innerHTML = '';
    if (!state.files.length) return;
    state.files.slice(0, 6).forEach((f, idx) => {
      const img = document.createElement('img');
      img.style.maxWidth = '100%';
      img.style.borderRadius = '8px';
      img.style.marginBottom = '8px';
      img.dataset.idx = String(idx);
      img.src = URL.createObjectURL(f);
      preview.appendChild(img);
    });
  }

  btnClear.addEventListener('click', ()=>{
    input.value = '';
    state.files = [];
    preview.innerHTML = '';
    labels.innerHTML = '';
  });

  btn.addEventListener('click', onAnalyze);

  async function onAnalyze(){
  if (!state.files.length) return alert('Выберите файл(ы)');
    busy.classList.remove('hidden');
    btn.disabled = true; btn.textContent = 'Загрузка…';
    btnClear.disabled = true;
    try {
  const fd = new FormData();
  state.files.forEach(f => fd.append('files', f));
      const resp = await fetch('/api/batch_predict', { method: 'POST', body: fd });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      labels.innerHTML = '';
      if (data?.results?.length) {
        data.results.forEach((item, idx) => {
          const wrap = document.createElement('div');
          wrap.className = 'line';
          if (item.error) {
            const err = document.createElement('div');
            err.className = 'badge bad';
            err.textContent = `❌ ${item.filename || ''}: ${item.error}`;
            wrap.appendChild(err);
            labels.appendChild(wrap);
            return;
          }
          const cleanLabel = item.pred_clean_ru || (item.pred_clean === 'clean' ? 'чистый' : 'грязный');
          const damageLabel = item.pred_damage_ru || (item.pred_damage === 'damaged' ? 'битый' : 'целый');

          const title = document.createElement('div');
          title.className = 'file';
          title.textContent = item.filename || '';
          labels.appendChild(title);

          const dirtBadge = document.createElement('span');
          const dmgBadge = document.createElement('span');
          const dirtIsClean = (cleanLabel === 'чистый');
          dirtBadge.className = 'badge ' + (dirtIsClean ? 'ok' : 'bad');
          dirtBadge.textContent = '';
          dirtBadge.appendChild(document.createTextNode(dirtIsClean ? '🚗✅ Чистая' : '🚗❌ Грязная'));

          const dmgIsDamaged = (damageLabel === 'битый');
          dmgBadge.className = 'badge ' + (dmgIsDamaged ? 'bad' : 'ok');
          dmgBadge.textContent = '';
          dmgBadge.appendChild(document.createTextNode(dmgIsDamaged ? '🚗❌ Есть повреждения' : '🚗✅ Целая'));

          wrap.appendChild(dirtBadge);
          wrap.appendChild(dmgBadge);
          labels.appendChild(wrap);

          classify[idx] = {
            dirty: !dirtIsClean,
            damaged: dmgIsDamaged,
            cleanProb: item.clean_prob,
            damagedProb: item.damaged_prob,
          };

          const detLine = document.createElement('div');
          detLine.className = 'line';
          detLine.dataset.idx = String(idx);
          detLine.innerHTML = '<span class="muted">Классификация выполнена • Детектор (YOLO): подготовка…</span>';
          const pr = document.createElement('div');
          pr.style.cssText = 'width:100%;height:8px;border-radius:999px;background:#0f1512;border:1px solid var(--stroke);overflow:hidden;';
          const bar = document.createElement('div');
          bar.style.cssText = 'height:100%;width:50%;background:linear-gradient(90deg, rgba(0,200,83,0.55), rgba(0,200,83,0.25));transition:width .2s';
          pr.appendChild(bar);
          detLine.appendChild(pr);
          detLine.dataset.progressRef = 'bar';
          labels.appendChild(detLine);
        });
      }

      // YOLO overlays with filters and optional imgsz
  await renderOverlays();

      // After render add Export JSON button near first canvas
      const wrap1 = qs('.canvas-wrap');
      if (wrap1){
        const exportBtn = document.createElement('button');
        exportBtn.className = 'dl';
        exportBtn.style.right = '86px';
        exportBtn.textContent = 'JSON';
        exportBtn.title = 'Экспорт результатов детекции в JSON';
        exportBtn.addEventListener('click', exportDetectionsJSON);
        wrap1.appendChild(exportBtn);
      }
    } catch (e) {
      console.error(e);
      toast('Ошибка: ' + (e?.message || e), 'err');
    } finally {
      btn.disabled = false; btn.textContent = 'Отправить';
      btnClear.disabled = false;
      busy.classList.add('hidden');
    }
  }

  async function renderOverlays(){
    // Clear old
    qsa('img.det,canvas.det').forEach(n => n.remove());
    for (let i = 0; i < Math.min(state.files.length, 6); i++) {
      const f = state.files[i];
      const url = URL.createObjectURL(f);
      const baseImg = new Image();
      baseImg.onload = async () => {
        try {
          const wrap = document.createElement('div');
          wrap.className = 'canvas-wrap';
          const canvas = document.createElement('canvas');
          canvas.className = 'det';
          canvas.width = baseImg.naturalWidth;
          canvas.height = baseImg.naturalHeight;
          canvas.style.maxWidth = '100%';
          canvas.style.border = '1px solid var(--stroke)';
          canvas.style.borderRadius = '8px';
          canvas.style.margin = '4px 0 6px 0';
          wrap.appendChild(canvas);
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);

          const fdFile = new FormData(); fdFile.append('file', f);
          const detLinePrep = qsa('.line').find(el => el.dataset?.idx === String(i));
          const params = new URLSearchParams({ conf: state.conf.toFixed(2) });
          if (state.imgsz) params.set('imgsz', String(state.imgsz));
          const rDet = await fetch(`/api/detect?${params.toString()}`, { method: 'POST', body: fdFile });
          const detJson = rDet.ok ? await rDet.json() : { detections: [] };
          if (detLinePrep){ const bar = detLinePrep.querySelector('div > div'); if (bar) bar.style.width = '75%'; detLinePrep.firstChild && (detLinePrep.firstChild.textContent = 'Детектор (YOLO): рисуем боксы…'); }
          const wanted = new Set(['царапина','ржавчина','вмятина','грязь','повреждение']);
          let detections = (detJson?.detections || []).filter(d => wanted.has((d.class||'').toLowerCase()));

          // Apply client-side filters
          detections = detections.filter(d =>{
            const cls = (d.class||'').toLowerCase();
            if (cls==='грязь' && !state.filters.dirt) return false;
            if (cls==='царапина' && !state.filters.scratch) return false;
            if (cls==='вмятина' && !state.filters.dent) return false;
            if (cls==='ржавчина' && !state.filters.rust) return false;
            if (cls==='повреждение' && !state.filters.damage) return false;
            return true;
          });

          if ((!detections || detections.length === 0) && classify[i]) {
            const clsVerdict = classify[i];
            const camFetch = async (head) => {
              const fdCam = new FormData(); fdCam.append('file', f);
              const rCam = await fetch(`/api/heatmap_boxes?head=${encodeURIComponent(head)}`, { method: 'POST', body: fdCam });
              if (!rCam.ok) return null;
              try { return await rCam.json(); } catch { return null; }
            };
            const camBoxes = [];
            try { if (clsVerdict.damaged) { const r1 = await camFetch('damaged'); if (r1?.boxes?.length) camBoxes.push({ label: r1.label || 'повреждение', boxes: r1.boxes }); } } catch {}
            try { if (clsVerdict.dirty) { const r2 = await camFetch('dirty'); if (r2?.boxes?.length) camBoxes.push({ label: r2.label || 'грязь', boxes: r2.boxes }); } } catch {}
            if (camBoxes.length) {
              const camDet = [];
              for (const g of camBoxes) for (const b of g.boxes) if (Array.isArray(b) && b.length === 4) camDet.push({ box: b.map(Number), class: String(g.label||'') , score: 0.5 });
              detections = camDet;
            }
          }

          if (detections.length) {
            ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) * 0.003));
            ctx.font = `${Math.max(12, Math.round(canvas.width * 0.018))}px Inter, Roboto, system-ui, Arial`;
            ctx.textBaseline = 'top';
            const drawAll = (selIndex = -1) => {
              ctx.clearRect(0,0,canvas.width,canvas.height);
              ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);
              detections.forEach((d, di) => {
                const [x1,y1,x2,y2] = d.box || [];
                if ([x1,y1,x2,y2].some(v => typeof v !== 'number')) return;
                const w = x2 - x1, h = y2 - y1;
                const cls = (d.class||'').toLowerCase();
                let color = '#FFD54F';
                let fill = 'rgba(255,213,79,0.18)';
                if (cls === 'грязь') { color = '#00C853'; fill = 'rgba(0,200,83,0.18)'; }
                else if (cls === 'вмятина') { color = '#FF6F00'; fill = 'rgba(255,111,0,0.18)'; }
                else if (cls === 'ржавчина') { color = '#F44336'; fill = 'rgba(244,67,54,0.18)'; }
                else if (cls === 'повреждение') { color = '#1E88E5'; fill = 'rgba(30,136,229,0.14)'; }
                const hl = di === selIndex ? 1.0 : 0.18;
                if (state.fill){
                  const fcol = hl>0.2 ? (cls==='повреждение'?'rgba(30,136,229,0.28)': fill.replace('0.18','0.35')) : fill;
                  ctx.fillStyle = fcol;
                  ctx.fillRect(x1, y1, w, h);
                }
                ctx.lineWidth = di === selIndex ? Math.max(3, ctx.lineWidth*1.6) : ctx.lineWidth;
                ctx.strokeStyle = color;
                ctx.strokeRect(x1, y1, w, h);
                if (state.labels){
                  const label = `${d.class || ''}`.trim();
                  const tw = ctx.measureText(label).width;
                  const th = Math.max(18, Math.round(canvas.width * 0.028));
                  const bx = Math.max(0, x1);
                  const by = Math.max(0, y1 - th);
                  ctx.fillStyle = 'rgba(20,22,21,0.88)';
                  ctx.fillRect(bx, by, tw + 14, th);
                  ctx.fillStyle = color;
                  ctx.fillRect(bx, by, 3, th);
                  ctx.fillStyle = '#EAEAEA';
                  ctx.fillText(label, bx + 6, by + 2);
                }
              });
            };
            drawAll(-1);
            canvasStore.set(canvas, { img: baseImg, detections, drawAll, selected: -1 });
            canvas.addEventListener('click', (ev)=>{
              const rect = canvas.getBoundingClientRect();
              const scaleX = canvas.width / rect.width;
              const scaleY = canvas.height / rect.height;
              const x = (ev.clientX - rect.left) * scaleX;
              const y = (ev.clientY - rect.top) * scaleY;
              let hit = -1;
              for (let di = detections.length-1; di>=0; di--){
                const [x1,y1,x2,y2] = detections[di].box || [];
                if (x>=x1 && x<=x2 && y>=y1 && y<=y2){ hit = di; break; }
              }
              const st = canvasStore.get(canvas);
              st.selected = hit;
              st.drawAll(hit);
            });
          }

          const counts = { 'грязь':0, 'царапина':0, 'вмятина':0, 'ржавчина':0, 'повреждение':0 };
          for (const d of detections) {
            const cls = (d.class||'').toLowerCase();
            if (counts.hasOwnProperty(cls)) counts[cls]++;
          }
          const total = detections.length;
          const detLine = qsa('.line').find(el => el.dataset?.idx === String(i));
          if (detLine) {
            detLine.innerHTML = '';
            if (total === 0) {
              const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = 'Объекты детектора не найдены (YOLO)'; detLine.appendChild(chip);
              const cls = classify[i] || {};
              if (cls.dirty || cls.damaged) {
                const note = document.createElement('span');
                note.className = 'muted';
                note.style.marginLeft = '8px';
                note.textContent = 'Классификация видит проблему по фото целиком. Попробуйте снизить порог сверху или подключить веса с нужными классами.';
                detLine.appendChild(note);
              }
            } else {
              const totalChip = document.createElement('span'); totalChip.className = 'chip'; totalChip.textContent = `Найдено объектов: ${total}`; detLine.appendChild(totalChip);
              const makeChip = (text, cls) => { const c = document.createElement('span'); c.className = 'chip ' + (cls||''); c.textContent = text; return c; };
              if (counts['грязь']) detLine.appendChild(makeChip(`грязь: ${counts['грязь']}`, 'dirt'));
              if (counts['царапина']) detLine.appendChild(makeChip(`царапина: ${counts['царапина']}`, 'scratch'));
              if (counts['вмятина']) detLine.appendChild(makeChip(`вмятина: ${counts['вмятина']}`, 'dent'));
              if (counts['ржавчина']) detLine.appendChild(makeChip(`ржавчина: ${counts['ржавчина']}`, 'rust'));
              if (counts['повреждение']) detLine.appendChild(makeChip(`повреждение: ${counts['повреждение']}`, 'damage'));
            }
          }

          // download PNG
          const dl = document.createElement('button');
          dl.className = 'dl';
          dl.textContent = 'Скачать';
          dl.addEventListener('click', ()=>{
            try{
              const link = document.createElement('a');
              link.download = (f.name?.replace(/\.[^.]+$/, '') || `image_${i}`) + '_boxes.png';
              link.href = canvas.toDataURL('image/png');
              link.click();
            }catch(e){ toast('Не удалось скачать изображение', 'err'); }
          });
          wrap.appendChild(dl);

          const prevImg = preview.querySelector(`img[data-idx="${i}"]`);
          if (prevImg) prevImg.replaceWith(wrap); else preview.appendChild(wrap);

          if (detLine){ const bar = detLine.querySelector('div > div'); if (bar) bar.style.width = '100%'; detLine.firstChild && (detLine.firstChild.textContent = 'Детектор (YOLO): готово'); }
        } catch (e) {
          console.warn('overlay error', e);
        } finally {
          URL.revokeObjectURL(url);
        }
      };
      baseImg.src = url;
    }
  }

  function exportDetectionsJSON(){
    try{
      const all = [];
      qsa('canvas.det').forEach((cv, idx)=>{
        const st = canvasStore.get(cv);
        if (!st) return;
        all.push({ index: idx, width: cv.width, height: cv.height, detections: st.detections });
      });
      const blob = new Blob([JSON.stringify({ generatedAt: new Date().toISOString(), results: all }, null, 2)], { type: 'application/json' });
      const link = document.createElement('a');
      link.download = 'detections.json';
      link.href = URL.createObjectURL(blob);
      link.click();
      setTimeout(()=> URL.revokeObjectURL(link.href), 2000);
    }catch(e){ toast('Не удалось экспортировать JSON', 'err'); }
  }

  // Rerender utility
  function rerenderAllCanvases(){
    qsa('canvas.det').forEach(cv=>{
      const st = canvasStore.get(cv);
      if (!st) return;
      const ctx = cv.getContext('2d');
      if (!ctx) return;
      const detections = st.detections;
      const baseImg = st.img;
      const drawAll = (selIndex = st.selected||-1) => {
        ctx.clearRect(0,0,cv.width,cv.height);
        ctx.drawImage(baseImg, 0, 0, cv.width, cv.height);
        ctx.lineWidth = Math.max(2, Math.round(Math.min(cv.width, cv.height) * 0.003));
        ctx.font = `${Math.max(12, Math.round(cv.width * 0.018))}px Inter, Roboto, system-ui, Arial`;
        ctx.textBaseline = 'top';
        detections.forEach((d, di) => {
          const [x1,y1,x2,y2] = d.box || [];
          if ([x1,y1,x2,y2].some(v => typeof v !== 'number')) return;
          const w = x2 - x1, h = y2 - y1;
          const cls = (d.class||'').toLowerCase();
          // respect filters
          if (cls==='грязь' && !state.filters.dirt) return;
          if (cls==='царапина' && !state.filters.scratch) return;
          if (cls==='вмятина' && !state.filters.dent) return;
          if (cls==='ржавчина' && !state.filters.rust) return;
          if (cls==='повреждение' && !state.filters.damage) return;
          let color = '#FFD54F';
          let fill = 'rgba(255,213,79,0.18)';
          if (cls === 'грязь') { color = '#00C853'; fill = 'rgba(0,200,83,0.18)'; }
          else if (cls === 'вмятина') { color = '#FF6F00'; fill = 'rgba(255,111,0,0.18)'; }
          else if (cls === 'ржавчина') { color = '#F44336'; fill = 'rgba(244,67,54,0.18)'; }
          else if (cls === 'повреждение') { color = '#1E88E5'; fill = 'rgba(30,136,229,0.14)'; }
          const hl = di === selIndex ? 1.0 : 0.18;
          if (state.fill){
            const fcol = hl>0.2 ? (cls==='повреждение'?'rgba(30,136,229,0.28)': fill.replace('0.18','0.35')) : fill;
            ctx.fillStyle = fcol;
            ctx.fillRect(x1, y1, w, h);
          }
          ctx.strokeStyle = color;
          ctx.lineWidth = di === selIndex ? Math.max(3, Math.round(Math.min(cv.width, cv.height) * 0.0045)) : Math.max(2, Math.round(Math.min(cv.width, cv.height) * 0.003));
          ctx.strokeRect(x1, y1, w, h);
          if (state.labels){
            const label = `${d.class || ''}`.trim();
            const tw = ctx.measureText(label).width;
            const th = Math.max(18, Math.round(cv.width * 0.028));
            const bx = Math.max(0, x1);
            const by = Math.max(0, y1 - th);
            ctx.fillStyle = 'rgba(20,22,21,0.88)';
            ctx.fillRect(bx, by, tw + 14, th);
            ctx.fillStyle = color;
            ctx.fillRect(bx, by, 3, th);
            ctx.fillStyle = '#EAEAEA';
            ctx.fillText(label, bx + 6, by + 2);
          }
        });
      };
      drawAll(st.selected||-1);
      st.drawAll = drawAll;
    });
  }
  window.rerenderAllCanvases = rerenderAllCanvases;
})();
