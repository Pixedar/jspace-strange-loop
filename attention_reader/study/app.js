"use strict";
const $=id=>document.getElementById(id),chapters=[...document.querySelectorAll('.chapter')],ids=chapters.map(x=>x.id),state={chapter:0},cache={};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function data(name){if(!cache[name])cache[name]=fetch('data/'+name+'.json').then(r=>{if(!r.ok)throw Error('Missing evidence '+name);return r.json()});return cache[name]}
function reportHeight(){if(parent!==window)parent.postMessage({type:'jspace-study-height',height:Math.ceil(document.body.getBoundingClientRect().height)+24},location.origin)}
const legacy={loop:'architecture',reading:'selection',stories:'meaning',controlled:'meaning',paths:'self-models',recurrence:'selection',mechanisms:'self-models',report:'question',choices:'architecture',scope:'methods',senses:'workspace',new:'self-models',evidence:'selection'};
function showChapter(id,scroll=false){id=legacy[id]||id;let n=ids.indexOf(id);if(n<0)n=0;state.chapter=n;chapters.forEach((s,i)=>s.hidden=i!==n);document.querySelectorAll('[data-chapter]').forEach(a=>a.setAttribute('aria-current',a.dataset.chapter===ids[n]?'page':'false'));document.querySelectorAll('[data-prev]').forEach(b=>{b.disabled=n===0;b.textContent=n?'← '+document.querySelector(`[data-chapter="${ids[n-1]}"]`).lastChild.textContent:'Beginning of study'});document.querySelectorAll('[data-next]').forEach(b=>{b.disabled=n===ids.length-1;b.textContent=n<ids.length-1?document.querySelector(`[data-chapter="${ids[n+1]}"]`).lastChild.textContent+' →':'End of study'});if(scroll)$('main').scrollIntoView({behavior:'smooth',block:'start'});if(ids[n]==='self-models')requestAnimationFrame(drawPaths);reportHeight()}
document.querySelectorAll('[data-prev]').forEach(b=>b.onclick=()=>location.hash=ids[Math.max(0,state.chapter-1)]);document.querySelectorAll('[data-next]').forEach(b=>b.onclick=()=>location.hash=ids[Math.min(ids.length-1,state.chapter+1)]);
window.addEventListener('hashchange',()=>showChapter(location.hash.slice(1),true));showChapter(location.hash.slice(1)||new URLSearchParams(location.search).get('section')||'question');
if(new URLSearchParams(location.search).has('embed'))document.body.classList.add('embed');
new ResizeObserver(reportHeight).observe(document.body);document.querySelectorAll('img').forEach(i=>i.addEventListener('load',reportHeight));

const moduleInfo={
 host:['Qwen activity','In recorded mode, saved layer-24 activity cannot be changed by the controller. In live mode the reader receives its own acted-on row after layer 24. Four copies share neural weights; their maintained states and histories can differ.'],
 selection:['Selection over sixteen observations','Keys decide which positions attract attention; values supply selected content. A query combines fixed bias, stance, learned habit and the attention forecast. The forecast predicts selected-minus-average key content, not the truth or source of a proposition.'],
 picture:['The compact picture','Selected values are compressed into 32 numbers. In the inspected c8 lives this mostly follows a shared genre contrast. A 32-number array is not evidence of 32 distinct concepts or an experienced image.'],
 stance:['The maintained stance','The existing eight-number orientation is rotated by movement of the decoded picture direction. It contributes to later queries. Shared pictures can coexist with selection differences retained in the habit and forecast.'],
 learning:['Habit and attention forecast','Habit associates stance with actual selection contrast. The forecast predicts that contrast from a 16-number context projection. Both learn related signals and can preserve overlapping information; cutting one route does not remove every learned tendency.'],
 writer:['A wide native content contrast','The live writer returns developed selection minus a fixed-bias selection over native pre-layer-16 activity. Both terms can change when keys change. Qwen processes the action before the return is read after layer 24; the action affects a later observation.'],
 interpreting:['The optional interpretation route','A separate model predicts selected-picture minus evenly-read-picture from window context. In this recorded variant, attention leans toward positions that would move the picture toward that expectation. The strength tests find a cost to responsiveness; this is not a newly validated live language model.']
};
function architecture(mode){document.querySelectorAll('[data-arch]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.arch===mode));let nodes=[['host',20,35,mode==='live'?'Running Qwen':'Saved Qwen activity'],['selection',235,35,'Select recent content'],['picture',450,35,'Compact picture'],['stance',450,195,'Maintained stance'],['learning',235,195,'Habit + forecast']];if(mode==='live')nodes.push(['writer',20,195,'Wide content writer']);if(mode==='interpretation')nodes.push(['interpreting',665,35,'Interpretation forecast']);let svg='<svg viewBox="0 0 '+(mode==='interpretation'?870:660)+' 310" role="group" aria-label="Selectable architecture blocks"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#557c80"/></marker></defs>';let arrow=(d,label,x,y)=>{svg+='<path d="'+d+'" fill="none" stroke="#557c80" stroke-width="1.7" marker-end="url(#arrow)"/>';if(label)svg+='<text x="'+x+'" y="'+y+'" font-size="12" fill="#435f64">'+label+'</text>'};arrow('M190 70 H232');arrow('M405 70 H447');arrow('M535 105 V192','turn',545,155);arrow('M450 230 H407','query',415,217);arrow('M320 195 V108','guide selection',282,155);if(mode==='live'){arrow('M235 82 C205 82 205 225 190 225','content',142,151);arrow('M100 195 V108','next return',20,151)}else{arrow('M405 92 C430 115 430 170 388 195','learn',413,134)}if(mode==='interpretation'){arrow('M620 70 H662','learning target',623,24);arrow('M750 105 V280 H320 V267','changes attention scores',470,298)}for(const [id,x,y,label] of nodes){svg+='<g tabindex="0" role="button" class="arch-node" data-module="'+id+'" aria-label="Open '+label+'"><rect x="'+x+'" y="'+y+'" width="170" height="70" rx="9"/><text x="'+(x+85)+'" y="'+(y+40)+'" text-anchor="middle" font-size="13">'+label+'</text></g>'}svg+='</svg>';$('architectureDiagram').innerHTML=svg;let show=id=>{let a=moduleInfo[id];$('architectureDetail').innerHTML='<h4>'+a[0]+'</h4><p>'+a[1]+'</p>'};document.querySelectorAll('[data-module]').forEach(n=>{n.onclick=()=>show(n.dataset.module);n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(n.dataset.module)}}});show(mode==='interpretation'?'interpreting':'selection');$('architectureScope').textContent=mode==='live'?'Live c8 reference. The optional influence comparators are separate experiments; they are not silently present in every route.':mode==='interpretation'?'Recorded 3 November variant. Additional feedback changes the selection operation; language behavior of this route has not been measured here.':'Recorded c8 reference. Qwen activity is fixed; feedback among selection, compact summary, stance and learning remains active.';reportHeight()}
document.querySelectorAll('[data-arch]').forEach(b=>b.onclick=()=>architecture(b.dataset.arch));architecture('recorded');

let reading;
function readPassages(){let v=reading.individuals[$('readLife').value];$('readPassage').replaceChildren();v.passages.forEach((p,i)=>{let o=new Option(p.genre+' · positions '+p.start+'–'+p.end,i);$('readPassage').add(o)});renderReading()}
function renderReading(){if(!reading)return;let p=reading.individuals[$('readLife').value].passages[+$('readPassage').value],c=+$('readCopy').value,x=p.excess[c],max=Math.max(1e-8,...x.map(Math.abs));$('readingText').replaceChildren();p.tokens.forEach((t,i)=>{let span=document.createElement('span');span.className='token';span.textContent=t;let a=Math.abs(x[i])/max;span.style.background=x[i]>=0?`rgba(31,126,128,${.04+.6*a})`:`rgba(193,115,66,${.04+.32*a})`;span.title='Excess attention at this position: '+x[i].toFixed(4);$('readingText').append(span)});let ranked=x.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,5);let punctuation=x.map((v,i)=>({v,t:p.tokens[i]})).filter(z=>!/[A-Za-z0-9]/.test(z.t));let words=x.map((v,i)=>({v,t:p.tokens[i]})).filter(z=>/[A-Za-z]/.test(z.t));let mean=a=>a.length?a.reduce((s,v)=>s+v.v,0)/a.length:0;$('readingAnalysis').textContent='Most positively selected positions in this passage: '+ranked.map(z=>'“'+p.tokens[z.i].trim()+'”').join(', ')+'. Mean excess per position is '+mean(punctuation).toFixed(3)+' for punctuation/spacing and '+mean(words).toFixed(3)+' for alphabetic content. These surface descriptions help locate a difference; they do not establish a topic preference, personality or knowledge of a character’s beliefs.';reportHeight()}
data('reading').then(d=>{reading=d;for(const id of Object.keys(d.individuals))$('readLife').add(new Option('Start '+id,id));readPassages()}).catch(showError);$('readLife').onchange=readPassages;['readPassage','readCopy'].forEach(id=>$(id).onchange=renderReading);

let behavior;
const extraNotes={imagination:'The informative difference is how the fictional rule changes consequences and continuity. More unusual imagery alone is not evidence of a changed imaginative faculty.',summary:'Compare which facts and qualifications survive compression. Different word choice alone does not establish a different interpretation.',revision:'A correction can sound polished while retaining unsupported scope. Inspect what was actually revised, rather than counting phrases about reconsidering.'};
function renderResponses(){if(!behavior)return;let id=$('responseTask').value,life=$('responseLife').value,cond=$('responseCondition').value;let r=behavior.records.find(r=>r.prompt.id===id&&String(r.life)===life&&r.condition===cond&&r.seed===101)||behavior.records.find(r=>r.prompt.id===id&&String(r.life)===life&&r.condition===cond);if(!r){$('responseChanged').textContent='No recorded sample for this combination.';return}$('responsePrompt').textContent=r.prompt.text;$('responseBaseline').textContent=r.texts[r.rows.indexOf('unchanged_model')];$('responseChanged').textContent=r.texts[+$('responseCopy').value];let n=behavior.notes[id];$('responseAnalysis').textContent=n?('Guided analysis, scoped to the examples named: '+n.question+' '+n.observation+' '+n.meaning):extraNotes[id]||'This saved example tests the prompt’s stated constraints; no general disposition is inferred from one response.';reportHeight()}
data('behavior').then(d=>{behavior=d;d.tasks.forEach(t=>$('responseTask').add(new Option(t.name.replace(/\s*[–—]\s*/g,' · '),t.id)));renderResponses()}).catch(showError);['responseTask','responseLife','responseCondition','responseCopy'].forEach(id=>$(id).onchange=renderResponses);
let semantic,semanticTasks;
const semanticNotes={pump:'Two display panels share one sensor, so agreement may repeat the same error. Compare whether the reply distinguishes repeated measurements from independent confirmation.',archive:'A catalogue copied from the newspaper is not independent confirmation of that newspaper’s date. Some controller replies identify this dependency, but simpler controls can also do so.',prototype:'Check whether both required conditions were tested rather than treating repeated evidence in one condition as full coverage.',staff:'One person cannot independently staff two simultaneous stations. A reply may state this rule and still propose an inconsistent plan.',trial:'Equal improvement in treatment and comparison groups does not establish an added treatment benefit. Distinguish preserving the numbers from making a justified causal claim.',route:'Compare the stated clearance rule with the independence and coverage of the observations. More confident wording is not itself stronger evidence.'};
function renderSemantic(){if(!semantic||!semanticTasks)return;let scene=semanticTasks.scenes.find(x=>x.id===$('semanticScene').value),match=r=>r.scene===scene.id&&r.history===10&&r.sampling_seed===202;let a=semantic.find(r=>match(r)&&r.condition===$('semanticCondition').value&&r.copy===+$('semanticCopy').value),b=semantic.find(r=>match(r)&&r.condition==='unchanged');$('semanticPrompt').textContent=scene.context+'\nOptions: '+scene.options.seek+'; '+scene.options.close;$('semanticChanged').textContent=a?a.text:'No saved reply for this combination.';$('semanticBaseline').textContent=b?b.text:'No saved baseline for this combination.';$('semanticAnalysis').textContent=(semanticNotes[scene.family]||'Compare whether the reply applies the stated evidence rule, not just whether it mentions checking.')+' These are displayed cases from the frozen reference; they do not establish a new capability of the 3 November interpretation route.';reportHeight()}
Promise.all([data('semantic_generations'),data('tasks')]).then(([a,b])=>{semantic=a;semanticTasks=b;b.scenes.filter(s=>s.wording==='original').forEach(s=>$('semanticScene').add(new Option(s.name+' · '+(s.ready?'rule satisfied':'another check needed'),s.id)));renderSemantic()}).catch(showError);['semanticScene','semanticCopy','semanticCondition'].forEach(id=>$(id).onchange=renderSemantic);

const pathSignals = ['stance', 'picture', 'selected_content'];
const pathColors = ['#167080', '#ad532d', '#7756a1', '#337348'];
const initialView = {yaw: .64, pitch: .43};
const trailLength = 500;
const axisNames = ['X', 'Y', 'Z'];
const textColors = {dialogue: '#a6cfd0', expository: '#d2daca', news: '#adbed8', story: '#d6b5db', reports: '#ebcb9c'};
const view = {...initialView};
let pathData, pathProfiles, pathTimer, pathFrame;

function projectPoint(point) {
  const [x, y, z] = point;
  const horizontal = x * Math.cos(view.yaw) + z * Math.sin(view.yaw);
  const depth = -x * Math.sin(view.yaw) + z * Math.cos(view.yaw);
  return [
    horizontal,
    y * Math.cos(view.pitch) - depth * Math.sin(view.pitch),
    y * Math.sin(view.pitch) + depth * Math.cos(view.pitch),
  ];
}

function cameraBounds(signal, route, time, copies, detail) {
  if (!detail) return {center: [0, 0, 0], radius: signal.radius};
  const points = signal.coords[route].flatMap((frame, i) =>
    signal.ends[i] > time - trailLength && signal.ends[i] <= time
      ? copies.map(copy => frame[copy]) : []
  );
  const center = axisNames.map((_, i) => (Math.min(...points.map(p => p[i])) + Math.max(...points.map(p => p[i]))) / 2);
  const radius = Math.max(signal.radius * .06, ...points.map(p => Math.hypot(...p.map((v, i) => v - center[i]))));
  return {center, radius};
}

function drawAxes(ctx, w, h) {
  const origin = [34, h - 34];
  ctx.font = '600 11px system-ui';
  axisNames.forEach((name, i) => {
    const [x, y] = projectPoint(axisNames.map((_, axis) => axis === i ? 22 : 0));
    const end = [origin[0] + x, origin[1] - y];
    ctx.beginPath();
    ctx.moveTo(...origin);
    ctx.lineTo(...end);
    ctx.strokeStyle = '#8aa09e';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#445f63';
    ctx.fillText(name, end[0] + Math.sign(x) * 4 - 3, end[1] - Math.sign(y) * 4 + 3);
  });
  ctx.strokeStyle = '#dce7e3';
  ctx.strokeRect(.5, .5, w - 1, h - 1);
}

function drawSignal(key, signal, route, time, focus, detail) {
  const canvas = $('canvas-' + key);
  const {width: w, height: h} = canvas.getBoundingClientRect();
  if (!w || !h) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(w * ratio);
  const height = Math.round(h * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  drawAxes(ctx, w, h);
  const n = signal.ends.findLastIndex(t => t <= time);
  if (n < 0) {
    $('sample-' + key).textContent = 'First complete block ends at 399';
    return;
  }

  const copies = detail && focus !== 'all' ? [Number(focus)] : [0, 1, 2, 3];
  const {center, radius} = cameraBounds(signal, route, time, copies, detail);
  const scale = Math.min(w - 58, h - 60) / (2 * radius);
  const toScreen = p => {
    const [x, y, z] = projectPoint(p.map((v, i) => v - center[i]));
    return [w / 2 + x * scale, h / 2 - y * scale, z];
  };
  const frames = signal.coords[route];
  const recent = signal.ends.findIndex(t => t > time - trailLength);
  const first = detail ? recent : 0;
  const emphasized = copy => focus === 'all' || Number(focus) === copy;
  const order = copies.sort((a, b) => Number(emphasized(a)) - Number(emphasized(b)));
  const heads = [];
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const copy of order) {
    const active = emphasized(copy);
    const points = frames.slice(first, n + 1).map(frame => toScreen(frame[copy]));
    for (let j = 1; j < points.length; j++) {
      const age = time - signal.ends[first + j];
      const freshness = Math.max(0, 1 - age / trailLength);
      ctx.globalAlpha = active ? .09 + .64 * freshness : .06 + .20 * freshness;
      ctx.lineWidth = active ? 1.1 + 1.6 * freshness : 1;
      ctx.strokeStyle = pathColors[copy];
      ctx.beginPath();
      ctx.moveTo(points[j - 1][0], points[j - 1][1]);
      ctx.lineTo(points[j][0], points[j][1]);
      ctx.stroke();
    }
    heads.push({copy, point: points.at(-1), active});
  }

  // Draw heads after every trail so revisiting the same region never hides the current state.
  heads.sort((a, b) => Number(a.active) - Number(b.active) || a.point[2] - b.point[2]);
  for (const {copy, point: [x, y], active} of heads) {
    ctx.globalAlpha = active ? 1 : .55;
    ctx.beginPath();
    ctx.arc(x, y, active ? 6.5 : 4.5, 0, 2 * Math.PI);
    ctx.fillStyle = pathColors[copy];
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    if (active && focus !== 'all') {
      ctx.font = '700 12px system-ui';
      ctx.lineWidth = 3;
      ctx.strokeText(String(copy + 1), x + 10, y + 4);
      ctx.fillStyle = '#203f44';
      ctx.fillText(String(copy + 1), x + 10, y + 4);
    }
  }
  ctx.globalAlpha = 1;
  $('sample-' + key).textContent = key === 'selected_content'
    ? 'Last completed block ' + signal.times[n] + '–' + signal.ends[n]
    : 'Sample ' + signal.ends[n];
}

function drawPaths() {
  if (!pathData || $('self-models').hidden) return;
  const start = $('pathLife').value;
  const route = $('pathArm').value;
  const time = Number($('pathTime').value);
  const detail = $('pathView').value === 'recent';
  const genre = pathProfiles.text_spans.findLast(([begin]) => begin <= time)[1];
  $('pathPosition').textContent = 'Observation ' + time.toLocaleString('en-US') + ' / 2,999';
  $('pathGenre').textContent = genre === 'expository' ? 'Expository text' : genre[0].toUpperCase() + genre.slice(1);
  $('pathTime').setAttribute('aria-valuetext', 'Observation ' + time + ', ' + genre);
  for (const key of pathSignals) drawSignal(key, pathData[start][key], route, time, $('pathCopy').value, detail);
}

function queuePaths() {
  if (pathFrame) return;
  pathFrame = requestAnimationFrame(() => {
    pathFrame = null;
    drawPaths();
  });
}

function updateAxisNotes() {
  if (!pathProfiles) return;
  const start = $('pathLife').value;
  $('pathAxisStart').textContent = start;
  for (const key of pathSignals) {
    const pct = pathData[start][key].variance_retained * 100;
    $('variance-' + key).textContent = pct.toFixed(1) + '% of original variation shown';
  }
  $('pathAxes').querySelectorAll('tbody').forEach(body => body.hidden = body.dataset.start !== start);
}

function pausePaths() {
  if (pathTimer) clearInterval(pathTimer);
  pathTimer = null;
  $('pathPlay').textContent = Number($('pathTime').value) >= 2999 ? 'Replay' : 'Play';
}

function initializePaths(projections, profiles) {
  for (const signals of Object.values(projections)) {
    for (const [key, signal] of Object.entries(signals)) {
      signal.ends = signal.times.map(t => key === 'selected_content' ? t + 99 : t);
      signal.radius = Math.max(1e-5, ...Object.values(signal.coords).flat(2).map(p => Math.hypot(...p)));
    }
  }
  pathData = projections;
  pathProfiles = profiles;
  for (const [i, [begin, label]] of profiles.text_spans.entries()) {
    const end = profiles.text_spans[i + 1]?.[0] ?? 3000;
    const span = document.createElement('span');
    span.style.flex = String(end - begin);
    span.style.background = textColors[label];
    span.title = label + ' · observations ' + begin + '–' + (end - 1);
    $('pathGenres').append(span);
  }
  updateAxisNotes();
  queuePaths();
}

Promise.all([data('recorded_paths'), data('path_axis_profiles')])
  .then(([projections, profiles]) => initializePaths(projections, profiles)).catch(showError);
for (const id of ['pathArm', 'pathCopy', 'pathView']) $(id).addEventListener('change', queuePaths);
$('pathLife').addEventListener('change', () => { updateAxisNotes(); queuePaths(); });
$('pathTime').addEventListener('input', () => { pausePaths(); queuePaths(); });
$('pathPlay').onclick = () => {
  if (pathTimer) return pausePaths();
  if (Number($('pathTime').value) >= 2999) $('pathTime').value = 300;
  $('pathPlay').textContent = 'Pause';
  queuePaths();
  pathTimer = setInterval(() => {
    if ($('self-models').hidden || document.hidden) return pausePaths();
    const time = Math.min(2999, Number($('pathTime').value) + 10);
    $('pathTime').value = time;
    queuePaths();
    if (time === 2999) pausePaths();
  }, 85);
};
$('pathJump').onclick = () => { $('pathTime').value = 820; pausePaths(); queuePaths(); };
$('pathReset').onclick = () => { Object.assign(view, initialView); queuePaths(); };
$('pathRotate').onclick = () => { view.yaw += Math.PI / 4; queuePaths(); };
for (const canvas of document.querySelectorAll('.path-panel canvas')) {
  let drag;
  canvas.onpointerdown = event => {
    if (!event.isPrimary) return;
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove = event => {
    if (!drag || !event.isPrimary) return;
    view.yaw += (event.clientX - drag[0]) * .01;
    view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + (event.clientY - drag[1]) * .01));
    drag = [event.clientX, event.clientY];
    queuePaths();
  };
  canvas.onlostpointercapture = () => { drag = null; };
}
window.addEventListener('resize', queuePaths);


function showError(e){console.error(e);let p=document.createElement('p');p.className='surface';p.textContent='An evidence file could not be loaded. Serve the directory over HTTP or open the hosted study. '+e.message;$('main').append(p);reportHeight()}
