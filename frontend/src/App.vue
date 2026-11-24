<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const active = ref('overview')
const loading = ref(false)
const error = ref('')
const message = ref('')
const overview = ref({datasets:0,samples:0,models:0,runs:0,active_runs:0})
const datasets = ref([])
const models = ref([])
const runs = ref([])
const selectedDataset = ref(null)
const datasetSamples = ref([])
const selectedRun = ref(null)
const selectedResult = ref(null)
const progress = ref(null)
const testCases = ref(null)
const caseFilter = ref('all')
const casePage = ref(1)
const casePageSize = ref(20)
const comparisons = ref([])
const compareIds = ref([])
const uploadFile = ref(null)
const offlineFile = ref(null)
const datasetForm = ref({name:'自定义数据集',version:'v1',description:''})
const modelForm = ref({name:'',kind:'openai',base_url:'http://127.0.0.1:8001/v1',api_model:'',api_mode:'completion',api_key_env:'',system_prompt:'You are an expert binary decompiler. Output only valid C source code.'})
const runForm = ref({dataset_id:null,model_id:null,optimizations:['O0','O1','O2','O3'],sample_limit:0,max_tokens:1024,temperature:0,inference_concurrency:0,compile_concurrency:64,generation_mode:'online',model_path:'',batch_size:32,prompt_template:'# This is the assembly code:\n{assembly}\n# What is the source code?\n'})
const offlineForm = ref({dataset_id:null,name:'离线结果导入'})

const nav = [
  ['overview','◫','评测总览'],['datasets','▤','数据集管理'],
  ['models','◇','模型管理'],['newrun','⊕','新建评测'],
  ['runs','≡','评测任务'],['compare','▥','模型对比']
]
const titles = {'overview':'评测总览','datasets':'数据集管理','models':'模型管理','newrun':'新建评测任务','runs':'评测任务','compare':'模型对比'}
const statusLabel = {'PASS':'通过','FAIL':'失败','NOT_RUN_COMPILE_ERROR':'编译失败未运行','NOT_RUN_DIAG_ERROR':'诊断无法运行','NOT_EXECUTED':'未执行','queued':'排队中','running':'评测中','cancel_requested':'停止中','completed':'已完成','completed_with_errors':'部分基础设施失败','cancelled':'已停止','interrupted':'已中断','TEST_PASS':'测试通过','WRONG_ANSWER':'结果错误','COMPILE_ERROR':'编译失败','COMPILE_TIMEOUT':'编译超时','RUNTIME_ERROR':'运行异常','TIMEOUT':'执行超时','INFRA_ERROR':'沙箱错误','INFERENCE_ERROR':'模型接口错误','MODEL_EMPTY':'空输出','GENERATED':'已生成待编译','COMPILING':'编译测试中','PASS':'通过','ASSERT_FAILURE':'断言失败','INPUT_TOO_LONG':'输入超长'}
const statusText = s => statusLabel[s] || s || '—'
const percent = v => v === null || v === undefined ? 'N/A' : `${Number(v).toFixed(1)}%`
const rate = v => v === null || v === undefined ? 0 : Math.max(0,Math.min(100,Number(v)))
const dateTime = d => d ? new Date(d).toLocaleString('zh-CN') : '—'
const isActiveRun = status => ['queued','running','cancel_requested'].includes(status)
const selectedRuns = computed(() => runs.value.filter(r=>compareIds.value.includes(r.id)))
const completedRuns = computed(() => runs.value.filter(r=>['completed','completed_with_errors'].includes(r.status)))
const deletableRuns = computed(() => runs.value.filter(r=>!isActiveRun(r.status)))
const prog = computed(() => progress.value || selectedRun.value?.progress || {})
const progressItems = computed(() => {
  const p = prog.value
  return [
    {label:'等待推理', value:p.waiting_inference},
    {label:'正在推理', value:p.inferring},
    {label:'已生成代码', value:p.inference_done},
    {label:'等待编译', value:p.waiting_compile},
    {label:'正在编译', value:p.compiling},
    {label:'已完成评测', value:p.evaluated},
    {label:'失败样本', value:p.failed},
  ]
})
const testStatusText = s => ({'PASS':'通过','ASSERT_FAILURE':'断言失败','WRONG_ANSWER':'结果错误','RUNTIME_ERROR':'运行异常','TIMEOUT':'超时','INFRA_ERROR':'沙箱错误','INPUT_TOO_LONG':'输入超长'}[s] || s || '—')
const testTagClass = s => s==='PASS' ? 'TEST_PASS' : s==='TIMEOUT' ? 'TIMEOUT' : s==='ASSERT_FAILURE' ? 'WRONG_ANSWER' : 'RUNTIME_ERROR'
const fmtRate = v => (v===undefined||v===null) ? '—' : `${Number(v).toFixed(2)} 样本/s`
const fmtElapsed = v => {
  if(v===undefined||v===null) return '—'
  const n = Number(v)
  return n>=60 ? `${Math.floor(n/60)}分${Math.round(n%60)}秒` : `${n.toFixed(1)}秒`
}
const runDone = r => r.progress ? (r.progress.evaluated||0)+(r.progress.inference_failed||0) : (r.completed||0)
const runPct = r => r.total ? Math.round(100*runDone(r)/r.total) : 0

async function request(path, opts={}) {
  const r = await fetch('/api'+path, opts)
  let data
  try { data = await r.json() } catch { throw new Error(`HTTP ${r.status}`) }
  if (!r.ok) {
    const detail = data.detail || `HTTP ${r.status}`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return data
}
async function safely(fn) {
  error.value = '';message.value = '';loading.value = true
  try { await fn() } catch(e) { error.value = e.message || String(e) }
  finally {loading.value = false}
}
async function refresh() {
  const [o,d,m,r] = await Promise.all([request('/overview'),request('/datasets'),request('/models'),request('/runs')])
  overview.value = o;datasets.value = d;models.value = m;runs.value = r
  if (!runForm.value.dataset_id && d.length) runForm.value.dataset_id=d[0].id
  if (!offlineForm.value.dataset_id && d.length) offlineForm.value.dataset_id=d[0].id
  if (!runForm.value.model_id && m.length) runForm.value.model_id=m.find(x=>x.kind==='fixture')?.id || m[0].id
}
function setPage(p){active.value=p; selectedResult.value=null}
async function showDataset(d) {
  await safely(async()=>{
    selectedDataset.value=d
    datasetSamples.value=await request(`/datasets/${d.id}/samples`)
    setPage('datasets')
  })
}
async function uploadDataset() {
  await safely(async()=>{
    if (!uploadFile.value) throw new Error('请先选择 .jsonl 文件')
    const fd = new FormData()
    for (const k of ['name','version','description']) fd.append(k,datasetForm.value[k])
    fd.append('file',uploadFile.value)
    const r=await request('/datasets/import',{method:'POST',body:fd})
    await refresh()
    message.value=`成功导入 ${r.sample_count} 个样本（数据集 #${r.dataset_id}）`
    uploadFile.value=null
  })
}
async function addModel() {
  await safely(async()=>{
    const m=await request('/models',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(modelForm.value)})
    await refresh()
    message.value=`模型已注册（#${m.model_id}）`
    modelForm.value.name='';modelForm.value.api_model=''
  })
}
async function testModel(m) {
  await safely(async()=>{
    const result=await request(`/models/${m.id}/test`,{method:'POST'})
    if(!result.ok)throw new Error(result.message)
    message.value=result.preview ? `连接成功。预览：${result.preview.slice(0,80)}` : (result.message || '模型连接正常')
  })
}
async function startRun() {
  await safely(async()=>{
    if (!runForm.value.optimizations.length) throw new Error('请至少选择一个优化等级')
    const batch=runForm.value.generation_mode==='offline_batch'
    if(batch && !runForm.value.model_path.trim()) throw new Error('离线批量模式需要填写本地模型路径')
    const path=batch?'/runs/batch':'/runs'
    const created=await request(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(runForm.value)})
    await refresh();await loadRun(created.run_id)
    active.value='runs';message.value=`任务 #${created.run_id} 已创建${batch?'（离线批量生成）':''}`
  })
}
async function offlineRun() {
  await safely(async()=>{
    if(!offlineFile.value)throw new Error('请选择离线预测 JSONL')
    const fd=new FormData()
    fd.append('file',offlineFile.value)
    fd.append('dataset_id',offlineForm.value.dataset_id)
    fd.append('name',offlineForm.value.name)
    const created=await request('/runs/offline',{method:'POST',body:fd})
    await refresh();await loadRun(created.run_id)
    active.value='runs';message.value=`离线评测任务 #${created.run_id} 已创建`
  })
}
async function loadRun(id) {
  selectedRun.value=await request(`/runs/${id}`)
  progress.value=selectedRun.value.progress || null
  selectedResult.value=null
}
async function fetchProgress() {
  if(!selectedRun.value || !isActiveRun(selectedRun.value.run.status))return
  try { progress.value=await request(`/runs/${selectedRun.value.run.id}/progress`) } catch(e) { /* transient */ }
}
async function showRun(id) {await safely(async()=>{await loadRun(id);active.value='runs'})}
async function showResult(id) {
  await safely(async()=>{
    selectedResult.value=await request(`/results/${id}`)
    testCases.value=null;caseFilter.value='all';casePage.value=1
    await loadTestCases(1)
  })
}
async function loadTestCases(page=1) {
  if(!selectedResult.value || !selectedRun.value)return
  const rid=selectedRun.value.run.id
  const status=caseFilter.value==='all'?'':caseFilter.value
  const data=await request(`/runs/${rid}/results/${selectedResult.value.id}/test-cases?status=${encodeURIComponent(status)}&page=${page}&page_size=${casePageSize.value}`)
  testCases.value=data;casePage.value=page
}
function setCaseFilter(f) {
  caseFilter.value=f;loadTestCases(1).catch(e=>{error.value=e.message || String(e)})
}
function changeCasePage(delta) {
  if(!testCases.value)return
  const max=Math.max(1,Math.ceil(testCases.value.total/casePageSize.value))
  const next=Math.min(max,Math.max(1,casePage.value+delta))
  loadTestCases(next).catch(e=>{error.value=e.message || String(e)})
}
function exportCases(format) {
  if(!selectedResult.value || !selectedRun.value)return
  const url=`/api/runs/${selectedRun.value.run.id}/results/${selectedResult.value.id}/test-cases?export=${format}`
  window.open(url,'_blank')
}
const caseTagClass = s => s==='PASS' ? 'TEST_PASS' : s==='FAIL' ? 'WRONG_ANSWER' : s&&s.startsWith('NOT_RUN') ? 'INFRA_ERROR' : 'TIMEOUT'
const caseCoverage = computed(() => {
  const s=testCases.value?.summary
  if(!s || !s.cases_declared)return null
  return s.coverage
})
const diagnosticConsistent = computed(() => testCases.value?.summary?.diagnostic_consistent)
const caseMaxPage = computed(() => testCases.value ? Math.max(1,Math.ceil(testCases.value.total/casePageSize.value)) : 1)
const bottleneckText = computed(() => {
  const p = prog.value
  if(!p || !p.total || !['running','cancel_requested'].includes(p.status)) return ''
  const vm = p.vllm_metrics
  if(vm && vm.num_requests_waiting > 0) return `vLLM 排队中（waiting=${vm.num_requests_waiting}），GPU 可能在处理其它任务或被限流。`
  if((p.waiting_inference || 0) > 0 && (p.inferring || 0) === 0)
    return `仍有 ${p.waiting_inference} 个样本待推理，但当前没有在途请求：检查调度/连接池/取消状态。`
  if((p.waiting_inference || 0) === 0 && ((p.compiling || 0) > 0 || (p.waiting_compile || 0) > 0))
    return `推理已无待处理样本，GPU 空闲；当前瓶颈是 Docker 编译/测试（编译中 ${p.compiling||0}，等待 ${p.waiting_compile||0}）。`
  return ''
})
async function runAction(id,action) {
  await safely(async()=>{
    await request(`/runs/${id}/${action}`,{method:'POST'})
    await refresh();await loadRun(id)
    message.value=action==='retry'?'任务已重新排队':'已提交停止请求'
  })
}
async function referenceCheck(d) {
  if(!window.confirm(`对「${d.name}」的参考源码执行同一套 Docker 编译/测试（O0–O3，全部样本）？\n这会创建一个独立的参考基线任务，可能耗时较长。`))return
  await safely(async()=>{
    const created=await request(`/datasets/${d.id}/reference-check`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({optimizations:['O0','O1','O2','O3'],sample_limit:0})})
    await refresh();await loadRun(created.run_id);active.value='runs'
    message.value=`参考源码基线任务 #${created.run_id} 已创建（失败表示数据集/环境问题）`
  })
}
async function deleteRun(id) {
  if(!window.confirm(`确定删除任务 #${id}？\n该任务的逐样本结果、日志和离线预测文件会一并删除，且不可恢复。`))return
  await safely(async()=>{
    await request(`/runs/${id}`,{method:'DELETE'})
    if(selectedRun.value?.run.id===id){selectedRun.value=null;selectedResult.value=null;progress.value=null}
    compareIds.value=compareIds.value.filter(x=>x!==id)
    comparisons.value=comparisons.value.filter(x=>x.id!==id)
    await refresh()
    message.value=`任务 #${id} 已删除`
  })
}
async function deleteFinishedRuns() {
  await safely(async()=>{
    const count=deletableRuns.value.length
    if(!count)throw new Error('没有可删除的已结束任务')
    if(!window.confirm(`确定删除全部 ${count} 个已结束任务（已完成 / 已停止 / 已中断）？\n该操作不可恢复。`))return
    const result=await request('/runs',{method:'DELETE'})
    selectedRun.value=null;selectedResult.value=null;progress.value=null
    compareIds.value=[];comparisons.value=[]
    await refresh()
    message.value=`已删除 ${result.deleted} 个任务`
  })
}
async function compare() {
  await safely(async()=>{
    if(!compareIds.value.length)throw new Error('请至少选择一个任务')
    comparisons.value=await request('/compare?run_ids='+compareIds.value.join(','))
  })
}
function toggleOpt(opt) {
  const list=runForm.value.optimizations
  runForm.value.optimizations=list.includes(opt)?list.filter(x=>x!==opt):[...list,opt]
}
function toggleCompare(id) {
  if(compareIds.value.includes(id))compareIds.value=compareIds.value.filter(x=>x!==id)
  else if(compareIds.value.length<8) compareIds.value=[...compareIds.value,id]
}
let poller, progressPoller
onMounted(async()=>{
  await safely(refresh)
  poller=setInterval(async()=>{
    try {
      if(runs.value.some(r=>isActiveRun(r.status)) || active.value==='overview'){
        await refresh()
        if(selectedRun.value && isActiveRun(selectedRun.value.run.status)){
          const id=selectedRun.value.run.id
          selectedRun.value=await request(`/runs/${id}`)
          progress.value=selectedRun.value.progress || progress.value
        }
      }
    }catch(e){ /* individual errors appear on explicit actions */ }
  },2500)
  progressPoller=setInterval(fetchProgress,1000)
})
onUnmounted(()=>{clearInterval(poller);clearInterval(progressPoller)})
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-glyph">⌘</span><div><strong>DecompileEval</strong><small>RESEARCH PLATFORM · MVP</small></div></div>
      <div class="sidebar-label">工作空间</div>
      <button v-for="item in nav" :key="item[0]" class="nav-item" :class="{active:active===item[0]}" @click="setPage(item[0])">
        <span class="nav-icon">{{ item[1] }}</span><span>{{ item[2] }}</span>
        <span v-if="item[0]==='runs'&&overview.active_runs" class="count-dot">{{overview.active_runs}}</span>
      </button>
      <div class="sidebar-bottom">
        <div class="sandbox-hint"><span class="status-led"></span><span>本地开发模式</span></div>
        <small>演示数据仅用于验证流程。<br>不可作为真实模型排行榜。</small>
        <a href="http://127.0.0.1:8010/docs" target="_blank" rel="noopener">API 文档 ↗</a>
      </div>
    </aside>

    <main class="main">
      <header class="topbar"><div class="breadcrumb">工作空间 <span>/</span> {{titles[active]}}</div><div class="topright"><span class="pill soft">v0.5 · 真实TCP</span><span class="avatar">DE</span></div></header>
      <section class="content">
        <div v-if="error" class="notice error"><strong>操作失败</strong>　{{error}}<button @click="error=''">×</button></div>
        <div v-if="message" class="notice success">{{message}}<button @click="message=''">×</button></div>

        <template v-if="active==='overview'">
          <div class="heading-row"><div><span class="eyebrow">OVERVIEW</span><h1>评测总览</h1><p>集中管理反编译数据、模型调用与功能正确性评测。</p></div><button class="primary" @click="setPage('newrun')">＋ 新建评测</button></div>
          <div class="stat-grid">
            <div class="stat" v-for="s in [{label:'数据集',value:overview.datasets,icon:'▤',hint:'可用 Benchmark'},{label:'评测样本',value:overview.samples,icon:'≡',hint:'函数级测试用例'},{label:'注册模型',value:overview.models,icon:'◇',hint:'API / 固定输出 / 离线'},{label:'评测任务',value:overview.runs,icon:'▥',hint:overview.active_runs+' 个任务进行中'}]" :key="s.label"><div class="stat-top"><span>{{s.label}}</span><span class="stat-icon">{{s.icon}}</span></div><div class="stat-value">{{s.value}}</div><small>{{s.hint}}</small></div>
          </div>
          <div class="two-columns">
            <div class="panel"><div class="panel-head"><div><h2>最近评测任务</h2><p>实时跟踪已提交任务</p></div><button class="linklike" @click="setPage('runs')">查看全部 →</button></div>
              <div class="table-wrap"><table><thead><tr><th>任务</th><th>模型</th><th>进度</th><th>状态</th></tr></thead><tbody>
                <tr v-for="run in runs.slice(0,6)" :key="run.id" class="click-row" @click="showRun(run.id)"><td>#{{run.id}}</td><td class="name-cell">{{run.model_name}}</td><td>{{run.completed}} / {{run.total}}</td><td><span class="tag" :class="run.status">{{statusText(run.status)}}</span></td></tr>
                <tr v-if="!runs.length"><td colspan="4" class="empty">暂无任务，先创建一次演示评测。</td></tr>
              </tbody></table></div>
            </div>
            <div class="panel quick-start"><div class="panel-head"><div><h2>快速开始</h2><p>约三步完成一次完整流程</p></div><span class="pill">GET STARTED</span></div>
              <div class="step"><span>01</span><div><strong>选择内置演示数据集</strong><small>包含汇编、参考 C 代码及测试用例</small></div></div>
              <div class="step"><span>02</span><div><strong>使用演示固定输出模型</strong><small>无需 API Key，非真实大模型推理</small></div></div>
              <div class="step"><span>03</span><div><strong>在 Docker 沙箱中编译测试</strong><small>查看 RC、RE、TCP 和失败日志</small></div></div>
              <button class="secondary wide" @click="setPage('newrun')">打开任务配置 →</button>
            </div>
          </div>
        </template>

        <template v-if="active==='datasets'">
          <div class="heading-row"><div><span class="eyebrow">DATASETS</span><h1>数据集管理</h1><p>导入统一 JSONL 格式的反编译基准样本。</p></div></div>
          <div class="two-columns col-left-wide">
            <div class="panel"><div class="panel-head"><div><h2>已导入数据集</h2><p>点击查看样本列表</p></div></div>
              <div class="dataset-cards"><button v-for="d in datasets" :key="d.id" class="dataset-item" :class="{selected:selectedDataset?.id===d.id}" @click="showDataset(d)"><span class="dataset-symbol">▤</span><span class="dataset-info"><strong>{{d.name}}</strong><small>{{d.version}} · {{d.description}}</small></span><span class="right-count">{{d.sample_count}}<small>样本</small></span></button></div>
              <div v-if="selectedDataset" class="dataset-actions"><button class="secondary" :disabled="loading" @click="referenceCheck(selectedDataset)">验证参考源码（同一沙箱）</button><span class="tip">结果保存为独立任务；参考源码失败说明数据集/环境有问题，不用于提高模型 RE。</span></div>
              <template v-if="selectedDataset"><div class="subheading">{{selectedDataset.name}} · 样本预览</div><div class="table-wrap"><table><thead><tr><th>样本 ID</th><th>问题</th><th>优化</th><th>编译器</th></tr></thead><tbody><tr v-for="s in datasetSamples" :key="s.id"><td class="mono">{{s.sample_key}}</td><td>{{s.problem_id}}</td><td><span class="pill">{{s.optimization}}</span></td><td>{{s.compiler}}</td></tr></tbody></table></div></template>
            </div>
            <div class="panel form-panel"><div class="panel-head"><div><h2>导入 JSONL</h2><p>包含汇编、参考源码和测试用例</p></div></div>
              <label>数据集名称<input v-model="datasetForm.name" placeholder="HumanEval-Decompile" /></label>
              <label>版本<input v-model="datasetForm.version" placeholder="v1" /></label>
              <label>说明<input v-model="datasetForm.description" placeholder="数据集介绍" /></label>
              <label>选择 JSONL 文件<input type="file" accept=".jsonl" @change="uploadFile=$event.target.files[0]" /></label>
              <button class="primary wide" :disabled="loading" @click="uploadDataset">导入数据集</button>
              <p class="tip">文件上限 64 MB、12000 条样本；支持 C/C++，必须附带 tests。</p>
            </div>
          </div>
        </template>

        <template v-if="active==='models'">
          <div class="heading-row"><div><span class="eyebrow">MODELS</span><h1>模型管理</h1><p>统一管理 OpenAI 兼容推理接口，密钥只从服务端环境变量读取。</p></div></div>
          <div class="two-columns">
            <div class="panel"><div class="panel-head"><div><h2>已注册模型</h2><p>{{models.length}} 个可选模型</p></div></div>
              <div v-for="m in models" :key="m.id" class="model-item"><div class="model-dot">◇</div><div class="model-info"><strong>{{m.name}}</strong><small>{{m.kind==='fixture'?'演示固定输出（非 LLM）':m.kind==='offline'?'离线预测文件':m.api_model+' · '+m.base_url}}</small></div><button class="ghost" :disabled="loading" @click="testModel(m)">检查</button></div>
            </div>
            <div class="panel form-panel"><div class="panel-head"><div><h2>添加模型</h2><p>OpenAI / vLLM 兼容 Chat Completions API</p></div></div>
              <label>显示名称<input v-model="modelForm.name" placeholder="例如 DeSIR-6.7B" /></label>
              <label>API Base URL<input v-model="modelForm.base_url" placeholder="http://127.0.0.1:8001/v1" /></label>
              <label>接口类型 <select v-model="modelForm.api_mode"><option value="completion">Completion（LLM4Decompile v1.5）</option><option value="chat">Chat Completion</option></select></label>
            <label>服务端模型 ID<input v-model="modelForm.api_model" placeholder="例如 my-decompiler" /></label>
              <label>API Key 对应环境变量名（可空）<input v-model="modelForm.api_key_env" placeholder="OPENAI_API_KEY" /></label>
              <button class="primary wide" :disabled="loading" @click="addModel">注册模型</button>
              <p class="tip">不要在这里粘贴实际 API Key。若使用 vLLM 本地服务，环境变量名可留空。</p>
            </div>
          </div>
        </template>

        <template v-if="active==='newrun'">
          <div class="heading-row"><div><span class="eyebrow">NEW EXPERIMENT</span><h1>新建评测任务</h1><p>选择模型和数据集，批量生成 C 代码并执行功能验证。</p></div></div>
          <div class="two-columns">
            <div class="panel form-panel"><div class="panel-head"><div><h2>在线模型评测</h2><p>完整流程：推理 → 编译 → 测试</p></div></div>
              <label>评测数据集<select v-model.number="runForm.dataset_id"><option v-for="d in datasets" :value="d.id" :key="d.id">{{d.name}}（{{d.sample_count}}）</option></select></label>
              <label>评测模型<select v-model.number="runForm.model_id"><option v-for="m in models" :value="m.id" :key="m.id" :disabled="m.kind==='offline'">{{m.name}}</option></select></label>
              <label>生成方式<select v-model="runForm.generation_mode"><option value="online">在线 HTTP（vLLM /v1/completions）</option><option value="offline_batch">离线批量（本地 vLLM LLM.generate）</option></select></label>
              <template v-if="runForm.generation_mode==='offline_batch'">
                <label>本地模型路径<input v-model="runForm.model_path" placeholder="/path/to/llm4decompile-1.3b-v1.5 或 HF id" /></label>
                <label>批大小（独立 Prompt）<input v-model.number="runForm.batch_size" type="number" min="1" max="4096" /></label>
                <p class="tip">离线批量模式需要后端环境安装 vllm；Prompt 仍是每个样本独立生成，不会拼接函数。</p>
              </template>
              <div class="field-label">优化等级</div><div class="check-row"><label v-for="o in ['O0','O1','O2','O3']" :key="o" class="check"><input type="checkbox" :checked="runForm.optimizations.includes(o)" @change="toggleOpt(o)"/>{{o}}</label></div>
              <div class="two-inputs"><label>样本数量上限（0=全部）<input v-model.number="runForm.sample_limit" type="number" min="0" max="12000" /></label><label>最大 Token 数<input v-model.number="runForm.max_tokens" type="number" min="128" max="8192" /></label></div>
              <div class="two-inputs"><label>推理并发数（0=AUTO）<input v-model.number="runForm.inference_concurrency" type="number" min="0" max="512" title="0 = AUTO 自适应持续投递" /></label><label>编译并发数<input v-model.number="runForm.compile_concurrency" type="number" min="1" max="128" /></label></div>
              <label>Prompt 模板<textarea v-model="runForm.prompt_template" rows="5"></textarea></label>
              <p class="tip">使用 <code>{assembly}</code> 占位符。不能向模型输入参考 C 代码或测试用例。推理与编译并发数受服务端上限约束。</p>
              <button class="primary wide" :disabled="loading" @click="startRun">▶ {{ runForm.generation_mode==='offline_batch' ? '创建并执行离线批量评测' : '创建并执行评测' }}</button>
            </div>
            <div class="panel form-panel"><div class="panel-head"><div><h2>离线预测结果评测</h2><p>无需调用模型 API，直接验证已有 C 代码</p></div></div>
              <label>评测数据集<select v-model.number="offlineForm.dataset_id"><option v-for="d in datasets" :value="d.id" :key="d.id">{{d.name}}</option></select></label>
              <label>结果组名称<input v-model="offlineForm.name" placeholder="Model-A offline" /></label>
              <label>预测结果 JSONL<input type="file" accept=".jsonl" @change="offlineFile=$event.target.files[0]" /></label>
              <div class="code-example"><code v-pre>{"sample_id":"demo_sum_O0",<br> "prediction":"int sum_two(...) {...}"}</code></div>
              <p class="tip">当前版本要求离线文件覆盖所选数据集全部样本，且 sample_id 必须一致。</p>
              <button class="secondary wide" :disabled="loading" @click="offlineRun">导入并执行评测</button>
              <div class="divider-gap"></div><h3>评测指标</h3>
              <div class="metric-note"><strong>RC</strong><span>重新编译成功率</span></div><div class="metric-note"><strong>RE</strong><span>全部测试通过的样本占比</span></div><div class="metric-note"><strong>TCP</strong><span>通过的测试用例占比</span></div>
            </div>
          </div>
        </template>

        <template v-if="active==='runs'">
          <div class="heading-row"><div><span class="eyebrow">EXPERIMENT RUNS</span><h1>评测任务</h1><p>查看任务进度、实验指标和逐样本错误分析。</p></div><button class="primary" @click="setPage('newrun')">＋ 新建评测</button></div>
          <div class="panel"><div class="panel-head"><div><h2>全部任务</h2><p>选择任意一条记录查看详细结果</p></div><div class="actions"><button class="ghost danger" :disabled="!deletableRuns.length" @click="deleteFinishedRuns">删除已结束任务</button><button class="ghost" @click="safely(refresh)">刷新 ↻</button></div></div>
            <div class="table-wrap"><table><thead><tr><th>任务编号</th><th>模型</th><th>数据集</th><th>进度</th><th>状态</th><th>创建时间</th><th>操作</th></tr></thead><tbody>
              <tr v-for="r in runs" :key="r.id" :class="{'row-selected':selectedRun?.run.id===r.id}"><td class="mono">#{{r.id}}</td><td class="name-cell">{{r.model_name}}</td><td>{{r.dataset_name}}</td><td><div class="progress-mini"><span :style="{width:runPct(r)+'%'}"></span></div><small>{{runDone(r)}}/{{r.total}}</small></td><td><span class="tag" :class="r.status">{{statusText(r.status)}}</span></td><td class="time">{{dateTime(r.created_at)}}</td><td><div class="actions"><button class="ghost" @click="showRun(r.id)">详情 →</button><button class="ghost danger" :disabled="isActiveRun(r.status)" :title="isActiveRun(r.status)?'请先停止任务再删除':''" @click="deleteRun(r.id)">删除</button></div></td></tr>
              <tr v-if="!runs.length"><td class="empty" colspan="7">暂无评测任务</td></tr>
            </tbody></table></div>
            <p class="tip">仅能删除已结束的任务；排队中/评测中的任务请先点击「停止任务」，待状态变为已停止后再删除。删除后逐样本结果、日志和离线预测文件均不可恢复。</p>
          </div>
          <div v-if="selectedRun" class="run-details">
            <div class="panel"><div class="panel-head"><div><h2>任务 #{{selectedRun.run.id}} · {{selectedRun.run.model_name}}</h2><p>{{selectedRun.run.dataset_name}} · {{statusText(selectedRun.run.status)}}</p></div><div class="actions"><button v-if="isActiveRun(selectedRun.run.status)" class="ghost danger" @click="runAction(selectedRun.run.id,'cancel')">停止任务</button><button v-if="['interrupted','completed_with_errors','cancelled'].includes(selectedRun.run.status)" class="secondary" @click="runAction(selectedRun.run.id,'retry')">重试未完成/基础设施失败样本</button><button v-if="!isActiveRun(selectedRun.run.status)" class="ghost danger" @click="deleteRun(selectedRun.run.id)">删除任务</button></div></div>
              <div class="progress-pairs">
                <div class="progress-box">
                  <div class="progress-label"><span>推理进度</span><strong>{{prog.inference_done||0}} / {{prog.total||selectedRun.run.total}}</strong></div>
                  <div class="progress-wide"><span :style="{width:rate(100*(prog.inference_done||0)/(prog.total||selectedRun.run.total||1))+'%'}"></span></div>
                </div>
                <div class="progress-box">
                  <div class="progress-label"><span>编译测试进度</span><strong>{{prog.evaluated||0}} / {{prog.total||selectedRun.run.total}}</strong></div>
                  <div class="progress-wide"><span :style="{width:rate(100*(prog.evaluated||0)/(prog.total||selectedRun.run.total||1))+'%'}"></span></div>
                </div>
              </div>
              <div class="progress-grid"><div v-for="item in progressItems" :key="item.label" class="progress-cell"><small>{{item.label}}</small><strong>{{item.value ?? 0}}</strong></div></div>
              <div class="progress-meta">
                <span>推理并发 <b>{{prog.inference_concurrency === 0 ? 'AUTO (0)' : (prog.inference_concurrency || '—')}}</b><small v-if="prog.auto_max_inflight"> · 当前上限 {{prog.auto_limit || '—'}} / {{prog.auto_max_inflight}}</small></span>
                <span>编译并发 <b>{{prog.compile_concurrency || '—'}}</b><small v-if="prog.compile_concurrency_requested && prog.compile_concurrency_requested !== prog.compile_concurrency"> · 请求 {{prog.compile_concurrency_requested}}</small></span>
                <span>在途请求 <b>{{prog.in_flight || 0}}</b></span>
                <span>推理请求数 <b>{{prog.inference_requests || 0}}</b></span>
                <span>推理吞吐 <b>{{fmtRate(prog.inference_rate)}}</b></span>
                <span>评测吞吐 <b>{{fmtRate(prog.eval_rate)}}</b></span>
                <span>Token/s <b>{{prog.token_rate || 0}}</b></span>
                <span>总 Token/s <b>{{prog.total_token_rate || 0}}</b></span>
                <span>输入超长 <b>{{prog.input_too_long || 0}}</b></span>
                <span>GPU 推理总耗时 <b>{{prog.inference_ms_sum || 0}} ms</b></span>
                <span>gcc 编译 <b>{{prog.compile_ms_sum || 0}} ms</b></span>
                <span>沙箱总耗时 <b>{{prog.sandbox_ms_sum || 0}} ms</b></span>
                <span>总耗时 <b>{{fmtElapsed(prog.elapsed_seconds)}}</b></span>
                <span v-if="prog.generation_mode==='offline_batch'">离线批量 · 批大小 <b>{{prog.batch_size || '—'}}</b></span>
                <span v-if="prog.vllm_metrics">GPU 运行/等待 <b>{{prog.vllm_metrics.num_requests_running ?? '—'}} / {{prog.vllm_metrics.num_requests_waiting ?? '—'}}</b></span>
                <span v-if="prog.vllm_metrics?.gpu_cache_usage_perc !== undefined">KV 缓存 <b>{{Math.round(prog.vllm_metrics.gpu_cache_usage_perc * 100)}}%</b></span>
              </div>
              <p v-if="selectedRun.run.config?.reference" class="tip"><b>参考源码基线任务：</b>失败代表数据集或评测环境问题，不能当作模型错误，也不能从模型 RE 分母中删除。</p>
              <p v-if="bottleneckText" class="tip"><b>瓶颈提示：</b>{{bottleneckText}}</p>
              <p v-if="selectedRun.run.error" class="danger-text">{{selectedRun.run.error}}</p>
              <div class="score-grid"><div v-for="score in [{name:'RC · 编译率',value:selectedRun.metrics.rc},{name:'RE · 执行率',value:selectedRun.metrics.re},{name:'TCP micro · 用例加权',value:selectedRun.metrics.tcp_micro},{name:'TCP macro · 函数平均',value:selectedRun.metrics.tcp_macro}]" :key="score.name" class="score"><div class="score-title">{{score.name}}</div><strong>{{percent(score.value)}}</strong><div class="score-track"><span :style="{width:rate(score.value)+'%'}"></span></div></div></div>
              <p class="tip">可统计样本 {{selectedRun.metrics.evaluated}} 个，基础设施/推理错误排除 {{selectedRun.metrics.excluded}} 个。RE 为官方完整 harness 全部通过的样本比例；TCP micro = 通过原子用例/有效原子用例，TCP macro = 每个可诊断函数通过率的平均。metric {{selectedRun.metrics.metric_version || 'legacy'}}。
                <span v-if="selectedRun.metrics.tcp_micro === null || selectedRun.metrics.tcp_micro === undefined"> 旧结果没有逐用例信息，TCP=N/A，需重新诊断评测。</span></p>
              <div class="progress-meta case-metrics">
                <span>诊断样本 <b>{{selectedRun.metrics.diagnosed_samples || 0}} / {{selectedRun.metrics.evaluated}}</b></span>
                <span>case 覆盖 <b>{{selectedRun.metrics.cases_passed || 0}} / {{selectedRun.metrics.cases_total || 0}}</b></span>
                <span>声明用例 <b>{{selectedRun.metrics.cases_declared || 0}}</b></span>
                <span>覆盖率 <b>{{selectedRun.metrics.case_coverage === null || selectedRun.metrics.case_coverage === undefined ? 'N/A' : percent(selectedRun.metrics.case_coverage)}}</b></span>
                <span>不可诊断样本 <b>{{selectedRun.metrics.unsupported_samples || 0}}</b></span>
                <span>未覆盖用例 <b>{{selectedRun.metrics.uncovered_cases || 0}}</b></span>
                <span>官方 RE 分母 <b>{{selectedRun.metrics.official_re_denominator ?? '—'}}</b></span>
              </div>
              <div class="progress-meta" v-if="selectedRun.metrics.failure_type_counts && Object.keys(selectedRun.metrics.failure_type_counts).length">
                <span v-for="(n,s) in selectedRun.metrics.failure_type_counts" :key="s">失败类型 <b>{{statusText(s)}}: {{n}}</b></span>
              </div>
              <h3>不同优化等级</h3><div class="opt-cards"><div v-for="(item,opt) in selectedRun.by_optimization" :key="opt" class="opt-card"><strong>{{opt}}</strong><small>{{item.evaluated}} 个有效样本</small><div>RC <b>{{percent(item.rc)}}</b></div><div>RE <b>{{percent(item.re)}}</b></div></div></div>
            </div>
            <div class="panel"><div class="panel-head"><div><h2>样本级评测结果</h2><p>点击查看源代码、模型输出、编译和运行日志</p></div></div>
              <div class="table-wrap"><table><thead><tr><th>样本 ID</th><th>优化</th><th>编译</th><th>官方测试</th><th>诊断用例</th><th>状态</th><th>诊断状态</th><th>耗时(推/编/测)</th><th></th></tr></thead><tbody><tr v-for="x in selectedRun.results" :key="x.id"><td class="mono">{{x.sample_key}}</td><td>{{x.optimization}}</td><td>{{x.compile_success===null?'—':x.compile_success?'✓':'✕'}}</td><td>{{x.tests_passed}} / {{x.tests_total}}</td><td>{{x.cases_passed||0}} / {{x.cases_total||0}} <small>({{x.cases_declared||0}})</small></td><td><span class="tag" :class="x.status">{{statusText(x.status)}}</span></td><td><span class="tag">{{x.test_case_status || '—'}}</span></td><td>{{x.inference_ms||0}} / {{x.compile_ms||0}} / {{x.test_ms||0}} ms</td><td><button class="ghost" @click="showResult(x.id)">查看 →</button></td></tr><tr v-if="!selectedRun.results.length"><td colspan="9" class="empty">尚无已完成样本</td></tr></tbody></table></div>
            </div>
          </div>
        </template>

        <template v-if="active==='compare'">
          <div class="heading-row"><div><span class="eyebrow">COMPARISON</span><h1>模型对比</h1><p>比较相同数据集、相同条件下的多个评测任务。</p></div></div>
          <div class="panel"><div class="panel-head"><div><h2>选择已完成的任务</h2><p>最多选择八个，必须属于同一数据集</p></div><button class="primary" @click="compare">生成对比报告</button></div>
            <div class="compare-picker"><label v-for="r in completedRuns" :key="r.id" class="compare-chip"><input type="checkbox" :checked="compareIds.includes(r.id)" @change="toggleCompare(r.id)"/><span>#{{r.id}} · {{r.model_name}}</span></label><p v-if="!completedRuns.length" class="empty">尚无可比较的已完成评测任务。</p></div>
          </div>
          <div v-if="comparisons.length" class="panel"><div class="panel-head"><div><h2>性能对比</h2><p>各指标以百分比展示，N/A 表示无有效样本</p></div></div>
            <div class="comparison-bars"><div v-for="r in comparisons" :key="r.id" class="comparison-row"><div class="comparison-name">{{r.model}} <small>Run #{{r.id}} · {{r.evaluated}} 样本</small></div><div class="bar-lines"><div class="bar-row"><span>RC</span><div class="bar-track"><i :style="{width:rate(r.rc)+'%',background:'#367dd6'}"></i></div><strong>{{percent(r.rc)}}</strong></div><div class="bar-row"><span>RE</span><div class="bar-track"><i :style="{width:rate(r.re)+'%',background:'#18a78e'}"></i></div><strong>{{percent(r.re)}}</strong></div><div class="bar-row"><span>TCP</span><div class="bar-track"><i :style="{width:rate(r.tcp)+'%',background:'#8c68cf'}"></i></div><strong>{{percent(r.tcp)}}</strong></div></div></div></div>
            <p class="tip">相同数据集不一定意味着公平对比：请同时核对样本范围、Prompt、模型输入和评测器版本。</p>
          </div>
        </template>
      </section>
    </main>
    <div v-if="selectedResult" class="modal-backdrop" @click.self="selectedResult=null"><div class="modal"><div class="modal-head"><div><span class="eyebrow">SAMPLE DETAIL</span><h2>{{selectedResult.sample_key}} <span class="tag" :class="selectedResult.status">{{statusText(selectedResult.status)}}</span></h2></div><button class="modal-close" @click="selectedResult=null">×</button></div>
      <div class="modal-meta">
        <span>推理 <b>{{selectedResult.inference_ms||0}} ms</b></span>
        <span>编译 <b>{{selectedResult.compile_ms||0}} ms</b></span>
        <span>测试 <b>{{selectedResult.test_ms||0}} ms</b></span>
        <span>沙箱 <b>{{selectedResult.sandbox_ms||0}} ms</b></span>
        <span>总耗时 <b>{{selectedResult.latency_ms||0}} ms</b></span>
        <span>Token <b>{{selectedResult.prompt_tokens||0}} / {{selectedResult.completion_tokens||0}}</b></span>
        <span>Harness <b>{{selectedResult.harness_mode==='program'?'完整程序（Test 0 = 整套一次执行）':'独立用例'}}</b></span>
      </div>
      <div class="modal-meta" v-if="selectedResult.submitted_at || selectedResult.compile_started_at">
        <span v-if="selectedResult.submitted_at">提交 <b>{{dateTime(selectedResult.submitted_at)}}</b></span>
        <span v-if="selectedResult.generated_at">生成完成 <b>{{dateTime(selectedResult.generated_at)}}</b></span>
        <span v-if="selectedResult.first_token_at">首 token <b>{{dateTime(selectedResult.first_token_at)}}</b></span>
        <span v-else>首 token <b>N/A（非流式接口）</b></span>
        <span v-if="selectedResult.compile_started_at">编译开始 <b>{{dateTime(selectedResult.compile_started_at)}}</b></span>
        <span v-if="selectedResult.compile_finished_at">编译结束 <b>{{dateTime(selectedResult.compile_finished_at)}}</b></span>
      </div>
      <p v-if="selectedResult.harness_note" class="tip">{{selectedResult.harness_note}}</p>
      <p v-if="selectedResult.failure_reason || selectedResult.error" class="danger-text">失败原因：{{selectedResult.failure_reason || selectedResult.error}}</p>
      <p v-if="selectedResult.generation_params && Object.keys(selectedResult.generation_params).length" class="tip">生成参数：max_tokens {{selectedResult.generation_params.max_tokens_requested ?? '—'}} → {{selectedResult.generation_params.max_tokens_effective ?? '—'}} · tokenizer {{selectedResult.generation_params.tokenizer || '—'}} · temperature {{selectedResult.generation_params.temperature ?? '—'}}</p>
      <div class="case-diagnostic">
        <div class="panel-head"><div><h3>官方 harness 结果 vs 逐用例诊断</h3><p>RE 始终以官方完整 harness 为准；TCP 仅使用可拆分的诊断用例。</p></div><div class="actions"><button class="ghost" @click="exportCases('json')">导出 JSON</button><button class="ghost" @click="exportCases('csv')">导出 CSV</button></div></div>
        <div v-if="!testCases" class="empty">正在加载逐用例结果…</div>
        <template v-else>
          <div class="case-summary">
            <span>官方 harness <b class="tag" :class="selectedResult.status">{{statusText(selectedResult.status)}}</b> · {{selectedResult.tests_passed}}/{{selectedResult.tests_total}}</span>
            <span>诊断状态 <b>{{testCases.summary.test_case_status || '—'}}</b></span>
            <span>用例 <b>{{testCases.summary.cases_passed}} / {{testCases.summary.cases_total}}</b>（声明 {{testCases.summary.cases_declared}}）</span>
            <span>用例覆盖 <b>{{testCases.summary.coverage === null ? 'N/A' : percent(testCases.summary.coverage)}}</b></span>
            <span>指标版本 <b>{{testCases.summary.metric_version || '—'}}</b></span>
          </div>
          <p v-if="diagnosticConsistent === false" class="notice error">诊断用例结果与官方 harness 不一致：RE 以官方结果为准；请查看诊断限制/转换风险，禁止静默覆盖。</p>
          <p v-if="!testCases.summary.cases_declared" class="tip">旧结果没有逐用例信息（或该 harness 不可拆分）：TCP=N/A，请重新运行诊断评测。不能把一次完整 harness 的执行当成 1 个逻辑用例。</p>
          <div v-if="testCases.summary.cases_declared" class="case-filters">
            <button v-for="f in [['all','全部'],['pass','通过'],['fail','失败'],['unsupported','无法诊断']]" :key="f[0]" class="ghost" :class="{active:caseFilter===f[0]}" @click="setCaseFilter(f[0])">{{f[1]}}</button>
            <span class="tip">共 {{testCases.total}} 条</span>
          </div>
          <div v-if="testCases.items.length" class="table-wrap"><table class="case-table"><thead><tr><th>#</th><th>case</th><th>输入 / 实际</th><th>预期</th><th>状态</th><th>诊断</th><th>日志</th></tr></thead><tbody>
            <tr v-for="c in testCases.items" :key="c.id">
              <td class="mono">{{c.case_index}}</td>
              <td class="mono">{{c.case_id}}<small v-if="c.case_source_span_json?.start_line"> · L{{c.case_source_span_json.start_line}}</small></td>
              <td class="case-value"><div>{{c.input_text || '—'}}</div><div class="case-actual">actual: {{c.actual_text || '—'}}</div></td>
              <td class="case-value">{{c.expected_text || '—'}}</td>
              <td><span class="tag" :class="caseTagClass(c.status)">{{statusText(c.status) || c.status}}</span><small v-if="c.error_type && c.error_type!=='NONE'"> {{c.error_type}}</small></td>
              <td class="case-value"><span>{{c.diagnostic_status || '—'}}</span><small v-if="c.comparison_rule"> · {{c.comparison_rule}}</small><small v-if="c.signal"> · signal {{c.signal}}</small><small v-if="c.exit_code!==null && c.exit_code!==undefined"> · exit {{c.exit_code}}</small></td>
              <td><details v-if="c.stdout || c.stderr"><summary>stdout/stderr</summary><pre v-if="c.stdout">[stdout]{{'\n'}}{{c.stdout}}</pre><pre v-if="c.stderr">[stderr]{{'\n'}}{{c.stderr}}</pre><small v-if="c.stdout_truncated || c.stderr_truncated">（日志已截断）</small></details><span v-else class="tip">—</span></td>
            </tr>
          </tbody></table></div>
          <div v-else class="empty">没有符合筛选条件的用例。未执行/无法拆分的用例会显式标记，不会伪造 PASS/FAIL。</div>
          <div v-if="testCases.total > casePageSize" class="case-pager"><button class="ghost" :disabled="casePage<=1" @click="changeCasePage(-1)">上一页</button><span>第 {{casePage}} / {{caseMaxPage}} 页</span><button class="ghost" :disabled="casePage>=caseMaxPage" @click="changeCasePage(1)">下一页</button></div>
        </template>
      </div>
      <div class="code-grid"><div><div class="code-title">输入汇编</div><pre>{{selectedResult.assembly}}</pre></div><div><div class="code-title">模型原始输出</div><pre>{{selectedResult.model_output || '(无输出)'}}</pre></div><div><div class="code-title">提取后的 C 代码</div><pre>{{selectedResult.extracted_source || '(空)'}}</pre></div><div><div class="code-title">参考 C 源代码（不提供给模型）</div><pre>{{selectedResult.reference_source}}</pre></div></div>
      <details v-if="selectedResult.prompt_text" class="prompt-details"><summary>实际发送给模型的 Prompt</summary><pre>{{selectedResult.prompt_text}}</pre></details>
      <div v-if="selectedResult.tests_detail?.length" class="tests-detail"><h3>逐用例执行</h3>
        <div v-for="t in selectedResult.tests_detail" :key="t.index" class="test-row">
          <div class="test-head"><strong>Test {{t.index}}</strong><span class="tag" :class="testTagClass(t.status)">{{testStatusText(t.status)}}</span><span>exit={{t.exit_code ?? '—'}} · signal={{t.signal_name || '—'}} · {{t.duration_ms}} ms</span><span v-if="t.timed_out" class="tag TIMEOUT">超时</span></div>
          <p v-if="t.reason" class="tip">{{t.reason}}</p>
          <details v-if="t.stdout || t.stderr"><summary>stdout / stderr</summary><pre v-if="t.stdout">[stdout]{{'\n'}}{{t.stdout}}</pre><pre v-if="t.stderr">[stderr]{{'\n'}}{{t.stderr}}</pre></details>
        </div>
      </div>
      <div class="logs"><div><h3>编译日志（stdout/stderr）</h3><pre>{{selectedResult.compile_log || '(无编译输出)'}}</pre></div><div><h3>测试汇总日志</h3><pre>{{selectedResult.run_log || '(无运行日志)'}}</pre></div></div>
      <p class="tip">测试通过 {{selectedResult.tests_passed}} / {{selectedResult.tests_total}}。exit=134 且 signal=SIGABRT 只有在捕获到断言失败信息时才按 WRONG_ANSWER 计；abort/堆损坏/段错误归为 RUNTIME_ERROR。</p>
    </div></div>
  </div>
</template>
