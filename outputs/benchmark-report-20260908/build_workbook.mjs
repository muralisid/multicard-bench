import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const dir = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(await fs.readFile(path.join(dir, 'report-data.json'), 'utf8'));
const wb = Workbook.create();
const previews = path.join(dir, 'previews');
await fs.mkdir(previews, { recursive: true });
const score = '0.00';
const usd = '$0.0000';
const integer = '#,##0';
const decimal = '#,##0.00';
const delta = '+0.00;-0.00;0.00';
const col = n => { let s = ''; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
const c = (key, title, width = 15, fmt = null, scale = 1) => ({ key, title, width, fmt, scale });
const s = (key, title = key) => c(key, title, 12, score, 100);
const descriptors = [];
const verification = [];

function sheet(name, title, note, rows, columns, options = {}) {
  const sh = wb.worksheets.add(name);
  const last = col(columns.length - 1), end = rows.length + 5;
  sh.showGridLines = false;
  sh.getRange(`A1:${last}${end}`).format = { font: { name: 'Arial', size: 11, color: '#243447' }, verticalAlignment: 'center', rowHeight: options.rowHeight || 30 };
  sh.getRange('A1').values = [[title]];
  sh.getRange('A1').format.font = { name: 'Arial', size: 17, bold: true, color: '#17324D' };
  sh.getRange('A1').format.rowHeight = 32;
  sh.getRange('A2:A3').format.rowHeight = 22;
  sh.mergeCells(`A2:${last}3`);
  sh.getRange('A2').values = [[note]];
  sh.getRange(`A2:${last}3`).format = { wrapText: true, font: { name: 'Arial', size: 10, color: '#526477' } };
  sh.getRange('A4').format.rowHeight = 10;
  const matrix = [columns.map(x => x.title), ...rows.map(r => columns.map(x => r[x.key] === undefined || r[x.key] === null ? null : typeof r[x.key] === 'number' ? r[x.key] * x.scale : r[x.key]))];
  sh.getRange(`A5:${last}${end}`).values = matrix;
  const tbl = sh.tables.add(`A5:${last}${end}`, true, name.replaceAll(' ', '') + 'Table');
  tbl.style = 'TableStyleLight9';
  sh.getRange(`A5:${last}5`).format = { fill: '#17324D', font: { name: 'Arial', size: 11, bold: true, color: '#FFFFFF' }, wrapText: true, rowHeight: 48 };
  for (let i = 0; i < columns.length; i++) {
    const x = columns[i], letter = col(i);
    sh.getRange(`${letter}1:${letter}${end}`).format.columnWidth = x.width;
    const range = sh.getRange(`${letter}6:${letter}${end}`);
    range.format.wrapText = !x.fmt;
    if (x.fmt) range.format.numberFormat = x.fmt;
  }
  sh.freezePanes.freezeRows(5);
  if (columns.length > 4) sh.freezePanes.freezeColumns(options.freezeColumns || 1);
  descriptors.push({ name, columns, rows, sh, end, last });
  return sh;
}

const summary = sheet('Summary', 'TG-VGRAG benchmark results',
  `Generated ${data.generated_at}. A: 20,142 questions, 19 variants. B/C/D: ${data.audit.BCD_complete_datasets.length}/19 complete. Scores /100; F1 and BEAM rubric mean are not percent correct. Compare B/C/D with A matched. C/D include ${data.mechanisms.reduce((n,r) => n+r.reversed_intervals,0)} reversed validity intervals; see Hypothesis findings. BEAM 10M deferred.`, data.summary,
  [c('test', 'Test', 34), c('metric', 'Metric', 19), c('full_n', 'A full N', 12, integer), s('A_full', 'A full score'), c('paired_n', 'Matched N', 12, integer), s('A', 'A matched'), s('B'), s('C'), s('D'), c('D_minus_A', 'D minus A', 13, delta, 100), c('best', 'Best matched', 15), c('scope', 'BCD scope', 13)], { rowHeight: 32 });
data.summary.forEach((r, i) => {
  if (r.D !== null) summary.getRange(`J${i + 6}`).formulas = [[`=I${i + 6}-F${i + 6}`]];
  if (r.status === 'complete') for (const arm of r.best.split('/')) {
    const j = 'ABCD'.indexOf(arm) + 5;
    summary.getRange(`${col(j)}${i + 6}`).format = { fill: '#E0EFE9', font: { name: 'Arial', bold: true, color: '#175842' } };
  }
});
summary.getRange(`J6:J${data.summary.length + 5}`).conditionalFormats.add('cellIs', { operator: 'lessThan', formula: 0, format: { font: { color: '#A52C3A' } } });

sheet('Test settings', 'Controlled variables', 'A is the baseline. B adds cached facts, C adds temporal validity, D adds graph ranking. Source: VERSION-BCD.md and frozen run.json.', data.settings,
  [c('variable', 'Variable', 30), c('value', 'Setting', 100), c('purpose', 'Purpose', 85)], { rowHeight: 44 });

sheet('Hypothesis findings', 'What the experiments establish', 'Measured changes use identical questions within each dataset. These are findings about this implementation; they do not establish industry superiority or independently validate each graph channel.', data.findings,
  [c('hypothesis', 'Hypothesis', 33), c('measurement', 'Observed result', 102), c('interpretation', 'What it means', 105)], { rowHeight: 88 });

const qcols = [c('test', 'Test', 34), c('arm', 'Arm', 11), c('n', 'Questions', 12, integer), c('metric', 'Metric', 18), s('score', 'Score /100'),
  c('correct', 'Correct', 12, integer), c('correct_basis', 'Correct basis', 18), s('exact_match', 'Exact match /100'),
  c('setup_usd', 'Fact setup USD', 16, usd), c('answer_usd', 'Answer USD', 16, usd), c('judge_usd', 'Judge USD', 16, usd),
  c('serving_usd', 'Serving USD', 16, usd), c('per_question_usd', 'Serving / Q', 16, usd), c('per_correct_usd', 'Serving / correct', 19, usd)];
for (const [name, rows] of [['A full quality', data.quality.filter(r => r.arm === 'A full')], ['Matched quality cost', data.quality.filter(r => r.arm !== 'A full')]]) {
  const sh = sheet(name, name === 'A full quality' ? 'Full Version A quality and cost' : 'Matched A/B/C/D quality and cost',
    'USD from recorded tokens and configured prices. Serving = answers + fact setup; judges separate. Shared setup shown per standalone arm, paid once across B/C/D. Correct means exact match for F1 datasets; BEAM has no binary count. Local compute unpriced.', rows, qcols);
  rows.forEach((r, i) => {
    const n = i + 6;
    sh.getRange(`L${n}:N${n}`).formulas = [[`=I${n}+J${n}`, `=L${n}/C${n}`, `=IF(OR(F${n}="",F${n}=0),"",L${n}/F${n})`]];
    if (r.exact_match !== null) sh.getRange(`H${n}`).formulas = [[`=F${n}/C${n}*100`]];
  });
  const computed = sh.getRange(`L6:N${rows.length + 5}`).values;
  rows.forEach((r, i) => {
    [r.serving_usd, r.per_question_usd, r.per_correct_usd].forEach((expected, j) => {
      const actual = computed[i][j];
      if (expected === null) {
        if (actual !== '' && actual !== null) throw new Error(`Expected blank: ${name}, row ${i + 6}`);
      } else if (typeof actual !== 'number' || Math.abs(actual - expected) > 1e-9) {
        throw new Error(`Formula differs from independent recomputation: ${name}, row ${i + 6}, column ${j}`);
      }
    });
  });
  verification.push({ sheet: name, costFormulaRowsChecked: rows.length });
}

const paired = sheet('Paired changes', 'Paired score changes', 'Deltas in points /100. Bootstrap: 2,000 resamples of histories, seed 13. Blank CI means one shared corpus. Exploratory intervals, no multiple-comparison adjustment. Wins/losses/ties count per-question score changes.', data.paired,
  [c('test', 'Test', 34), c('comparison', 'Comparison', 14), c('n', 'N', 10, integer), c('histories', 'Histories', 12, integer), c('metric', 'Metric', 18), s('reference', 'Reference'), s('new', 'New'), c('delta', 'Delta', 12, delta, 100), c('ci_low', '95% low', 12, delta, 100), c('ci_high', '95% high', 12, delta, 100), c('wins', 'Wins', 10, integer), c('losses', 'Losses', 10, integer), c('ties', 'Ties', 10, integer)]);
data.paired.forEach((r, i) => paired.getRange(`H${i + 6}`).formulas = [[`=G${i + 6}-F${i + 6}`]]);

sheet('Question types', 'Results by question type', 'Scores /100. Full A and matched A use different denominators. Blank values indicate no sampled questions or unfinished results. Source: question metadata and saved answer scores.', data.by_type,
  [c('test', 'Test', 32), c('type', 'Question type', 34), c('metric', 'Metric', 18), c('full_n', 'Full N', 12, integer), s('A_full', 'A full'), c('sample_n', 'Matched N', 12, integer), ...'ABCD'.split('').map(a => s(a))], { freezeColumns: 2 });

sheet('Mechanisms', 'Fact and temporal interventions', 'Counts summed across prepared histories. Fallback counts are pieces, not whole units. Superseded facts counted once per stored history. B/C/D changed compare B-A, C-B and D-C respectively. Source: groups/*.json and contexts.jsonl.', data.mechanisms,
  [c('test', 'Test', 34), c('prepared_queries', 'Prepared Q', 12, integer), c('extraction_pieces', 'Extracted pieces', 14, integer), c('fallback_pieces', 'Fallback pieces', 14, integer), c('rejected_facts', 'Rejected facts', 14, integer), c('facts', 'Stored facts', 14, integer), c('compressed_units', 'Compressed units', 16, integer), c('superseded_facts', 'Superseded facts', 16, integer), c('queries_hiding_facts', 'Q hiding facts', 14, integer), c('B_changed', 'B changed Q', 14, integer), c('C_changed', 'C changed Q', 14, integer), c('D_changed', 'D changed Q', 14, integer), c('reversed_intervals', 'End before start', 17, integer)]);

sheet('Graph structure', 'Discovered evidence graph', 'D uses separate topic and community indexes plus typed entity/fact/source relationships. These counts cover discovered evidence, not an exhaustive graph of the corpus. Setup cost is shared across B/C/D.', data.mechanisms,
  [c('test', 'Test', 34), c('groups', 'Histories', 14, integer), c('topics', 'Topics', 14, integer), c('communities', 'Communities', 16, integer), c('entities', 'Entities', 14, integer), c('edges', 'Typed edges', 16, integer), c('setup_usd', 'Shared setup USD', 20, usd)]);

sheet('Context and indexing', 'Corpus size and recorded build time', 'Tokens use MiniLM WordPiece, which can differ from published nominal size labels. Index time sums saved group build/load durations, excluding interrupted lost work. BCD preparation includes extraction and graph work. These are not campaign elapsed time or CPU core-hours.', data.runtime,
  [c('test', 'Test', 34), c('source_groups', 'Groups', 11, integer), c('source_documents', 'Source units', 15, integer), c('source_tokens', 'Corpus tokens', 18, integer), c('mean_history_tokens', 'Mean tokens / group', 19, integer), c('index_minutes', 'A index minutes', 17, decimal), c('index_mb', 'Index MB', 14, decimal), c('prep_minutes', 'BCD prep minutes', 19, decimal), c('graph_seconds', 'Graph seconds', 17, decimal), c('retrieval_p50_ms', 'A retrieval p50 ms', 19, decimal), c('retrieval_p95_ms', 'A retrieval p95 ms', 19, decimal)]);

sheet('Tokens and latency', 'Reader token use and latency', 'Reader p50/p95 exclude cached calls and omit retrieval, fact setup and judging. Input/output use provider tokens; evidence uses MiniLM WordPiece. Empty answers score zero. Source: saved per-question answers.', data.quality,
  [c('test', 'Test', 34), c('arm', 'Arm', 11), c('n', 'Questions', 12, integer), c('input_tokens', 'Input tokens', 18, integer), c('output_tokens', 'Output tokens', 18, integer), c('mean_context_tokens', 'Mean evidence tokens', 20, decimal), c('cached', 'Cached Q', 12, integer), c('empty', 'Empty Q', 12, integer), c('reader_p50_s', 'Reader p50 seconds', 20, decimal), c('reader_p95_s', 'Reader p95 seconds', 20, decimal), c('max_context_tokens', 'Max evidence tokens', 20, integer), c('context_over_budget', 'Over 4K questions', 18, integer)]);

sheet('A retrieval', 'Version A retrieval diagnostics', 'Scores /100. MultiHop-RAG uses source/article evidence coverage, not the paper fact Hit@K. Blank means no applicable metric or gold evidence. Gold Q is the denominator; incomplete evidence exclusions are recorded separately.', data.retrieval,
  [c('test', 'Test', 34), c('gold_n', 'Gold Q', 12, integer), c('excluded', 'Incomplete excluded', 18, integer), s('recall_at_5', 'Recall@5'), s('recall_at_10', 'Recall@10'), s('recall_at_100', 'Recall@100'), c('evidence_recall_at_budget', 'Recall in 4K', 17, score, 100), c('all_evidence_at_budget', 'All evidence in 4K', 19, score, 100)]);

const spend = sheet('API spend', 'Completed API requests and reservations', 'Each ledger request counted once. Costs are token-based estimates, not a reconciled invoice. BCD includes $0.38770325 preflight cache imports. Earlier Part 1 work excluded. Local compute, storage and operations unpriced. Source: calls.sqlite.', data.budget,
  [c('campaign', 'Campaign', 14), c('role', 'Role', 14), c('status', 'Status', 14), c('model', 'Model', 30), c('calls', 'Calls', 14, integer), c('usd', 'USD', 18, usd), c('input_tokens', 'Input tokens', 20, integer), c('output_tokens', 'Output tokens', 20, integer)]);
const spendEnd = data.budget.length + 5, totalRow = spendEnd + 2;
spend.getRange(`A${totalRow}`).values = [['Completed spend']];
spend.getRange(`A${totalRow + 1}`).values = [['Reserved / unresolved']];
spend.getRange(`A${totalRow}:D${totalRow + 1}`).format.font = { name: 'Arial', size: 11, bold: true };
spend.getRange(`F${totalRow}:F${totalRow + 1}`).formulas = [[`=SUMIF(C6:C${spendEnd},"done",F6:F${spendEnd})`], [`=SUM(F6:F${spendEnd})-F${totalRow}`]];
spend.getRange(`F${totalRow}:F${totalRow + 1}`).format.numberFormat = usd;

sheet('Notes and limitations', 'Findings, scope and remaining work', 'Read alongside the summary before drawing quality or cost conclusions. Evidence: saved experiment protocols, metrics, API ledgers and validation artifacts.', data.notes,
  [c('topic', 'Topic', 28), c('note', 'Finding or limitation', 170)], { rowHeight: 62 });

function sourceText(value) {
  try {
    const x = JSON.parse(value);
    if (Array.isArray(x)) return x.map(v => v.url || v.source_url || JSON.stringify(v)).join('\n');
    return x.url || x.dataset || value;
  } catch { return value; }
}
sheet('Dataset sources', 'Dataset provenance and protocol differences', 'Source manifests pin revisions and file hashes. The companion AUDIT.json records the exact local artifacts used for this report. Protocol differences prevent treating published provider scores as a controlled head-to-head comparison.', data.sources.map(r => ({ ...r, source: sourceText(r.source) })),
  [c('test', 'Test', 32), c('variant', 'Variant evaluated', 58), c('source', 'Source URL or dataset', 82), c('protocol', 'Protocol differences', 115)], { rowHeight: 125 });

const inspection = await wb.inspect({ kind: 'table', range: 'Summary!A5:L9', include: 'values,formulas', tableMaxRows: 5, tableMaxCols: 12, maxChars: 4500 });
await fs.writeFile(path.join(dir, 'workbook-inspection.ndjson'), inspection.ndjson);
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 100 }, summary: 'Formula error scan' });
await fs.writeFile(path.join(dir, 'workbook-errors.ndjson'), errors.ndjson);
console.log(errors.ndjson);
for (const d of descriptors) {
  const final = process.argv.includes('--final');
  if (final && ['Test settings', 'A full quality'].includes(d.name)) continue;
  let start = 1;
  let end = ['Summary', 'API spend'].includes(d.name) ? d.end + (d.name === 'API spend' ? 3 : 0) : Math.min(d.end, d.name === 'Dataset sources' ? 11 : 14);
  if (final && ['Matched quality cost', 'Paired changes', 'Question types', 'Mechanisms', 'Graph structure', 'Context and indexing', 'A retrieval', 'Dataset sources'].includes(d.name)) {
    start = Math.max(5, d.end - (d.name === 'Question types' ? 19 : d.name === 'Paired changes' ? 14 : d.name === 'Matched quality cost' ? 11 : 3));
    end = d.end;
  }
  const preview = await wb.render({ sheetName: d.name, range: `A${start}:${d.last}${end}`, scale: 1, format: 'png' });
  await fs.writeFile(path.join(previews, `${d.name.replaceAll(' ', '_')}.png`), new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(dir, 'workbook-verification.json'), JSON.stringify({ generated: data.generated_at, checks: verification }, null, 2));
const exported = await SpreadsheetFile.exportXlsx(wb);
await exported.save(path.join(dir, 'TG-VGRAG-results.xlsx'));
console.log(JSON.stringify({ output: path.join(dir, 'TG-VGRAG-results.xlsx'), sheets: descriptors.length, generated: data.generated_at }));
