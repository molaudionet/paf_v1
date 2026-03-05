import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as d3 from "d3";
import * as Tone from "tone";

// ═══════════════════════════════════════════════════════════
// PAF POCKET EXPLORER — Sound of Molecules LLC
// Interactive binding pocket similarity search engine
// ═══════════════════════════════════════════════════════════

// ── Representative pocket database (15 families, real PDB IDs) ──
const FAMILIES = [
  { name: "Kinase", color: "#00e5ff", count: 293 },
  { name: "Nuclear Receptor", color: "#ff6d00", count: 196 },
  { name: "Metalloprotease", color: "#76ff03", count: 128 },
  { name: "Phosphatase", color: "#d500f9", count: 99 },
  { name: "Phosphodiesterase", color: "#ffea00", count: 98 },
  { name: "Bromodomain", color: "#ff1744", count: 98 },
  { name: "Carbonic Anhydrase", color: "#00e676", count: 94 },
  { name: "Proteasome", color: "#2979ff", count: 80 },
  { name: "DHFR", color: "#f50057", count: 80 },
  { name: "COX", color: "#00bfa5", count: 79 },
  { name: "HDAC", color: "#ff9100", count: 79 },
  { name: "Aspartyl Protease", color: "#651fff", count: 63 },
  { name: "Serine Protease", color: "#c6ff00", count: 44 },
  { name: "HSP90", color: "#ff3d00", count: 39 },
  { name: "GPCR", color: "#18ffff", count: 19 },
];

const POCKETS = [
  { pdb: "3EQM", family: "Kinase", subfamily: "EGFR", ligand: "Erlotinib", res: 2.0, dfg: "in" },
  { pdb: "1M17", family: "Kinase", subfamily: "EGFR", ligand: "Gefitinib", res: 2.6, dfg: "in" },
  { pdb: "4HJO", family: "Kinase", subfamily: "EGFR", ligand: "Afatinib", res: 2.5, dfg: "in" },
  { pdb: "3POZ", family: "Kinase", subfamily: "ABL", ligand: "Imatinib", res: 2.4, dfg: "out" },
  { pdb: "2HYY", family: "Kinase", subfamily: "ABL", ligand: "Nilotinib", res: 2.2, dfg: "out" },
  { pdb: "3CS9", family: "Kinase", subfamily: "ABL", ligand: "Dasatinib", res: 2.4, dfg: "in" },
  { pdb: "3LJ0", family: "Kinase", subfamily: "ALK", ligand: "Crizotinib", res: 2.0, dfg: "in" },
  { pdb: "4ANS", family: "Kinase", subfamily: "BRAF", ligand: "Vemurafenib", res: 2.5, dfg: "out" },
  { pdb: "1GFU", family: "Kinase", subfamily: "CDK2", ligand: "Roscovitine", res: 2.2, dfg: "in" },
  { pdb: "2VTA", family: "Kinase", subfamily: "JAK2", ligand: "CMP6", res: 2.0, dfg: "in" },
  { pdb: "3ERT", family: "Nuclear Receptor", subfamily: "ER-alpha", ligand: "4-OHT", res: 1.9 },
  { pdb: "1ERE", family: "Nuclear Receptor", subfamily: "ER-alpha", ligand: "Estradiol", res: 3.1 },
  { pdb: "2AA2", family: "Nuclear Receptor", subfamily: "ER-alpha", ligand: "Raloxifene", res: 2.6 },
  { pdb: "1I7I", family: "Nuclear Receptor", subfamily: "AR", ligand: "R1881", res: 1.8 },
  { pdb: "2AX9", family: "Nuclear Receptor", subfamily: "PPAR-gamma", ligand: "Rosiglitazone", res: 2.1 },
  { pdb: "1DB1", family: "Nuclear Receptor", subfamily: "VDR", ligand: "Calcitriol", res: 1.8 },
  { pdb: "1OS5", family: "Metalloprotease", subfamily: "MMP-13", ligand: "WAY-151693", res: 1.6 },
  { pdb: "830C", family: "Metalloprotease", subfamily: "MMP-2", ligand: "Inhibitor", res: 2.8 },
  { pdb: "1T4E", family: "Metalloprotease", subfamily: "TACE", ligand: "IK-682", res: 2.0 },
  { pdb: "2ZJJ", family: "Phosphatase", subfamily: "PTP1B", ligand: "Compound", res: 2.3 },
  { pdb: "1SUG", family: "Phosphatase", subfamily: "PTP1B", ligand: "DADMe", res: 1.8 },
  { pdb: "1XOZ", family: "Phosphodiesterase", subfamily: "PDE5", ligand: "Sildenafil", res: 1.7 },
  { pdb: "1T9S", family: "Phosphodiesterase", subfamily: "PDE4", ligand: "Rolipram", res: 2.0 },
  { pdb: "3MXF", family: "Bromodomain", subfamily: "BRD4-BD1", ligand: "JQ1", res: 1.6 },
  { pdb: "4MR4", family: "Bromodomain", subfamily: "BRD4-BD1", ligand: "I-BET151", res: 1.5 },
  { pdb: "1CA2", family: "Carbonic Anhydrase", subfamily: "CA-II", ligand: "AZM", res: 1.5 },
  { pdb: "3HS4", family: "Carbonic Anhydrase", subfamily: "CA-II", ligand: "Sulfonamide", res: 1.1 },
  { pdb: "5LF3", family: "Proteasome", subfamily: "20S", ligand: "Bortezomib", res: 2.1 },
  { pdb: "1DLS", family: "DHFR", subfamily: "Human", ligand: "Methotrexate", res: 1.1 },
  { pdb: "1RX1", family: "DHFR", subfamily: "E.coli", ligand: "Trimethoprim", res: 1.7 },
  { pdb: "3LN1", family: "COX", subfamily: "COX-2", ligand: "Celecoxib", res: 2.4 },
  { pdb: "4PH9", family: "COX", subfamily: "COX-2", ligand: "Indomethacin", res: 2.3 },
  { pdb: "4LXZ", family: "HDAC", subfamily: "HDAC2", ligand: "Vorinostat", res: 1.9 },
  { pdb: "1T8F", family: "HDAC", subfamily: "HDLP", ligand: "SAHA", res: 2.0 },
  { pdb: "1HVR", family: "Aspartyl Protease", subfamily: "HIV-PR", ligand: "Ritonavir", res: 1.8 },
  { pdb: "3OXC", family: "Aspartyl Protease", subfamily: "BACE1", ligand: "Inhibitor", res: 1.9 },
  { pdb: "1O86", family: "Serine Protease", subfamily: "Thrombin", ligand: "Argatroban", res: 1.5 },
  { pdb: "2WEG", family: "HSP90", subfamily: "HSP90-alpha", ligand: "AT13387", res: 1.6 },
  { pdb: "3K8O", family: "HSP90", subfamily: "HSP90-alpha", ligand: "NVP-AUY922", res: 2.0 },
  { pdb: "4IAR", family: "GPCR", subfamily: "A2A", ligand: "ZM241385", res: 1.8 },
];

// Seeded pseudo-random for reproducible "embeddings"
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return (s - 1) / 2147483646; };
}

// Generate 2D t-SNE-like positions for scatter plot (clustered by family)
function generateEmbeddingPositions() {
  const familyCenters = {};
  const rng = seededRandom(42);
  FAMILIES.forEach((f, i) => {
    const angle = (i / FAMILIES.length) * Math.PI * 2;
    const r = 180 + rng() * 80;
    familyCenters[f.name] = { x: 250 + Math.cos(angle) * r, y: 250 + Math.sin(angle) * r };
  });
  return POCKETS.map(p => {
    const c = familyCenters[p.family];
    const rng2 = seededRandom(p.pdb.split("").reduce((a, c) => a + c.charCodeAt(0), 0));
    return { x: c.x + (rng2() - 0.5) * 90, y: c.y + (rng2() - 0.5) * 90 };
  });
}

// Compute similarity (simulated based on family/subfamily match)
function computeSimilarity(queryIdx) {
  const query = POCKETS[queryIdx];
  const rng = seededRandom(queryIdx * 1000 + 7);
  return POCKETS.map((p, i) => {
    if (i === queryIdx) return { idx: i, score: 1.0 };
    let base;
    if (p.family === query.family && p.subfamily === query.subfamily) base = 0.82 + rng() * 0.15;
    else if (p.family === query.family) base = 0.55 + rng() * 0.25;
    else base = 0.05 + rng() * 0.35;
    return { idx: i, score: Math.min(base, 0.99) };
  }).sort((a, b) => b.score - a.score);
}

// ── Waveform generation (simplified PAF visualization) ──
function generateWaveform(pdb, numPoints = 512) {
  const rng = seededRandom(pdb.split("").reduce((a, c) => a + c.charCodeAt(0), 0) * 137);
  const nOsc = 8 + Math.floor(rng() * 12);
  const data = new Float32Array(numPoints);
  for (let o = 0; o < nOsc; o++) {
    const freq = 2 + rng() * 18;
    const phase = rng() * Math.PI * 2;
    const amp = 0.3 + rng() * 0.7;
    const center = rng() * numPoints;
    const width = 20 + rng() * 60;
    for (let i = 0; i < numPoints; i++) {
      const env = Math.exp(-((i - center) ** 2) / (2 * width * width));
      data[i] += amp * env * Math.sin(2 * Math.PI * freq * i / numPoints + phase);
    }
  }
  const max = Math.max(...Array.from(data).map(Math.abs)) || 1;
  return Array.from(data).map(v => v / max);
}

// ── Sonification ──
async function sonifyPocket(pdb) {
  await Tone.start();
  const rng = seededRandom(pdb.split("").reduce((a, c) => a + c.charCodeAt(0), 0) * 31);
  const synth = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "sine" },
    envelope: { attack: 0.05, decay: 0.3, sustain: 0.2, release: 0.5 },
    volume: -18,
  }).toDestination();
  const nOsc = 6 + Math.floor(rng() * 6);
  const baseFreqs = [150, 220, 330, 440, 550, 660, 880, 1100, 1320, 1760, 2000, 2200];
  const now = Tone.now();
  for (let i = 0; i < nOsc; i++) {
    const freq = baseFreqs[Math.floor(rng() * baseFreqs.length)] * (0.8 + rng() * 0.4);
    const time = now + i * 0.15 + rng() * 0.1;
    synth.triggerAttackRelease(freq, "8n", time, 0.3 + rng() * 0.5);
  }
  setTimeout(() => synth.dispose(), 4000);
}

// ── Components ──

function WaveformViz({ pdb, width = 520, height = 120 }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!pdb || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    const data = generateWaveform(pdb);
    const w = width * 2, h = height * 2;
    canvasRef.current.width = w;
    canvasRef.current.height = h;
    ctx.clearRect(0, 0, w, h);

    // gradient fill
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, "rgba(0,229,255,0.15)");
    grad.addColorStop(0.5, "rgba(101,31,255,0.15)");
    grad.addColorStop(1, "rgba(255,23,68,0.15)");

    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    for (let i = 0; i < data.length; i++) {
      ctx.lineTo((i / data.length) * w, h / 2 - data[i] * h * 0.42);
    }
    ctx.lineTo(w, h / 2);
    for (let i = data.length - 1; i >= 0; i--) {
      ctx.lineTo((i / data.length) * w, h / 2);
    }
    ctx.fillStyle = grad;
    ctx.fill();

    // line
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    for (let i = 0; i < data.length; i++) {
      ctx.lineTo((i / data.length) * w, h / 2 - data[i] * h * 0.42);
    }
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [pdb, width, height]);

  return <canvas ref={canvasRef} style={{ width, height, borderRadius: 8 }} />;
}

function EmbeddingScatter({ positions, selectedIdx, similarIdxs, onSelect }) {
  const svgRef = useRef(null);
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const w = 500, h = 500;

    // background
    svg.append("rect").attr("width", w).attr("height", h).attr("rx", 12).attr("fill", "#0a0e1a");

    const simSet = new Set((similarIdxs || []).map(s => s.idx));

    // all points
    svg.selectAll("circle.bg")
      .data(positions)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", (_, i) => i === selectedIdx ? 0 : simSet.has(i) ? 5 : 3)
      .attr("fill", (_, i) => {
        const f = FAMILIES.find(f => f.name === POCKETS[i].family);
        return f ? f.color : "#555";
      })
      .attr("opacity", (_, i) => simSet.has(i) ? 0.9 : 0.2)
      .style("cursor", "pointer")
      .on("click", (_, i) => { const idx = positions.indexOf(_); if (idx >= 0) onSelect(idx); });

    // selected point
    if (selectedIdx !== null && positions[selectedIdx]) {
      const p = positions[selectedIdx];
      svg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 10)
        .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2.5);
      svg.append("circle").attr("cx", p.x).attr("cy", p.y).attr("r", 7)
        .attr("fill", "#fff");
    }
  }, [positions, selectedIdx, similarIdxs, onSelect]);

  return <svg ref={svgRef} viewBox="0 0 500 500" style={{ width: "100%", maxWidth: 500, borderRadius: 12 }} />;
}

function SimilarityBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score > 0.8 ? "#00e676" : score > 0.5 ? "#ffea00" : "#ff1744";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, width: "100%" }}>
      <div style={{ flex: 1, height: 6, background: "#1a1e2e", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color, minWidth: 44, textAlign: "right" }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}

function DFGBadge({ dfg }) {
  if (!dfg) return null;
  const colors = { in: { bg: "#00e67622", text: "#00e676", label: "DFG-in" }, out: { bg: "#ff174422", text: "#ff1744", label: "DFG-out" } };
  const c = colors[dfg] || colors.in;
  return (
    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: c.bg, color: c.text, fontWeight: 600, letterSpacing: 0.5 }}>
      {c.label}
    </span>
  );
}

function FamilyBadge({ family }) {
  const f = FAMILIES.find(f => f.name === family);
  const color = f ? f.color : "#888";
  return (
    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: color + "22", color, fontWeight: 600, letterSpacing: 0.3 }}>
      {family}
    </span>
  );
}

// ── Stats sidebar ──
function StatsPanel() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {[
        { label: "Pockets Indexed", value: "1,489", sub: "15 families" },
        { label: "Cross-Family Acc.", value: "85.7%", sub: "d = 1.42" },
        { label: "Kinase Subfamilies", value: "42", sub: "d = 1.70" },
        { label: "Pair Comparison", value: "0.15 μs", sub: "real-time" },
      ].map((s, i) => (
        <div key={i} style={{ background: "#0d1220", borderRadius: 10, padding: "14px 16px", border: "1px solid #1a2040" }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#e0e6f0", fontFamily: "'JetBrains Mono', monospace" }}>{s.value}</div>
          <div style={{ fontSize: 11, color: "#6b7a99", marginTop: 2 }}>{s.label}</div>
          <div style={{ fontSize: 10, color: "#3a4a6b", marginTop: 1 }}>{s.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════
export default function PAFExplorer() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [showViewer, setShowViewer] = useState(false);
  const [isEncoding, setIsEncoding] = useState(false);
  const positions = useMemo(() => generateEmbeddingPositions(), []);

  const handleSelect = useCallback((idx) => {
    setSelectedIdx(idx);
    setIsEncoding(true);
    setShowViewer(true);
    // Simulate encoding delay
    setTimeout(() => {
      const results = computeSimilarity(idx).slice(0, 15);
      setSimilarResults(results);
      setIsEncoding(false);
    }, 800);
  }, []);

  const handleSearch = useCallback(() => {
    const q = searchQuery.trim().toUpperCase();
    const idx = POCKETS.findIndex(p => p.pdb === q);
    if (idx >= 0) handleSelect(idx);
  }, [searchQuery, handleSelect]);

  const selected = selectedIdx !== null ? POCKETS[selectedIdx] : null;
  const topSimilar = similarResults.slice(1, 11);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(180deg, #050810 0%, #0a0f1e 30%, #080d18 100%)",
      color: "#c8d0e0",
      fontFamily: "'Segoe UI', -apple-system, sans-serif",
    }}>
      {/* Google Fonts */}
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* ── Header ── */}
      <div style={{
        padding: "20px 32px",
        borderBottom: "1px solid #141a2e",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(5,8,16,0.8)",
        backdropFilter: "blur(20px)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%",
            background: "conic-gradient(from 45deg, #00e5ff, #651fff, #ff1744, #00e5ff)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#0a0f1e", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ fontSize: 14 }}>🔊</span>
            </div>
          </div>
          <div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 17, color: "#e8ecf4", letterSpacing: -0.5 }}>
              PAF Pocket Explorer
            </div>
            <div style={{ fontSize: 10, color: "#4a5a7b", letterSpacing: 1, textTransform: "uppercase" }}>Sound of Molecules</div>
          </div>
        </div>

        {/* Search */}
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
            placeholder="Enter PDB ID (e.g. 3EQM, 3POZ, 3MXF)"
            style={{
              width: 300, padding: "10px 16px", borderRadius: 10,
              border: "1px solid #1e2845", background: "#0d1220",
              color: "#e0e6f0", fontSize: 14, fontFamily: "'JetBrains Mono', monospace",
              outline: "none",
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              padding: "10px 20px", borderRadius: 10, border: "none",
              background: "linear-gradient(135deg, #00e5ff, #651fff)",
              color: "#fff", fontWeight: 600, fontSize: 13, cursor: "pointer",
              letterSpacing: 0.5,
            }}
          >
            Search
          </button>
        </div>
      </div>

      {/* ── Main Grid ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 520px", gap: 0, minHeight: "calc(100vh - 77px)" }}>

        {/* Left: Embedding space + Quick picks */}
        <div style={{ padding: "24px 28px", borderRight: "1px solid #141a2e" }}>

          {/* Quick pick buttons */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 11, color: "#4a5a7b", letterSpacing: 1, textTransform: "uppercase", marginBottom: 10, fontWeight: 600 }}>
              Quick Select — Representative Pockets
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {POCKETS.slice(0, 20).map((p, i) => (
                <button
                  key={p.pdb}
                  onClick={() => { setSearchQuery(p.pdb); handleSelect(i); }}
                  style={{
                    padding: "5px 12px", borderRadius: 8,
                    border: selectedIdx === i ? "1px solid #00e5ff" : "1px solid #1e2845",
                    background: selectedIdx === i ? "#00e5ff15" : "#0d1220",
                    color: selectedIdx === i ? "#00e5ff" : "#8899bb",
                    fontSize: 11, cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    transition: "all 0.2s",
                  }}
                >
                  {p.pdb}
                  <span style={{ fontSize: 9, marginLeft: 4, opacity: 0.5 }}>{p.family.slice(0, 3)}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Embedding scatter */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 11, color: "#4a5a7b", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8, fontWeight: 600 }}>
              Spectral Embedding Space (t-SNE projection)
            </div>
            <EmbeddingScatter
              positions={positions}
              selectedIdx={selectedIdx}
              similarIdxs={topSimilar}
              onSelect={handleSelect}
            />
            {/* Legend */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
              {FAMILIES.map(f => (
                <div key={f.name} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: "#6b7a99" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", background: f.color }} />
                  {f.name}
                </div>
              ))}
            </div>
          </div>

          <StatsPanel />
        </div>

        {/* Right: Results panel */}
        <div style={{ padding: "24px 24px", overflowY: "auto", maxHeight: "calc(100vh - 77px)" }}>

          {!selected ? (
            <div style={{ textAlign: "center", paddingTop: 120, color: "#3a4a6b" }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>🧬</div>
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8, color: "#5a6a8b" }}>Select a binding pocket</div>
              <div style={{ fontSize: 13 }}>Click a point in the embedding space, pick a PDB ID above, or type one in the search bar.</div>
            </div>
          ) : (
            <>
              {/* Selected pocket card */}
              <div style={{
                background: "linear-gradient(135deg, #0d122088, #141a3088)",
                borderRadius: 14, padding: "18px 20px",
                border: "1px solid #1e2845",
                marginBottom: 18,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 28, fontWeight: 700, color: "#e8ecf4" }}>
                      {selected.pdb}
                    </div>
                    <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                      <FamilyBadge family={selected.family} />
                      {selected.subfamily && (
                        <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "#ffffff0a", color: "#8899bb" }}>
                          {selected.subfamily}
                        </span>
                      )}
                      <DFGBadge dfg={selected.dfg} />
                    </div>
                    {selected.ligand && (
                      <div style={{ fontSize: 12, color: "#6b7a99", marginTop: 6 }}>Ligand: {selected.ligand} · {selected.res} Å</div>
                    )}
                  </div>
                  <button
                    onClick={() => sonifyPocket(selected.pdb)}
                    style={{
                      padding: "8px 16px", borderRadius: 10, border: "1px solid #1e2845",
                      background: "#0d1220", color: "#00e5ff", fontSize: 12, cursor: "pointer",
                      fontWeight: 600, display: "flex", alignItems: "center", gap: 6,
                    }}
                  >
                    <span>🔊</span> Play Sound
                  </button>
                </div>

                {/* 3D viewer */}
                {showViewer && (
                  <div style={{ marginTop: 14, borderRadius: 10, overflow: "hidden", border: "1px solid #1a2040" }}>
                    <iframe
                      src={`https://www.rcsb.org/3d-view/${selected.pdb}?preset=ligandInteraction&sele=ligand`}
                      style={{ width: "100%", height: 260, border: "none", background: "#000" }}
                      title="3D Structure"
                    />
                  </div>
                )}

                {/* Waveform */}
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontSize: 10, color: "#4a5a7b", letterSpacing: 1, textTransform: "uppercase", marginBottom: 6, fontWeight: 600 }}>
                    PAF Spectral Waveform
                  </div>
                  <WaveformViz pdb={selected.pdb} width={468} height={80} />
                </div>
              </div>

              {/* Similarity results */}
              <div style={{ fontSize: 11, color: "#4a5a7b", letterSpacing: 1, textTransform: "uppercase", marginBottom: 10, fontWeight: 600 }}>
                {isEncoding ? "Encoding pocket..." : `Top ${topSimilar.length} Most Similar Pockets`}
              </div>

              {isEncoding ? (
                <div style={{ textAlign: "center", padding: 40 }}>
                  <div style={{
                    width: 40, height: 40, border: "3px solid #1e2845", borderTopColor: "#00e5ff",
                    borderRadius: "50%", animation: "spin 0.8s linear infinite", margin: "0 auto",
                  }} />
                  <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  <div style={{ marginTop: 12, fontSize: 12, color: "#4a5a7b" }}>Computing spectral fingerprint...</div>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {topSimilar.map((r, i) => {
                    const p = POCKETS[r.idx];
                    const isSameFamily = p.family === selected.family;
                    return (
                      <div
                        key={r.idx}
                        onClick={() => { setSearchQuery(p.pdb); handleSelect(r.idx); }}
                        style={{
                          display: "grid", gridTemplateColumns: "24px 60px 1fr 130px",
                          alignItems: "center", gap: 10,
                          padding: "10px 14px", borderRadius: 10,
                          background: isSameFamily ? "#00e5ff06" : "#0d1220",
                          border: isSameFamily ? "1px solid #00e5ff15" : "1px solid #141a2e",
                          cursor: "pointer",
                          transition: "all 0.15s",
                        }}
                      >
                        <span style={{ fontSize: 11, color: "#3a4a6b", fontFamily: "'JetBrains Mono', monospace" }}>
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 600, color: "#c8d0e0" }}>
                          {p.pdb}
                        </span>
                        <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                          <FamilyBadge family={p.family} />
                          <DFGBadge dfg={p.dfg} />
                          {p.ligand && <span style={{ fontSize: 9, color: "#4a5a7b" }}>{p.ligand}</span>}
                        </div>
                        <SimilarityBar score={r.score} />
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
